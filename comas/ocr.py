# Tesseract wrapper and caching
from __future__ import annotations
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Optional
from PIL import Image
log = logging.getLogger(__name__)
_pytesseract = None

# OCR is single threaded per image, so batches scale almost linearly with cores.
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)
FLUSH_EVERY = 25


def _get_pytesseract():
    global _pytesseract
    if _pytesseract is None:
        import pytesseract
        _pytesseract = pytesseract
    return _pytesseract


def _image_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _region_key(key: str, region: Optional[tuple[float, float]]) -> str:
    # Region crops are cached separately; a plain hash always means the full frame.
    if region is None:
        return key
    return f"{key}:r{region[0]:.2f}-{region[1]:.2f}"


def _ocr_image(path: Path, region: Optional[tuple[float, float]] = None) -> str:
    pytesseract = _get_pytesseract()
    try:
        img = Image.open(path)
        if region is not None:
            w, h = img.size
            top, bottom = region
            img = img.crop((0, int(h * top), w, int(h * bottom)))
        return pytesseract.image_to_string(img.convert("RGB")).strip()
    except Exception as e:
        log.warning("OCR failed on %s: %s", path, e)
        return ""


def _worker_init():
    # Tesseract multithreads through OpenMP. Left alone, every worker spawns its
    # own thread pool and they fight over the cores - measurably slower than
    # running sequentially. One thread per worker process is what makes the
    # parallel path a win.
    os.environ["OMP_THREAD_LIMIT"] = "1"


def _ocr_job(args):
    path, region = args
    return str(path), _ocr_image(Path(path), region)


class OCRCache:
    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._dirty = 0
        self._store: dict[str, str] = {}
        if self.cache_path.exists():
            try:
                self._store = json.loads(self.cache_path.read_text())
            except json.JSONDecodeError:
                log.warning("OCR cache at %s is corrupt; starting fresh.", self.cache_path)

    def _key(self, image_path: Path, region) -> str:
        return _region_key(_image_hash(image_path), region)

    def peek(self, image_path: Path, region=None) -> Optional[str]:
        with self._lock:
            return self._store.get(self._key(image_path, region))

    def get_or_extract(self, image_path: Path, region=None) -> str:
        key = self._key(image_path, region)
        with self._lock:
            cached = self._store.get(key)
        if cached is not None:
            return cached
        text = _ocr_image(Path(image_path), region)
        self._put(key, text)
        return text

    def extract_many(self, paths: Iterable[Path], region=None,
                     workers: Optional[int] = None, progress=None) -> None:
        """OCR a batch in parallel, filling the cache. Already-cached images are skipped."""
        paths = [Path(p) for p in paths]
        todo = []
        keys = {}
        for p in paths:
            key = self._key(p, region)
            keys[str(p)] = key
            with self._lock:
                hit = key in self._store
            if not hit:
                todo.append((str(p), region))

        done = len(paths) - len(todo)
        if progress:
            progress(done, len(paths))
        if not todo:
            return

        n = workers or DEFAULT_WORKERS
        if n <= 1 or len(todo) == 1:
            for job in todo:
                sp, text = _ocr_job(job)
                self._put(keys[sp], text)
                done += 1
                if progress:
                    progress(done, len(paths))
            self.flush()
            return

        try:
            with ProcessPoolExecutor(max_workers=n, initializer=_worker_init) as ex:
                for sp, text in ex.map(_ocr_job, todo):
                    self._put(keys[sp], text)
                    done += 1
                    if progress:
                        progress(done, len(paths))
        except Exception as e:
            # Fall back to in-process OCR rather than losing the whole batch.
            log.warning("Parallel OCR unavailable (%s); running sequentially.", e)
            for job in todo:
                sp, text = _ocr_job(job)
                self._put(keys[sp], text)
                done += 1
                if progress:
                    progress(done, len(paths))
        self.flush()

    def _put(self, key: str, text: str) -> None:
        with self._lock:
            self._store[key] = text
            self._dirty += 1
            due = self._dirty >= FLUSH_EVERY
        if due:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._store))
            os.replace(tmp, self.cache_path)
            self._dirty = 0

    def __len__(self) -> int:
        return len(self._store)
