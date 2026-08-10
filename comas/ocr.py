# Tesseract wrapper and caching 
from __future__ import annotations
import hashlib
import json
import logging
import threading
from pathlib import Path
from PIL import Image
log = logging.getLogger(__name__)
_pytesseract = None
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

class OCRCache:
    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._dirty = False
        self._store: dict[str, str] = {}
        if self.cache_path.exists():
            try:
                self._store = json.loads(self.cache_path.read_text())
            except json.JSONDecodeError:
                log.warning("OCR cache at %s is corrupt; starting fresh.", self.cache_path)
    def get_or_extract(self, image_path: Path) -> str:
        key = _image_hash(image_path)
        with self._lock:
            cached = self._store.get(key)
        if cached is not None:
            return cached
        text = self._run_ocr(image_path)
        with self._lock:
            self._store[key] = text
            self._dirty = True
        return text
    def _run_ocr(self, image_path: Path) -> str:
        pytesseract = _get_pytesseract()
        try:
            img = Image.open(image_path).convert("RGB")
            return pytesseract.image_to_string(img).strip()
        except Exception as e:
            log.warning("OCR failed on %s: %s", image_path, e)
            return ""
    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._store))
            self._dirty = False
    def __len__(self) -> int:
        return len(self._store)
