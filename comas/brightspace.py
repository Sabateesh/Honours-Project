from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .config import load_config
from .ocr import OCRCache

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

ON_QUIZ_KEYWORDS = [
    "brightspace",
    "d2l/lms",
    "submit quiz",
    "questions saved",
    "quiz information",
    "course quiz",
    "(1 point)",
    "elapsed",
]

OFF_TASK_KEYWORDS = {
    "chat.openai.com": "chatgpt",
    "chatgpt": "chatgpt",
    "stackoverflow": "stackoverflow",
    "stack overflow": "stackoverflow",
    "google.com/search": "google",
    "results for": "google",
    "youtube": "youtube",
    "reddit": "reddit",
    "wikipedia": "wikipedia",
    "discord": "discord",
    "chegg": "chegg",
    "coursehero": "coursehero",
    "quizlet": "quizlet",
}


@dataclass
class BrightspaceEvidence:
    path: str
    score: float
    verdict: str
    on_quiz_matches: list[str] = field(default_factory=list)
    off_task_matches: list[str] = field(default_factory=list)
    off_task_sites: list[str] = field(default_factory=list)
    is_ide: bool = False
    ocr_snippet: Optional[str] = None


def find_markers(text: str) -> tuple[list[str], list[str]]:
    if not text:
        return [], []
    low = text.lower()
    on = [k for k in ON_QUIZ_KEYWORDS if k in low]
    off = [k for k in OFF_TASK_KEYWORDS if k in low]
    return on, off



LEFT_QUIZ_SCORE = 1.0
ON_QUIZ_SCORE = 0.05


def score_markers(on: list[str], off: list[str]) -> tuple[float, str]:
    if on:
        return ON_QUIZ_SCORE, "on_quiz"
    return LEFT_QUIZ_SCORE, "left_quiz"


from .copilot import IDE_KEYWORDS, looks_like_ide  # noqa: E402,F401


TOP_STRIP = (0.0, 0.25)


class BrightspaceDetector:
    def __init__(self, ocr_cache: OCRCache, use_roi: bool = True):
        self.ocr_cache = ocr_cache
        self.use_roi = use_roi

    def _read(self, image_path: Path) -> tuple[str, list[str], list[str]]:
        if self.use_roi:
            text = self.ocr_cache.get_or_extract(image_path, region=TOP_STRIP)
            on, off = find_markers(text)
            if on:
                return text, on, off
        text = self.ocr_cache.get_or_extract(image_path)
        on, off = find_markers(text)
        return text, on, off

    def prefetch(self, paths, workers=None, progress=None) -> None:
        """Warm the cache in parallel: crops first, then full frames for
        anything the crop couldn't clear as on-quiz."""
        paths = [Path(p) for p in paths]
        if self.use_roi:
            self.ocr_cache.extract_many(paths, region=TOP_STRIP,
                                        workers=workers, progress=progress)
            undecided = []
            for p in paths:
                text = self.ocr_cache.peek(p, region=TOP_STRIP) or ""
                on, _ = find_markers(text)
                if not on:
                    undecided.append(p)
            if undecided:
                self.ocr_cache.extract_many(undecided, workers=workers)
        else:
            self.ocr_cache.extract_many(paths, workers=workers, progress=progress)

    def detect(self, image_path: Path) -> BrightspaceEvidence:
        text, on, off = self._read(image_path)
        score, verdict = score_markers(on, off)
        sites = sorted({OFF_TASK_KEYWORDS[k] for k in off})
        return BrightspaceEvidence(
            path=str(image_path),
            score=score,
            verdict=verdict,
            on_quiz_matches=on,
            off_task_matches=off,
            off_task_sites=sites,
            is_ide=looks_like_ide(text),
            ocr_snippet=text[:300] if text else None,
        )


def scan_folder(folder: Path, detector: BrightspaceDetector) -> list[BrightspaceEvidence]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise RuntimeError(f"No images in {folder}")
    results = [detector.detect(p) for p in paths]
    detector.ocr_cache.flush()
    results.sort(key=lambda e: e.score, reverse=True)
    return results


def write_csv(results: list[BrightspaceEvidence], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    for r in rows:
        r["on_quiz_matches"] = "|".join(r["on_quiz_matches"])
        r["off_task_matches"] = "|".join(r["off_task_matches"])
        r["off_task_sites"] = "|".join(r["off_task_sites"])
        if r.get("ocr_snippet"):
            r["ocr_snippet"] = r["ocr_snippet"].replace("\n", " ")
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--out", type=Path, default=Path("brightspace_ranked.csv"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    cfg = load_config(args.config) if args.config.exists() else load_config(None)
    detector = BrightspaceDetector(OCRCache(cfg.paths.ocr_cache))

    log.info("Scanning %s ...", args.folder)
    results = scan_folder(args.folder, detector)
    write_csv(results, args.out)

    flagged = sum(1 for r in results if r.score >= 0.5)
    log.info("Scanned %d images; %d flagged (score>=0.5). Wrote %s",
             len(results), flagged, args.out)
    print("\nTop 10:")
    for r in results[:10]:
        bits = []
        if r.off_task_sites:
            bits.append(f"sites=[{','.join(r.off_task_sites)}]")
        if r.on_quiz_matches:
            bits.append(f"quiz_markers={len(r.on_quiz_matches)}")
        print(f"  {r.score:.3f}  [{r.verdict:15s}]  "
              f"{Path(r.path).name:40s} {' '.join(bits)}")


if __name__ == "__main__":
    main()