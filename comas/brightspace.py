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
    ocr_snippet: Optional[str] = None


def find_markers(text: str) -> tuple[list[str], list[str]]:
    if not text:
        return [], []
    low = text.lower()
    on = [k for k in ON_QUIZ_KEYWORDS if k in low]
    off = [k for k in OFF_TASK_KEYWORDS if k in low]
    return on, off


def score_markers(on: list[str], off: list[str]) -> tuple[float, str]:
    if off and not on:
        return 0.95, "left_quiz"
    if off and on:
        return 0.60, "split_view"
    if not on:
        return 0.50, "no_quiz_visible"
    return 0.05, "on_quiz"


class BrightspaceDetector:
    def __init__(self, ocr_cache: OCRCache):
        self.ocr_cache = ocr_cache

    def detect(self, image_path: Path) -> BrightspaceEvidence:
        text = self.ocr_cache.get_or_extract(image_path)
        on, off = find_markers(text)
        score, verdict = score_markers(on, off)
        sites = sorted({OFF_TASK_KEYWORDS[k] for k in off})
        return BrightspaceEvidence(
            path=str(image_path),
            score=score,
            verdict=verdict,
            on_quiz_matches=on,
            off_task_matches=off,
            off_task_sites=sites,
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