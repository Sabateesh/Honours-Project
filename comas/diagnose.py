# Shows exactly what the GUI computes for each screenshot, and why.
#
# The GUI collapses everything into one score and one evidence line, which is
# right for a reviewer and useless for debugging. This prints every stage:
# the model's raw score, whether OCR recognised an editor, whether the model
# was gated, which keywords matched, and the final verdict - so a screenshot
# that "should have been flagged" can be traced to the stage that dropped it.
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .copilot import CopilotDetector, build_ml_scorer
from .ocr import OCRCache

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--expect", choices=["positive", "negative", "unknown"],
                    default="unknown",
                    help="what these screenshots should be, for a recall/FPR line")
    ap.add_argument("--neg-folder", type=Path,
                    help="labelled negatives; enables a real AUROC and a "
                         "threshold sweep, which is the only honest way to "
                         "judge whether the model separates the classes")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    cfg = load_config(args.config if args.config.exists() else None)
    print(f"checkpoint : {cfg.paths.checkpoint} "
          f"({'exists' if cfg.paths.checkpoint.exists() else 'MISSING'})")
    print(f"img_size   : {cfg.data.img_size}")
    print(f"tiled      : {getattr(cfg, 'tile', None) and cfg.tile.enabled}")
    print(f"threshold  : {args.threshold}")
    print()

    scorer = build_ml_scorer(args.config if args.config.exists() else None)
    if scorer is None:
        print("!! No model loaded - the GUI would be running on OCR alone.")

    cache = OCRCache(cfg.paths.ocr_cache)
    det = CopilotDetector(ocr_cache=cache, ml_scorer=scorer,
                          skip_ocr_when_confident=False)

    paths = sorted(p for p in args.folder.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No images in {args.folder}")

    cache.extract_many(paths)
    ml = {}
    if scorer is not None:
        ml = scorer.score_batch(paths)

    print(f"{'file':<34} {'model':>6} {'ide':>5} {'gated':>6} "
          f"{'final':>6}  {'flag':<5} evidence")
    print("-" * 104)
    flagged = 0
    for p in paths:
        ev = det.detect(p, ml_score=ml.get(str(p)))
        hit = ev.score >= args.threshold
        flagged += hit
        kw = ",".join(ev.strong_matches[:2]) or (
            f"{len(ev.weak_matches)} weak" if ev.weak_matches else "-")
        print(f"{p.name[:34]:<34} "
              f"{'' if ev.ml_score is None else format(ev.ml_score, '.4f'):>8} "
              f"{str(ev.is_ide):>5} {str(ev.ml_gated):>6} "
              f"{ev.score:>8.4f}  {'YES' if hit else 'no':<5} {kw}")
    cache.flush()

    print("-" * 104)
    print(f"{flagged}/{len(paths)} flagged at threshold {args.threshold}")
    if args.expect == "positive":
        print(f"recall = {flagged / len(paths):.3f}")
    elif args.expect == "negative":
        print(f"false positive rate = {flagged / len(paths):.3f}")

    gated = sum(1 for p in paths if not det.detect(p, ml_score=ml.get(str(p))).is_ide)
    if gated:
        print(f"\n!! {gated} screenshot(s) were not recognised as an editor, so the "
              f"model's score was discarded for them.\n"
              f"   In VSCODE mode the GUI reports those as 'LEFT VS CODE'.")

    if args.neg_folder:
        neg_paths = sorted(p for p in args.neg_folder.iterdir()
                           if p.suffix.lower() in IMAGE_EXTS)
        cache.extract_many(neg_paths)
        neg_ml = scorer.score_batch(neg_paths) if scorer is not None else {}
        pos_s = [ml.get(str(p), 0.0) for p in paths]
        neg_s = [neg_ml.get(str(p), 0.0) for p in neg_paths]
        cache.flush()

        print(f"\n{'=' * 60}")
        print(f"REAL DATA: {len(pos_s)} positives vs {len(neg_s)} negatives")
        print(f"{'=' * 60}")
        print(f"positive scores: {min(pos_s):.4f} - {max(pos_s):.4f}")
        print(f"negative scores: {min(neg_s):.4f} - {max(neg_s):.4f}")

        wins = sum((1.0 if p > n else 0.5 if p == n else 0.0)
                   for p in pos_s for n in neg_s)
        auroc = wins / (len(pos_s) * len(neg_s))
        print(f"\nAUROC = {auroc:.3f}   "
              f"(0.5 = chance; the model cannot beat a coin at 0.5)")

        print(f"\n{'threshold':>10} {'recall':>8} {'FPR':>8}")
        for t in sorted({round(s, 4) for s in pos_s + neg_s} |
                        {0.3, 0.5, 0.9, 0.99}):
            r = sum(1 for s in pos_s if s >= t) / len(pos_s)
            f = sum(1 for s in neg_s if s >= t) / len(neg_s)
            print(f"{t:>10.4f} {r:>8.2f} {f:>8.2f}")
        print("\nIf no row has high recall AND low FPR, no threshold makes this\n"
              "model usable on real screenshots, whatever the synthetic\n"
              "metrics say.")


if __name__ == "__main__":
    main()
