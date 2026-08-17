# Per-variant breakdown of the trained model: recall for each way an assistant
# shows up (ghost text / chat panel / both), false-positive rate for each kind
# of negative (clean editor / other side panel / browser). The headline AUROC
# hides exactly this - a model can ace panels and miss ghost text entirely.
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from .config import load_config
from .copilot import build_ml_scorer
from .data import session_split

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
POS_VARIANTS = {"ghost", "panel", "both"}
NEG_VARIANTS = {"clean", "hardneg"}


def parse_variant(path: Path) -> str:
    """Filenames look like synth07_ghost_0012.png. Anything without a variant
    token (e.g. browser negatives folded in from data_brightspace) is 'other'."""
    parts = path.stem.split("_")
    for token in parts[1:]:
        if token in POS_VARIANTS or token in NEG_VARIANTS:
            return token
    return "other"


def collect(root: Path) -> list[tuple[Path, int, str]]:
    rows = []
    for folder, label in [("copilot_active", 1), ("no_copilot", 0)]:
        d = root / folder
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                rows.append((p, label, parse_variant(p)))
    return rows


def auroc(pos_scores, neg_scores) -> float:
    """Rank-based AUROC: probability a random positive outscores a random
    negative. Threshold-free, so it compares models fairly even when their
    score distributions sit in different places."""
    if not pos_scores or not neg_scores:
        return float("nan")
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos_scores) * len(neg_scores))


def report(rows, scores, threshold: float) -> str:
    groups = defaultdict(list)
    raw = defaultdict(list)
    for (p, label, variant), score in zip(rows, scores):
        groups[(label, variant)].append(score >= threshold)
        raw[(label, variant)].append(score)

    lines = [f"threshold = {threshold:.2f}", "", "positives (recall):"]
    for (label, variant), hits in sorted(groups.items(), reverse=True):
        if label != 1:
            continue
        lines.append(f"  {variant:8s} {sum(hits):4d}/{len(hits):<4d} "
                     f"recall={sum(hits) / len(hits):.3f}")
    lines.append("negatives (false-positive rate):")
    for (label, variant), hits in sorted(groups.items()):
        if label != 0:
            continue
        lines.append(f"  {variant:8s} {sum(hits):4d}/{len(hits):<4d} "
                     f"fpr={sum(hits) / len(hits):.3f}")

    # The hardest discrimination in the dataset, isolated. Overall AUROC is
    # dominated by easy pairs (panel vs browser) and hides this number.
    g_vs_c = auroc(raw.get((1, "ghost"), []), raw.get((0, "clean"), []))
    lines += ["", f"ghost-vs-clean AUROC = {g_vs_c:.3f}  "
                  f"(the fine-grained signal; 0.5 = chance)"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data_vscode"))
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--split", choices=["test", "val", "all"], default="test",
                    help="'test' (default) scores only held-out sessions the "
                         "model never trained on; 'all' includes training "
                         "images and inflates every number")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    scorer = build_ml_scorer(args.config if args.config.exists() else None)
    if scorer is None:
        raise SystemExit("No trained checkpoint (or torch missing); "
                         "run `python3 -m comas.train` first.")

    rows = collect(args.data)
    if not rows:
        raise SystemExit(f"No images under {args.data}")

    if args.split != "all":
        cfg = load_config(args.config if args.config.exists() else None)
        # reproduce the exact split train.py used (same seed, same fractions)
        samples = [(p, label) for p, label, _ in rows]
        _, val, test = session_split(samples, cfg.data.val_frac,
                                     cfg.data.test_frac, cfg.data.seed)
        keep = {str(p) for p, _ in (test if args.split == "test" else val)}
        rows = [r for r in rows if str(r[0]) in keep]
        log.info("Restricted to %s split: %d images "
                 "(the model never trained on these)", args.split, len(rows))

    log.info("Scoring %d images...", len(rows))
    scored = scorer.score_batch([p for p, _, _ in rows])
    scores = [scored[str(p)] for p, _, _ in rows]
    print()
    print(report(rows, scores, args.threshold))


if __name__ == "__main__":
    main()
