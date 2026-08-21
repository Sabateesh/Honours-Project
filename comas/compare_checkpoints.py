# Scores every saved checkpoint on the SAME set of real screenshots.
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def auroc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_operating_point(pos, neg):
    """Highest recall subject to FPR <= 0.15, matching the OCR baseline's
    false-positive rate so the comparison is like for like."""
    best = (0.0, 1.0, 0.0)
    for t in sorted(set(pos + neg)):
        r = sum(1 for s in pos if s >= t) / len(pos)
        f = sum(1 for s in neg if s >= t) / len(neg)
        if f <= 0.15 and r > best[0]:
            best = (r, f, t)
    return best


def score_with(ckpt: Path, pos_paths, neg_paths, tiled: bool = False):
    import torch

    from .data import build_transforms
    from .model import build_model
    from .config import ModelConfig
    from .tiling import iter_tiles
    from PIL import Image

    meta = json.loads(ckpt.with_suffix(".meta.json").read_text())
    img_size = meta.get("img_size", 768)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")

    mcfg = ModelConfig(use_ocr=False,
                       cnn_backbone=meta.get("cnn_backbone", "resnet50"))
    model = build_model(mcfg, text_emb_dim=0, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    _, tfms = build_transforms(img_size)

    def score_images(images):
        out = []
        with torch.no_grad():
            for i in range(0, len(images), 8):
                chunk = images[i:i + 8]
                x = torch.stack([tfms(im) for im in chunk]).to(device)
                t = torch.zeros(len(chunk), 0, device=device)
                out.extend(float(v) for v in torch.sigmoid(model(x, t)).cpu())
        return out

    def run(paths):
        if not tiled:
            return score_images([Image.open(p).convert("RGB") for p in paths])
        # a tile-trained model applied as intended: score native-resolution
        # crops and take the highest
        scores = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            tiles = [t for _, t in iter_tiles(img)]
            scores.append(max(score_images(tiles)) if tiles else 0.0)
        return scores

    return run(pos_paths), run(neg_paths), img_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=Path, default=Path("test_shots_positive"))
    ap.add_argument("--neg", type=Path, default=Path("test_shots"))
    ap.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    ap.add_argument("--tiled", action="store_true",
                    help="apply each model to native-resolution tiles and take "
                         "the maximum, instead of resizing the whole screenshot")
    ap.add_argument("--only", help="substring filter on checkpoint name")
    ap.add_argument("--save", help="write this run's AUROCs to a JSON file")
    ap.add_argument("--compare", help="a saved JSON from another run; reports "
                                      "how much the two orderings agree")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pos_paths = sorted(p for p in args.pos.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS)
    neg_paths = sorted(p for p in args.neg.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS)
    print(f"real data: {len(pos_paths)} positives, {len(neg_paths)} negatives\n")

    ckpts = sorted(p for p in args.checkpoints.glob("*.pt")
                   if p.with_suffix(".meta.json").exists()
                   and (not args.only or args.only in p.name))

    print(f"inference: {'TILED (native resolution)' if args.tiled else 'whole image (resized)'}\n")
    print(f"{'checkpoint':<24} {'input':>6} {'AUROC':>7} {'recall':>7} "
          f"{'FPR':>6} {'thresh':>9}")
    print("-" * 64)
    rows = []
    for c in ckpts:
        try:
            pos, neg, size = score_with(c, pos_paths, neg_paths, args.tiled)
        except Exception as e:
            print(f"{c.name:<24} failed: {str(e)[:34]}")
            continue
        a = auroc(pos, neg)
        r, f, t = best_operating_point(pos, neg)
        rows.append((a, c.name))
        print(f"{c.name:<24} {size:>6} {a:>7.3f} {r:>7.2f} {f:>6.2f} {t:>9.4f}")

    print("-" * 64)
    if rows:
        ranked = sorted(rows, reverse=True)
        print(f"\nbest: {ranked[0][1]} (AUROC {ranked[0][0]:.3f})")
        print("\nordering (best first):")
        for i, (a, name) in enumerate(ranked, 1):
            print(f"  {i}. {name:<24} {a:.3f}")

        if args.compare:
            prev = json.loads(Path(args.compare).read_text())
            common = [n for _, n in rows if n in prev]
            if len(common) >= 3:
                here = {n: a for a, n in rows}
                r1 = _ranks([here[n] for n in common])
                r2 = _ranks([prev[n] for n in common])
                rho = _spearman(r1, r2)
                print(f"\nSpearman rank correlation with {args.compare}: "
                      f"{rho:+.3f}")
                print("  +1 = identical ordering, 0 = unrelated, -1 = reversed")
                if rho < 0.5:
                    print("  -> the two evaluations do not agree on which model "
                          "is better.")

        if args.save:
            Path(args.save).write_text(
                json.dumps({n: a for a, n in rows}, indent=1))
            print(f"\nsaved scores to {args.save}")


def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def _spearman(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else float("nan")


if __name__ == "__main__":
    main()
