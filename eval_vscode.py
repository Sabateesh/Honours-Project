# Scores every saved checkpoint on a labelled folder of real screenshots.
#
# Labels come from the filename, so one folder holds the whole set:
#     "CHAT YES ..."  an assistant chat panel is open
#     "YES ..."       inline ghost text
#     "NO ..."        neither
#
#     python3 eval_vscode.py                      both inference modes
#     python3 eval_vscode.py --mode tiled         tiled only
#     python3 eval_vscode.py --only tiles         checkpoints matching "tiles"
#
# Scores are cached in scores_vscode.json, so an interrupted run resumes
# where it stopped. Delete that file to rescore from scratch.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SHIPPED_THRESHOLD = 0.9925          # gui.FLAG_THRESHOLD


def label_of(name: str) -> str:
    if name.startswith("CHAT YES"):
        return "chat"
    if name.startswith("YES"):
        return "ghost"
    if name.startswith("NO"):
        return "neg"
    return "unlabelled"


def auroc(pos, neg) -> float:
    """Probability a random positive outranks a random negative. 0.5 = chance."""
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_operating_point(pos, neg, max_fpr):
    """Highest recall reachable without exceeding max_fpr."""
    best = (0.0, 0.0, 1.0)
    for t in sorted(set(pos + neg)) + [1.0]:
        r = sum(1 for s in pos if s >= t) / len(pos)
        f = sum(1 for s in neg if s >= t) / len(neg)
        if f <= max_fpr and r > best[0]:
            best = (r, f, t)
    return best


def load_checkpoint(ckpt: Path, device):
    import torch

    from comas.config import ModelConfig
    from comas.data import build_transforms
    from comas.model import build_model

    meta = json.loads(ckpt.with_suffix(".meta.json").read_text())
    size = meta.get("img_size", 768)
    model = build_model(
        ModelConfig(use_ocr=False,
                    cnn_backbone=meta.get("cnn_backbone", "resnet50")),
        text_emb_dim=0, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    _, tfms = build_transforms(size)
    return model, tfms, size


def score_model(ckpt: Path, paths, mode: str, device):
    """Whole image resized to the input size, or the maximum over
    native-resolution tiles - the two ways the app can apply a checkpoint."""
    import torch
    from PIL import Image

    from comas.tiling import iter_tiles

    model, tfms, size = load_checkpoint(ckpt, device)

    def batch_score(images, batch=8):
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                chunk = images[i:i + batch]
                x = torch.stack([tfms(im) for im in chunk]).to(device)
                t = torch.zeros(len(chunk), 0, device=device)
                out.extend(float(v) for v in torch.sigmoid(model(x, t)).cpu())
        return out

    scores = {}
    for p in paths:
        img = Image.open(p).convert("RGB")
        if mode == "whole":
            scores[p.name] = batch_score([img])[0]
        else:
            tiles = [t for _, t in iter_tiles(img)]
            scores[p.name] = max(batch_score(tiles)) if tiles else 0.0
    return scores, size


def score_keywords(paths, cache_path: Path):
    """The OCR keyword layer, on the same images and the same scale."""
    from comas.copilot import find_keywords, score_from_keywords
    from comas.ocr import OCRCache

    cache = OCRCache(cache_path)
    cache.extract_many(paths)
    scores = {}
    for p in paths:
        strong, weak = find_keywords(cache.get_or_extract(p))
        scores[p.name] = score_from_keywords(strong, weak)[0]
    cache.flush()
    return scores


def report(results, mode, counts, max_fpr):
    rows = []
    for key, d in results.items():
        name, m = key.rsplit("|", 1)
        if m != mode:
            continue
        s = d["scores"]
        by = {k: [v for n, v in s.items() if label_of(n) == k]
              for k in ("chat", "ghost", "neg")}
        pos = by["chat"] + by["ghost"]
        if not pos or not by["neg"]:
            continue
        rows.append((auroc(pos, by["neg"]), name, d["img_size"], by, pos))
    if not rows:
        return

    title = ("whole image (resized to the model's input)" if mode == "whole"
             else "tiled (native-resolution crops, highest tile wins)")
    print(f"\n=== {title}")
    print(f"    {counts['chat']} chat + {counts['ghost']} ghost "
          f"= {counts['chat'] + counts['ghost']} positives "
          f"vs {counts['neg']} negatives\n")
    hdr = (f"{'checkpoint':<18}{'input':>6}{'AUROC':>7}{'ghost':>7}{'chat':>7}"
           f"{'recall':>9}{'FPR':>6}{'thresh':>9}"
           f"{'rec@ship':>10}{'fpr@ship':>10}")
    print(hdr)
    print("-" * len(hdr))
    for a, name, size, by, pos in sorted(rows, reverse=True):
        r, f, t = best_operating_point(pos, by["neg"], max_fpr)
        rs = sum(1 for x in pos if x >= SHIPPED_THRESHOLD) / len(pos)
        fs = sum(1 for x in by["neg"] if x >= SHIPPED_THRESHOLD) / len(by["neg"])
        print(f"{name:<18}{size if size else '-':>6}{a:>7.3f}"
              f"{auroc(by['ghost'], by['neg']):>7.3f}"
              f"{auroc(by['chat'], by['neg']):>7.3f}"
              f"{r:>9.2f}{f:>6.2f}{t:>9.4f}{rs:>10.2f}{fs:>10.2f}")
    print("-" * len(hdr))
    print(f"  ghost / chat are per-group AUROCs against the same negatives.")
    print(f"  recall / FPR / thresh: the best operating point with "
          f"FPR <= {max_fpr:.2f}.")
    print(f"  *@ship: what you get at the app's built-in "
          f"{SHIPPED_THRESHOLD} threshold.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, default=Path("test_shots/VSCODE"))
    ap.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    ap.add_argument("--mode", choices=["whole", "tiled", "both"], default="both")
    ap.add_argument("--only", help="substring filter on checkpoint name")
    ap.add_argument("--max-fpr", type=float, default=0.11,
                    help="false-positive budget for the operating point")
    ap.add_argument("--cache", type=Path, default=Path("scores_vscode.json"))
    ap.add_argument("--ocr-cache", type=Path, default=Path("cache/ocr.json"))
    ap.add_argument("--csv", type=Path, default=Path("scores_vscode.csv"))
    ap.add_argument("--no-keywords", action="store_true",
                    help="skip the OCR keyword baseline")
    args = ap.parse_args()

    import torch

    paths = sorted(p for p in args.folder.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No images in {args.folder}")
    counts = {k: sum(1 for p in paths if label_of(p.name) == k)
              for k in ("chat", "ghost", "neg", "unlabelled")}
    if counts["unlabelled"]:
        raise SystemExit(
            f"{counts['unlabelled']} file(s) in {args.folder} start with "
            f"none of 'CHAT YES', 'YES', 'NO' - rename or move them.")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"{len(paths)} images from {args.folder}  "
          f"({counts['chat']} chat, {counts['ghost']} ghost, "
          f"{counts['neg']} negative)")
    print(f"device: {device}")

    results = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    ckpts = sorted(p for p in args.checkpoints.glob("*.pt")
                   if p.with_suffix(".meta.json").exists()
                   and (not args.only or args.only in p.name))
    if not ckpts:
        raise SystemExit(f"No checkpoints with a .meta.json in {args.checkpoints}")

    modes = ["whole", "tiled"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for ckpt in ckpts:
            key = f"{ckpt.stem}|{mode}"
            if key in results:
                continue
            t0 = time.time()
            scores, size = score_model(ckpt, paths, mode, device)
            results[key] = {"img_size": size, "scores": scores}
            args.cache.write_text(json.dumps(results, indent=1))
            print(f"  {key:<26} {time.time() - t0:6.1f}s", flush=True)

    if not args.no_keywords:
        for mode in modes:
            key = f"OCR keywords|{mode}"
            if key not in results:
                try:
                    results[key] = {"img_size": 0,
                                    "scores": score_keywords(paths,
                                                             args.ocr_cache)}
                except Exception as e:                     # tesseract missing
                    print(f"  keyword baseline skipped: {e}")
                    break
        args.cache.write_text(json.dumps(results, indent=1))

    for mode in modes:
        report(results, mode, counts, args.max_fpr)

    import csv
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = sorted(results)
        w.writerow(["file", "label"] + keys)
        for p in paths:
            w.writerow([p.name, label_of(p.name)]
                       + [f"{results[k]['scores'].get(p.name, float('nan')):.6f}"
                          for k in keys])
    print(f"\nper-image scores: {args.csv}")


if __name__ == "__main__":
    main()
