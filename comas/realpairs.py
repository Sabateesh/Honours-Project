from __future__ import annotations
import argparse
import logging
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .synthetic import _load_font, load_corpus
log = logging.getLogger(__name__)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def find_flat_region(img: Image.Image, block_w: int, block_h: int,
                     rng: random.Random) -> tuple[int, int]:
    """Locate a low-variance (empty editor) area in the central band."""
    ds = 4
    g = np.asarray(img.convert("L").reduce(ds), dtype=np.float32)
    bw, bh = block_w // ds, block_h // ds
    H, W = g.shape
    x_lo, x_hi = int(W * 0.20), int(W * 0.75) - bw
    y_lo, y_hi = int(H * 0.15), int(H * 0.80) - bh
    if x_hi <= x_lo or y_hi <= y_lo:
        return int(img.width * 0.3), int(img.height * 0.4)
    candidates = []
    for y in range(y_lo, y_hi, max(bh // 2, 4)):
        for x in range(x_lo, x_hi, max(bw // 2, 4)):
            patch = g[y:y + bh, x:x + bw]
            candidates.append((float(patch.std()), x * ds, y * ds))
    candidates.sort(key=lambda c: c[0])
    std, x, y = rng.choice(candidates[:8])
    return x, y

def composite_ghost(img: Image.Image, corpus, rng: random.Random) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font_size = max(12, round(out.height / rng.uniform(75, 95)))
    font = _load_font(font_size)
    line_h = int(font_size * 1.45)
    n_lines = rng.randint(2, 5)

    _, lines = rng.choice(corpus)
    at = rng.randint(0, len(lines) - n_lines - 1)
    ghost = [ln.strip() for ln in lines[at:at + n_lines]]
    ghost = [ln if ln else "pass" for ln in ghost]

    block_w = int(max(font.getlength(ln) for ln in ghost)) + 20
    block_h = n_lines * line_h + 10
    x, y = find_flat_region(out, block_w, block_h, rng)

    patch = np.asarray(out.convert("L").crop((x, y, x + block_w, y + block_h)))
    bg = float(patch.mean())
    delta = rng.uniform(55, 85)
    gray = int(min(bg + delta, 230)) if bg < 128 else int(max(bg - delta, 40))

    indent = " " * rng.choice([0, 4, 4, 8])
    for i, ln in enumerate(ghost):
        pad = indent if i else ""
        draw.text((x + 8, y + 5 + i * line_h), pad + ln,
                  fill=(gray, gray, gray), font=font)
    return out

def jitter_crop(img: Image.Image, rng: random.Random) -> Image.Image:
    """Tiny random edge crop so duplicated negatives aren't byte-identical."""
    W, H = img.size
    dx, dy = int(W * 0.012), int(H * 0.012)
    l, t = rng.randint(0, dx), rng.randint(0, dy)
    r, b = W - rng.randint(0, dx), H - rng.randint(0, dy)
    return img.crop((l, t, r, b))

def build_pairs(folder: Path, out_dir: Path, holdout_dir: Path,
                n_train: int, variants: int, max_width: int, seed: int,
                corpus_roots: list[Path]):
    rng = random.Random(seed)
    corpus = load_corpus(corpus_roots)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise RuntimeError(f"No images in {folder}")
    july = [p for p in paths if "2026-05" not in p.name]
    older = [p for p in paths if "2026-05" in p.name]
    train_paths, holdout = july[:n_train], july[n_train:] + older
    log.info("train reals: %d | holdout reals: %d", len(train_paths), len(holdout))
    def _load(p: Path) -> Image.Image:
        img = Image.open(p).convert("RGB")
        if img.width > max_width:
            img = img.resize((max_width, int(img.height * max_width / img.width)),
                             Image.LANCZOS)
        return img
    for si, p in enumerate(train_paths):
        img = _load(p)
        session = f"real{si:02d}"
        for v in range(variants):
            base = jitter_crop(img, rng)
            pos = composite_ghost(base, corpus, rng)
            pos.save(out_dir / "copilot_active" / f"{session}_{v:04d}.png")
            base.save(out_dir / "no_copilot" / f"{session}_{v:04d}.png")
        log.info("built %d pairs from %s", variants, p.name)

    (holdout_dir / "copilot_active").mkdir(parents=True, exist_ok=True)
    (holdout_dir / "no_copilot").mkdir(parents=True, exist_ok=True)
    for si, p in enumerate(holdout):
        img = _load(p)
        img.save(holdout_dir / "no_copilot" / f"holdout{si:02d}_0000.png")
        for v in range(2):
            pos = composite_ghost(img, corpus, rng)
            pos.save(holdout_dir / "copilot_active" / f"holdout{si:02d}_{v + 1:04d}.png")
    log.info("holdout eval set written to %s", holdout_dir)

def add_real_positives(pos_folder: Path, out_dir: Path, holdout_dir: Path,
                       n_train: int, variants: int, max_width: int, seed: int):
    """Real screenshots with actual Copilot ghost text — the gold data.
    Jitter crops stay tiny so the inline suggestion never leaves the frame."""
    rng = random.Random(seed)
    paths = sorted(p for p in pos_folder.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise RuntimeError(f"No images in {pos_folder}")
    train_paths, holdout = paths[:n_train], paths[n_train:]
    log.info("real positives: %d train / %d holdout", len(train_paths), len(holdout))
    def _load(p: Path) -> Image.Image:
        img = Image.open(p).convert("RGB")
        if img.width > max_width:
            img = img.resize((max_width, int(img.height * max_width / img.width)),
                             Image.LANCZOS)
        return img
    for si, p in enumerate(train_paths):
        img = _load(p)
        session = f"realpos{si:02d}"
        for v in range(variants):
            jitter_crop(img, rng).save(
                out_dir / "copilot_active" / f"{session}_{v:04d}.png")
    (holdout_dir / "copilot_active").mkdir(parents=True, exist_ok=True)
    for si, p in enumerate(holdout):
        _load(p).save(holdout_dir / "copilot_active"
                      / f"holdoutrealpos{si:02d}_0000.png")
    log.info("real-positive holdouts written to %s", holdout_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, default=Path("test_shots"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--holdout-out", type=Path, default=Path("data_holdout"))
    ap.add_argument("--n-train", type=int, default=9)
    ap.add_argument("--pos-folder", type=Path, default=None)
    ap.add_argument("--n-pos-train", type=int, default=9)
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--max-width", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--corpus", type=Path, nargs="*",
                    default=[Path("comas"), Path("tests")])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    (args.out / "copilot_active").mkdir(parents=True, exist_ok=True)
    (args.out / "no_copilot").mkdir(parents=True, exist_ok=True)
    build_pairs(args.folder, args.out, args.holdout_out, args.n_train,
                args.variants, args.max_width, args.seed, args.corpus)
    if args.pos_folder is not None:
        add_real_positives(args.pos_folder, args.out, args.holdout_out,
                           args.n_pos_train, args.variants, args.max_width,
                           args.seed)

if __name__ == "__main__":
    main()
