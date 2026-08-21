# Tiled detection: cut the screenshot into overlapping tiles, score each at
# native resolution, take the highest. Resizing a whole screenshot to the
# network input shrinks ghost text to a few pixels; a chat panel survives it.
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

from PIL import Image

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def tile_boxes(width: int, height: int, tile_frac: float = 0.34,
               overlap: float = 0.35, min_px: int = 256) -> list[tuple[int, int, int, int]]:
    """Overlapping square tiles. Overlap stops a ghost line landing on a tile
    edge and looking like nothing in either half."""
    side = max(min_px, int(height * tile_frac))
    side = min(side, width, height)
    stride = max(1, int(side * (1.0 - overlap)))

    def starts(total):
        pts = list(range(0, max(1, total - side + 1), stride))
        if pts[-1] + side < total:
            pts.append(total - side)      # flush against the far edge
        return pts

    return [(x, y, x + side, y + side)
            for y in starts(height) for x in starts(width)]


def _overlap_area(a, b) -> int:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def tile_is_positive(tile, regions, min_frac: float = 0.5) -> bool:
    """A tile is positive when it shows enough of a signal region.

    Two ways to qualify, because the two signals have very different shapes.
    A ghost-text line is small and thin, so the test is whether the tile holds
    most of THE REGION. A chat panel is taller than any tile, so no tile can
    ever hold half of it; there the test is whether the region fills most of
    THE TILE. Checking only the first way silently produces almost no positive
    panel tiles."""
    tile_area = max(1, (tile[2] - tile[0]) * (tile[3] - tile[1]))
    for r in regions:
        box = r["box"] if isinstance(r, dict) else r
        region_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        inter = _overlap_area(tile, box)
        if inter / region_area >= min_frac or inter / tile_area >= min_frac:
            return True
    return False


def iter_tiles(img: Image.Image, tile_frac=0.34, overlap=0.35, min_px=256):
    for box in tile_boxes(img.width, img.height, tile_frac, overlap, min_px):
        yield box, img.crop(box)



def build_tile_dataset(src: Path, out: Path, tile_frac=0.34, overlap=0.35,
                       min_px=256, min_frac=0.5, neg_per_pos=3, seed=42,
                       max_pos_per_image=3, allow_orphans: bool = False) -> dict:
    """Turn full screenshots plus region labels into a tile dataset. Tiles from
    a positive image that miss the signal become negatives - identical theme
    and code, differing only by the absence of what is being detected."""
    rng = random.Random(seed)
    labels_path = src / "labels.json"
    if not labels_path.exists():
        raise SystemExit(
            f"{labels_path} not found. Regenerate the dataset with "
            f"comas.synthetic so region labels are written.")
    labels = json.loads(labels_path.read_text())

    pos_dir = out / "copilot_active"
    neg_dir = out / "no_copilot"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    stats = {"pos": 0, "neg": 0, "neg_from_pos": 0, "images": 0}
    for rel, meta in sorted(labels.items()):
        path = src / rel
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        regions = meta.get("regions", [])
        stem = Path(rel).stem
        variant = meta.get("variant", "unknown")

        positives, negatives = [], []
        for box, tile in iter_tiles(img, tile_frac, overlap, min_px):
            if regions and tile_is_positive(box, regions, min_frac):
                positives.append((box, tile))
            else:
                negatives.append((box, tile))

        if len(positives) > max_pos_per_image:
            positives = rng.sample(positives, max_pos_per_image)

        keep_neg = negatives
        if positives:
            budget = max(1, len(positives) * neg_per_pos)
            keep_neg = rng.sample(negatives, min(len(negatives), budget))
            stats["neg_from_pos"] += len(keep_neg)
        elif len(negatives) > neg_per_pos:
            keep_neg = rng.sample(negatives, neg_per_pos)

        for i, (box, tile) in enumerate(positives):
            tile.save(pos_dir / f"{stem}_t{i:02d}.png")
            stats["pos"] += 1
        for i, (box, tile) in enumerate(keep_neg):
            tile.save(neg_dir / f"{stem}_n{i:02d}.png")
            stats["neg"] += 1
        stats["images"] += 1
        if stats["images"] % 100 == 0:
            log.info("%d images -> %d pos / %d neg tiles",
                     stats["images"], stats["pos"], stats["neg"])
        _ = variant

    known = set(labels)
    orphans = {folder: [p for p in sorted((src / folder).glob("*.png"))
                        if f"{folder}/{p.name}" not in known]
               for folder in ("copilot_active", "no_copilot")
               if (src / folder).exists()}
    stale = sum(len(v) for v in orphans.values())
    if stale and not allow_orphans:
        raise SystemExit(
            f"{stale} image(s) under {src} are missing from labels.json "
            f"({len(orphans.get('copilot_active', []))} positive, "
            f"{len(orphans.get('no_copilot', []))} negative).\n"
            "They are left over from an earlier run: a full "
            "`python3 -m comas.synthetic` rewrites labels.json, and older "
            "builds did not clear the image folders first.\n"
            "Regenerate the folder from scratch, or pass --allow-orphans to "
            "tile the unlabelled negatives anyway.")

    for folder in ("copilot_active", "no_copilot"):
        d = src / folder
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS or f"{folder}/{p.name}" in known:
                continue
            if folder == "copilot_active":
                log.warning("Unlabelled image in positive folder, skipping: %s", p)
                continue
            img = Image.open(p).convert("RGB")
            tiles = [t for _, t in iter_tiles(img, tile_frac, overlap, min_px)]
            for i, tile in enumerate(rng.sample(tiles, min(len(tiles), neg_per_pos))):
                tile.save(neg_dir / f"{p.stem}_x{i:02d}.png")
                stats["neg"] += 1
                stats["neg_unlabelled"] = stats.get("neg_unlabelled", 0) + 1
            stats["images"] += 1
    return stats

class TiledScorer:
    """Scores every tile at native resolution, reports the maximum and which
    tile produced it."""

    def __init__(self, scorer, tile_frac=0.34, overlap=0.35, min_px=256,
                 batch=16):
        self.scorer = scorer
        self.tile_frac = tile_frac
        self.overlap = overlap
        self.min_px = min_px
        self.batch = batch
        self.last_boxes: dict[str, Optional[tuple]] = {}

    def __call__(self, path) -> float:
        return self.score_batch([path])[str(path)]

    def score_batch(self, paths, progress=None) -> dict[str, float]:
        out: dict[str, float] = {}
        total = len(paths)
        for i, p in enumerate(paths):
            p = Path(p)
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                log.warning("Could not read %s: %s", p, e)
                out[str(p)] = 0.0
                self.last_boxes[str(p)] = None
                if progress:
                    progress(i + 1, total)
                continue

            boxes, tiles = [], []
            for box, tile in iter_tiles(img, self.tile_frac, self.overlap,
                                        self.min_px):
                boxes.append(box)
                tiles.append(tile)

            scores = self.scorer.score_images(tiles)
            best = max(range(len(scores)), key=lambda k: scores[k]) if scores else None
            out[str(p)] = float(scores[best]) if best is not None else 0.0
            self.last_boxes[str(p)] = boxes[best] if best is not None else None
            if progress:
                progress(i + 1, total)
        return out

    def box_for(self, path):
        return self.last_boxes.get(str(path))


def main():
    ap = argparse.ArgumentParser(
        description="Build a tile dataset from full screenshots.")
    ap.add_argument("--src", type=Path, default=Path("data_vscode"))
    ap.add_argument("--out", type=Path, default=Path("data_tiles"))
    ap.add_argument("--tile-frac", type=float, default=0.34)
    ap.add_argument("--overlap", type=float, default=0.35)
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--neg-per-pos", type=int, default=3)
    ap.add_argument("--max-pos-per-image", type=int, default=3,
                    help="caps panel tiles so they do not swamp ghost tiles")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-orphans", action="store_true",
                    help="tile images that labels.json does not describe "
                         "instead of stopping; they become negatives")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    stats = build_tile_dataset(args.src, args.out, args.tile_frac,
                               args.overlap, 256, args.min_frac,
                               args.neg_per_pos, args.seed,
                               args.max_pos_per_image, args.allow_orphans)
    log.info("Done. %d images -> %d positive / %d negative tiles under %s",
             stats["images"], stats["pos"], stats["neg"], args.out)
    log.info("%d of the negatives came from positive images "
             "(same theme and code, signal absent).", stats["neg_from_pos"])
    if stats.get("neg_unlabelled"):
        log.info("%d negative tiles came from unlabelled images "
                 "(browser screenshots).", stats["neg_unlabelled"])


if __name__ == "__main__":
    main()
