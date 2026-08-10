# infers scores for a bunch images using a trained model
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import Config, load_config
from .data import build_transforms
from .model import TextEncoder, build_model
from .ocr import OCRCache

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


class FolderDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = sorted(p for p in folder.iterdir()
                            if p.suffix.lower() in IMAGE_EXTS)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), str(path)


def _load_model_and_meta(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sidecar = cfg.paths.checkpoint.with_suffix(".meta.json")
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing sidecar metadata at {sidecar}. Was train.py run?")
    meta = json.loads(sidecar.read_text())

    if meta["use_ocr"] != cfg.model.use_ocr:
        log.warning("Overriding cfg.model.use_ocr=%s to match checkpoint (%s)",
                    cfg.model.use_ocr, meta["use_ocr"])
        cfg.model.use_ocr = meta["use_ocr"]

    model = build_model(cfg.model, text_emb_dim=meta.get("text_emb_dim", 0),
                        pretrained=False).to(device)
    model.load_state_dict(torch.load(cfg.paths.checkpoint, map_location=device))
    model.eval()
    return model, meta, device


@torch.no_grad()
def rank_folder(folder: Path, cfg: Config) -> pd.DataFrame:
    model, meta, device = _load_model_and_meta(cfg)
    _, eval_tfms = build_transforms(cfg.data.img_size)
    ds = FolderDataset(folder, eval_tfms)
    if not len(ds):
        raise RuntimeError(f"No images in {folder}")
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, num_workers=2)

    text_encoder = None
    ocr_cache = None
    if cfg.model.use_ocr:
        text_encoder = TextEncoder(cfg.model.text_encoder, device=str(device))
        ocr_cache = OCRCache(cfg.paths.ocr_cache)

    rows = []
    for x, paths in tqdm(loader, desc="scoring"):
        x = x.to(device)
        if cfg.model.use_ocr:
            texts = [ocr_cache.get_or_extract(Path(p)) for p in paths]
            text_emb = text_encoder.encode(texts).to(device)
        else:
            text_emb = torch.zeros(len(paths), 0, device=device)
        logits = model(x, text_emb)
        scores = torch.sigmoid(logits).cpu().tolist()
        for p, s in zip(paths, scores):
            rows.append({"path": p, "score": s})

    if ocr_cache:
        ocr_cache.flush()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--out", type=Path, default=Path("ranked.csv"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.logging.level)

    df = rank_folder(args.folder, cfg)
    df.to_csv(args.out, index=False)
    log.info("Wrote %d rows to %s", len(df), args.out)
    print("\nTop 10:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
