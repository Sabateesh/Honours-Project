# trins the CNN on labeled screenshots 
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from .config import Config, load_config
from .data import build_loaders
from .evaluate import evaluate, format_metrics
from .model import build_model
log = logging.getLogger(__name__)

def _setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

def _compute_pos_weight(train_samples, override):
    if override is not None:
        return torch.tensor(float(override))
    n_pos = sum(1 for _, y in train_samples if y == 1)
    n_neg = sum(1 for _, y in train_samples if y == 0)
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(max(n_neg / n_pos, 1.0))


def _train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total, count = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for x, text_emb, y in pbar:
        x = x.to(device, non_blocking=True)
        text_emb = text_emb.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x, text_emb)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * x.size(0)
        count += x.size(0)
        pbar.set_postfix(loss=f"{total / count:.4f}")
    return total / max(count, 1)


def train(cfg: Config) -> dict:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    log.info("Device: %s", device)

    loaders = build_loaders(cfg)
    pos_weight = _compute_pos_weight(
        loaders["train_samples"],
        None if cfg.train.pos_weight_auto else cfg.train.pos_weight_override,
    ).to(device)
    log.info("pos_weight=%.2f", pos_weight.item())

    model = build_model(cfg.model, text_emb_dim=loaders["text_emb_dim"]).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr,
                      weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    cfg.paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_auprc, best_state = -1.0, None
    for epoch in range(cfg.train.epochs):
        loss = _train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate(model, loaders["val"], device)
        scheduler.step()
        log.info("epoch %02d | loss %.4f | val %s",
                 epoch + 1, loss, format_metrics(val_metrics))

        if val_metrics.get("auprc", 0) > best_auprc:
            best_auprc = val_metrics["auprc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), cfg.paths.checkpoint)
    metadata = {
        "text_emb_dim": loaders["text_emb_dim"],
        "use_ocr": cfg.model.use_ocr,
        "cnn_backbone": cfg.model.cnn_backbone,
        "img_size": cfg.data.img_size,
    }
    sidecar = cfg.paths.checkpoint.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(metadata, indent=2))
    log.info("Saved checkpoint to %s", cfg.paths.checkpoint)

    test_metrics = evaluate(model, loaders["test"], device)
    log.info("TEST | %s", format_metrics(test_metrics))
    return test_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    _setup_logging(cfg.logging.level)
    train(cfg)


if __name__ == "__main__":
    main()
