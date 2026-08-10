#Computes ranking metrics for a trained model on a dataset.
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_scores, all_labels = [], []
    for x, text_emb, y in loader:
        x = x.to(device, non_blocking=True)
        text_emb = text_emb.to(device, non_blocking=True)
        logits = model(x, text_emb).cpu()
        scores = torch.sigmoid(logits)
        all_scores.append(scores)
        all_labels.append(y)
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)

    metrics: dict[str, float] = {}
    if len(set(labels.tolist())) > 1:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["auprc"] = float(average_precision_score(labels, scores))

        prec, rec, thr = precision_recall_curve(labels, scores)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = int(f1.argmax())
        best_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5
        metrics["best_f1"] = float(f1[best_idx])
        metrics["best_threshold"] = best_thr
        metrics["precision@best"] = float(prec[best_idx])
        metrics["recall@best"] = float(rec[best_idx])
    else:
        metrics.update({"auroc": float("nan"), "auprc": float("nan")})

    preds_at_half = (scores >= 0.5).astype(int)
    metrics["acc@0.5"] = float((preds_at_half == labels).mean())
    if labels.sum() > 0:
        metrics["precision@0.5"] = float(precision_score(labels, preds_at_half, zero_division=0))
        metrics["recall@0.5"] = float(recall_score(labels, preds_at_half, zero_division=0))
    return metrics


def format_metrics(m: dict) -> str:
    parts = []
    for k in ["auroc", "auprc", "best_f1", "best_threshold",
              "precision@best", "recall@best", "acc@0.5"]:
        if k in m:
            parts.append(f"{k}={m[k]:.4f}")
    return " | ".join(parts)
