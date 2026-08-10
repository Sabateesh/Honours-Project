from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from .config import ModelConfig


def _build_cnn_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=weights)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        feat_dim = m.classifier[-1].in_features
        m.classifier = nn.Identity()
        return m, feat_dim
    raise ValueError(f"Unknown CNN backbone: {name}")


class CNNOnlyModel(nn.Module):
    def __init__(self, backbone="resnet50", dropout=0.3, pretrained=True):
        super().__init__()
        self.backbone, feat_dim = _build_cnn_backbone(backbone, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x, text_emb=None):
        feats = self.backbone(x)
        return self.head(feats).squeeze(-1)


class FusionModel(nn.Module):
    def __init__(self, backbone="resnet50", text_emb_dim=384,
                 hidden=256, dropout=0.3, pretrained=True):
        super().__init__()
        self.backbone, cnn_dim = _build_cnn_backbone(backbone, pretrained)
        self.text_emb_dim = text_emb_dim
        self.head = nn.Sequential(
            nn.Linear(cnn_dim + text_emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, text_emb):
        cnn_feats = self.backbone(x)
        fused = torch.cat([cnn_feats, text_emb], dim=1)
        return self.head(fused).squeeze(-1)


class TextEncoder:
    def __init__(self, model_name, device=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        if device:
            self.model = self.model.to(device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(self, texts):
        embs = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embs.cpu().float()


def build_model(cfg: ModelConfig, text_emb_dim: Optional[int] = None,
                pretrained: bool = True) -> nn.Module:
    if cfg.use_ocr:
        if text_emb_dim is None:
            raise ValueError("FusionModel requires text_emb_dim")
        return FusionModel(
            backbone=cfg.cnn_backbone,
            text_emb_dim=text_emb_dim,
            hidden=cfg.fusion_hidden,
            dropout=cfg.dropout,
            pretrained=pretrained,
        )
    return CNNOnlyModel(
        backbone=cfg.cnn_backbone,
        dropout=cfg.dropout,
        pretrained=pretrained,
    )
