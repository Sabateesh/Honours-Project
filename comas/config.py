from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class AugmentConfig:
    """Train-time degradations that narrow the synthetic-to-real gap. These
    fight against the ghost-text signal, which is only a pixel or two of dim
    text, so the defaults are deliberately mild."""
    downscale_p: float = 0.5
    downscale_min: float = 0.75
    downscale_max: float = 0.95
    jpeg_p: float = 0.25
    jpeg_qmin: int = 65
    jpeg_qmax: int = 95
    blur_p: float = 0.15
    blur_min: float = 0.3
    blur_max: float = 0.6


@dataclass
class TileConfig:
    """Tiled inference. Tiles are square, sized as a fraction of image height
    so a tile covers the same amount of interface regardless of whether the
    screenshot came from a 1440px laptop or a 3600px Retina capture."""
    enabled: bool = False
    tile_frac: float = 0.34
    overlap: float = 0.35
    min_px: int = 256


@dataclass
class DataConfig:
    root: Path = Path("data/")
    img_size: int = 384
    num_workers: int = 4
    batch_size: int = 16
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42




@dataclass
class ModelConfig:
    use_ocr: bool = True
    cnn_backbone: str = "resnet50"
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    fusion_hidden: int = 256
    dropout: float = 0.3


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    pos_weight_auto: bool = True
    pos_weight_override: Optional[float] = None


@dataclass
class PathsConfig:
    checkpoint: Path = Path("checkpoints/best.pt")
    ocr_cache: Path = Path("cache/ocr.json")


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    tile: TileConfig = field(default_factory=TileConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _coerce_paths(d: dict) -> dict:
    if "data" in d and "root" in d["data"]:
        d["data"]["root"] = Path(d["data"]["root"])
    if "paths" in d:
        for k, v in d["paths"].items():
            d["paths"][k] = Path(v)
    return d


def load_config(path: Optional[Path] = None) -> Config:
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw = _coerce_paths(raw)
    return Config(
        data=DataConfig(**raw.get("data", {})),
        augment=AugmentConfig(**raw.get("augment", {})),
        tile=TileConfig(**raw.get("tile", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        paths=PathsConfig(**raw.get("paths", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
    )


def dump_config(cfg: Config, path: Path) -> None:
    def _serialize(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        return obj

    data = _serialize(asdict(cfg))
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))