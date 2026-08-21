from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from PIL import Image


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

    class Dataset:
        pass

    torch = None
    DataLoader = None
    transforms = None

log = logging.getLogger(__name__)


class RandomDownscale:
    def __init__(self, p=0.5, scale=(0.5, 0.9)):
        self.p = p
        self.scale = scale

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        s = random.uniform(*self.scale)
        w, h = img.size
        small = img.resize((max(1, int(w * s)), max(1, int(h * s))),
                           Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class RandomJPEG:
    def __init__(self, p=0.4, quality=(40, 90)):
        self.p = p
        self.quality = quality

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG",
                 quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomBlur:
    def __init__(self, p=0.3, radius=(0.4, 1.2)):
        self.p = p
        self.radius = radius

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        from PIL import ImageFilter
        return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius)))


def build_transforms(img_size: int, aug=None) -> tuple[Any, Any]:
    if not _ML_AVAILABLE:
        raise RuntimeError("torch/torchvision required")
    if aug is None:
        from .config import AugmentConfig
        aug = AugmentConfig()
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        RandomDownscale(p=aug.downscale_p,
                        scale=(aug.downscale_min, aug.downscale_max)),
        RandomJPEG(p=aug.jpeg_p, quality=(aug.jpeg_qmin, aug.jpeg_qmax)),
        RandomBlur(p=aug.blur_p, radius=(aug.blur_min, aug.blur_max)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


class ScreenshotDataset(Dataset):
    def __init__(self, samples, transform, text_embs=None):
        self.samples = samples
        self.transform = transform
        self.text_embs = text_embs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        if self.text_embs is None:
            text_emb = torch.zeros(0)
        else:
            text_emb = self.text_embs[idx]
        return img, text_emb, torch.tensor(label, dtype=torch.float32)


def _session_id(path: Path) -> str:
    m = re.match(r"([^_]+)_", path.stem)
    return m.group(1) if m else path.stem


def collect_samples(data_root: Path) -> list[tuple[Path, int]]:
    samples = []
    for label_name, label in [("no_copilot", 0), ("copilot_active", 1)]:
        folder = data_root / label_name
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                samples.append((p, label))
    return samples


def session_split(samples, val_frac, test_frac, seed):
    by_session = defaultdict(list)
    for path, label in samples:
        by_session[_session_id(path)].append((path, label))

    sessions = sorted(by_session.keys())
    rng = random.Random(seed)
    rng.shuffle(sessions)

    n_test = max(1, int(len(sessions) * test_frac))
    n_val = max(1, int(len(sessions) * val_frac))
    test_s = set(sessions[:n_test])
    val_s = set(sessions[n_test:n_test + n_val])

    train, val, test = [], [], []
    for sid, items in by_session.items():
        if sid in test_s:
            test.extend(items)
        elif sid in val_s:
            val.extend(items)
        else:
            train.extend(items)
    return train, val, test


def precompute_text_embeddings(samples, text_encoder, ocr_cache, batch_size=32):
    texts = [ocr_cache.get_or_extract(path) for path, _ in tqdm(samples, desc="OCR")]
    ocr_cache.flush()

    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encoding text"):
        embs.append(text_encoder.encode(texts[i:i + batch_size]))
    return torch.cat(embs, dim=0)


def build_loaders(cfg) -> dict:
    if not _ML_AVAILABLE:
        raise RuntimeError("torch/torchvision required")
    from .model import TextEncoder
    from .ocr import OCRCache

    dcfg = cfg.data
    samples = collect_samples(dcfg.root)
    if not samples:
        raise RuntimeError(f"No images found under {dcfg.root}")
    train, val, test = session_split(samples, dcfg.val_frac, dcfg.test_frac, dcfg.seed)
    log.info("split: %d train / %d val / %d test", len(train), len(val), len(test))

    train_tfms, eval_tfms = build_transforms(dcfg.img_size,
                                             getattr(cfg, "augment", None))

    train_embs = val_embs = test_embs = None
    text_emb_dim = 0
    if cfg.model.use_ocr:
        encoder = TextEncoder(cfg.model.text_encoder)
        text_emb_dim = encoder.embedding_dim
        cache = OCRCache(cfg.paths.ocr_cache)
        train_embs = precompute_text_embeddings(train, encoder, cache)
        val_embs = precompute_text_embeddings(val, encoder, cache)
        test_embs = precompute_text_embeddings(test, encoder, cache)

    train_ds = ScreenshotDataset(train, train_tfms, train_embs)
    val_ds = ScreenshotDataset(val, eval_tfms, val_embs)
    test_ds = ScreenshotDataset(test, eval_tfms, test_embs)

    return {
        "train": DataLoader(train_ds, batch_size=dcfg.batch_size, shuffle=True,
                            num_workers=dcfg.num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=dcfg.batch_size, shuffle=False,
                          num_workers=dcfg.num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=dcfg.batch_size, shuffle=False,
                           num_workers=dcfg.num_workers, pin_memory=True),
        "train_samples": train,
        "text_emb_dim": text_emb_dim,
    }
