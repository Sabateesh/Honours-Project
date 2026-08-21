
from __future__ import annotations

import os
import sys
from pathlib import Path

from .config import Config, load_config

FROZEN = getattr(sys, "frozen", False)
APP_NAME = "CoMas"


def bundle_root() -> Path:
    """The directory holding the app's read-only files."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent.parent


def bundled(*parts: str) -> Path:
    """A file shipped with the app: the model, config.yaml, tesseract."""
    return bundle_root().joinpath(*parts)


def writable_dir() -> Path:
    """Somewhere the OCR cache and exported reports can actually be written.

    Per-user application data when frozen; the project directory otherwise, so
    a checkout keeps using ./cache as it always has."""
    if not FROZEN:
        return Path.cwd()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    d = base / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_tesseract() -> Path | None:
    """The OCR binary, which is not a Python package and so is not something
    pip or PyInstaller will find on their own. Bundled copy first, then the
    places the official Windows installer puts it."""
    candidates = [
        bundled("tesseract", "tesseract.exe"),
        bundled("tesseract", "tesseract"),
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return p
        except OSError:                 # a Windows path evaluated on POSIX
            continue
    return None


def load_app_config() -> Config:
    """The shipped config, with every path rewritten to somewhere that exists
    at runtime. Reading it straight from the working directory is what left a
    frozen build silently running with no model."""
    cfg_path = bundled("config.yaml")
    cfg = load_config(cfg_path if cfg_path.exists() else None)
    if not FROZEN:
        return cfg
    cfg.paths.checkpoint = bundled(*cfg.paths.checkpoint.parts)
    cfg.paths.ocr_cache = writable_dir() / Path(cfg.paths.ocr_cache).name
    return cfg
