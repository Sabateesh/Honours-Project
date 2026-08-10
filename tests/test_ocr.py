from pathlib import Path
from unittest.mock import patch
from PIL import Image
from comas.ocr import OCRCache, _image_hash


def _make_png(path: Path, color=(0, 0, 0)) -> None:
    Image.new("RGB", (32, 32), color).save(path)


def test_image_hash_is_content_based(tmp_path: Path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _make_png(a, (10, 10, 10))
    _make_png(b, (10, 10, 10))
    assert _image_hash(a) == _image_hash(b)


def test_image_hash_differs_for_different_content(tmp_path: Path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _make_png(a, (10, 10, 10))
    _make_png(b, (20, 20, 20))
    assert _image_hash(a) != _image_hash(b)


def test_cache_persists_to_disk(tmp_path: Path):
    cache_path = tmp_path / "ocr.json"
    img = tmp_path / "img.png"
    _make_png(img)
    cache = OCRCache(cache_path)
    with patch.object(cache, "_run_ocr", return_value="hello world"):
        assert cache.get_or_extract(img) == "hello world"
        assert cache.get_or_extract(img) == "hello world"
    cache.flush()
    cache2 = OCRCache(cache_path)
    with patch.object(cache2, "_run_ocr",
                      side_effect=AssertionError("should not run")):
        assert cache2.get_or_extract(img) == "hello world"
