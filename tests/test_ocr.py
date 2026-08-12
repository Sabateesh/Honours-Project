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
    with patch("comas.ocr._ocr_image", return_value="hello world"):
        assert cache.get_or_extract(img) == "hello world"
        assert cache.get_or_extract(img) == "hello world"
    cache.flush()
    cache2 = OCRCache(cache_path)
    with patch("comas.ocr._ocr_image",
               side_effect=AssertionError("should not run")):
        assert cache2.get_or_extract(img) == "hello world"


def test_region_cached_separately_from_full_frame(tmp_path: Path):
    cache = OCRCache(tmp_path / "ocr.json")
    img = tmp_path / "img.png"
    _make_png(img)
    with patch("comas.ocr._ocr_image", return_value="top strip"):
        assert cache.get_or_extract(img, region=(0.0, 0.25)) == "top strip"
    # the crop must not satisfy a later full-frame lookup
    with patch("comas.ocr._ocr_image", return_value="whole page"):
        assert cache.get_or_extract(img) == "whole page"
    assert len(cache) == 2


def test_extract_many_skips_cached_and_fills_rest(tmp_path: Path):
    cache = OCRCache(tmp_path / "ocr.json")
    imgs = []
    for i in range(3):
        p = tmp_path / f"i{i}.png"
        _make_png(p, (i * 20, 0, 0))
        imgs.append(p)
    with patch("comas.ocr._ocr_image", return_value="text"):
        cache.get_or_extract(imgs[0])
    seen = []

    def fake(args):
        seen.append(args[0])
        return args[0], "text"

    with patch("comas.ocr._ocr_job", side_effect=fake):
        cache.extract_many(imgs, workers=1)
    assert len(seen) == 2          # the cached one was not re-OCR'd
    assert len(cache) == 3
