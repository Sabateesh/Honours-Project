import pickle
import random
from pathlib import Path

from comas.data import RandomBlur, RandomDownscale, RandomJPEG
from comas.variant_report import parse_variant
from PIL import Image


def _img():
    im = Image.new("RGB", (64, 48))
    px = im.load()
    for x in range(64):
        for y in range(48):
            px[x, y] = (x * 4 % 256, y * 5 % 256, (x + y) % 256)
    return im


def test_augmentations_preserve_size_and_change_pixels():
    random.seed(0)
    for tfm in [RandomDownscale(p=1.0), RandomJPEG(p=1.0), RandomBlur(p=1.0)]:
        src = _img()
        out = tfm(src)
        assert out.size == src.size
        assert list(out.getdata()) != list(src.getdata())


def test_augmentations_are_noops_at_p_zero():
    for tfm in [RandomDownscale(p=0.0), RandomJPEG(p=0.0), RandomBlur(p=0.0)]:
        src = _img()
        assert tfm(src) is src


def test_augmentations_are_picklable():
    # macOS DataLoader workers spawn: transforms must survive pickling
    for tfm in [RandomDownscale(), RandomJPEG(), RandomBlur()]:
        clone = pickle.loads(pickle.dumps(tfm))
        assert clone.p == tfm.p


def test_parse_variant_reads_synthetic_names():
    assert parse_variant(Path("synth07_ghost_0012.png")) == "ghost"
    assert parse_variant(Path("synth00_both_0181.png")) == "both"
    assert parse_variant(Path("synth19_hardneg_0003.png")) == "hardneg"


def test_auroc_rank_based():
    from comas.variant_report import auroc
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0          # perfect separation
    assert auroc([0.1, 0.2], [0.9, 0.8]) == 0.0          # perfectly wrong
    assert auroc([0.5], [0.5]) == 0.5                     # tie = chance
    assert abs(auroc([0.7, 0.3], [0.5, 0.1]) - 0.75) < 1e-9
    import math
    assert math.isnan(auroc([], [0.5]))


def test_parse_variant_defaults_to_other():
    # browser negatives folded in from data_brightspace keep their own names
    assert parse_variant(Path("bl0_0000.png")) == "other"
    assert parse_variant(Path("bq15_0175.png")) == "other"
