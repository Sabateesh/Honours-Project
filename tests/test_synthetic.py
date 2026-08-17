import random

from comas.synthetic import NEG_PANELS, _plan, _wrap, make_sample
from PIL import ImageFont


def _corpus():
    lines = [f"x{i} = compute(i) + {i}" for i in range(60)]
    return [("alpha.py", lines), ("beta.py", list(reversed(lines)))]


def test_plan_balances_positive_variants():
    jobs = _plan(100, 100, hard_neg_frac=0.5, rng=random.Random(0))
    pos = [v for lab, v in jobs if lab == "copilot_active"]
    neg = [v for lab, v in jobs if lab == "no_copilot"]
    assert len(pos) == 100 and len(neg) == 100
    # every way an assistant shows up is represented
    assert set(pos) == {"ghost", "panel", "both"}
    assert pos.count("panel") >= 15 and pos.count("ghost") >= 20


def test_ghost_only_plan_has_no_panels():
    jobs = _plan(100, 100, hard_neg_frac=0.35, rng=random.Random(0),
                 ghost_only=True)
    pos = [v for lab, v in jobs if lab == "copilot_active"]
    assert set(pos) == {"ghost"}


def test_ghost_suggestion_is_never_degenerate():
    """A suggestion one space wide is an image labelled positive that shows
    nothing - label noise on the hardest class. 10% of an earlier dataset
    was like this."""
    rng = random.Random(7)
    corpus = _corpus()
    for _ in range(40):
        _, boxes = make_sample(corpus, rng, "ghost")
        ghost = [b for b in boxes if b["kind"] == "ghost"]
        assert ghost, "ghost variant produced no ghost region"
        x0, _, x1, _ = ghost[0]["box"]
        assert x1 - x0 > 30, f"degenerate ghost region {x1 - x0}px wide"


def test_plan_favours_ghost_over_panel():
    # panels are already detected perfectly; ghost text is where the errors are
    jobs = _plan(100, 100, hard_neg_frac=0.5, rng=random.Random(0))
    pos = [v for lab, v in jobs if lab == "copilot_active"]
    assert pos.count("ghost") > pos.count("panel")


def test_plan_hard_negative_fraction():
    jobs = _plan(10, 100, hard_neg_frac=0.5, rng=random.Random(0))
    neg = [v for lab, v in jobs if lab == "no_copilot"]
    assert neg.count("hardneg") == 50
    assert neg.count("clean") == 50


def test_plan_hard_negatives_can_be_disabled():
    jobs = _plan(10, 20, hard_neg_frac=0.0, rng=random.Random(0))
    neg = [v for lab, v in jobs if lab == "no_copilot"]
    assert neg.count("hardneg") == 0


def test_every_variant_renders():
    rng = random.Random(3)
    corpus = _corpus()
    for variant in ["ghost", "panel", "both", "clean", "hardneg"]:
        img, _ = make_sample(corpus, rng, variant)
        assert img.size[0] > 0 and img.size[1] > 0


def test_positive_variants_report_signal_regions():
    rng = random.Random(4)
    corpus = _corpus()
    for variant, kinds in [("ghost", {"ghost"}), ("panel", {"panel"}),
                           ("both", {"ghost", "panel"})]:
        img, boxes = make_sample(corpus, rng, variant)
        assert {b["kind"] for b in boxes} == kinds, variant
        for b in boxes:
            x0, y0, x1, y1 = b["box"]
            assert x1 > x0 and y1 > y0
            assert 0 <= x0 and x1 <= img.size[0]
            assert 0 <= y0 and y1 <= img.size[1]


def test_negative_variants_report_no_regions():
    rng = random.Random(5)
    corpus = _corpus()
    for variant in ["clean", "hardneg"]:
        _, boxes = make_sample(corpus, rng, variant)
        assert boxes == []


def test_neg_panels_are_not_chat():
    # the hard negatives must never render the chat panel itself
    assert "chat" not in NEG_PANELS


def test_wrap_respects_width():
    font = ImageFont.load_default()
    lines = _wrap("the quick brown fox jumps over the lazy dog", font, 60)
    assert len(lines) > 1
    assert all(font.getlength(ln) <= 60 for ln in lines if " " in ln)
