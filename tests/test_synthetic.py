import random

from comas.synthetic import (
    NEG_PANELS,
    _blend,
    _load_font,
    _plan,
    _wrap,
    make_sample,
)
from PIL import Image, ImageDraw, ImageFont


def _render(text, font, size=(360, 40)):
    img = Image.new("L", size, 0)
    ImageDraw.Draw(img).text((6, 6), text, fill=255, font=font)
    return img


def test_ghost_face_is_slanted_not_just_a_different_file():
    """The one failure that cannot be seen by glancing at the output folder.

    The italic face used to be taken by collection index - "Menlo.ttc index 1
    is the italic" - and on macOS index 1 is Menlo BOLD, so every positive was
    generated with upright ghost text and no font signal at all. Selection is
    by style name now; this pins that the chosen face actually slants."""
    upright = _load_font(18)
    italic = _load_font(18, italic=True)
    sample = "value = compute(total)"
    a, b = _render(sample, upright), _render(sample, italic)
    assert a.tobytes() != b.tobytes(), \
        "ghost face renders identically to the upright face"

    # A slanted face leans its strokes: the top half of a tall glyph sits to
    # the right of the bottom half. Bold or a different family would not.
    def lean(img):
        px = img.load()
        w, h = img.size
        rows = []
        for y in range(h):
            xs = [x for x in range(w) if px[x, y] > 96]
            if xs:
                rows.append((y, sum(xs) / len(xs)))
        top = [c for y, c in rows if y < rows[len(rows) // 2][0]]
        bot = [c for y, c in rows if y > rows[len(rows) // 2][0]]
        return sum(top) / len(top) - sum(bot) / len(bot)

    assert lean(b) > lean(a) + 1.0, \
        f"ghost face does not lean (upright {lean(a):.2f}, ghost {lean(b):.2f})"


def test_ghost_colour_is_dimmed_syntax_not_flat_grey():
    """Measured on real captures, a suggestion keeps its syntax hue and is
    dimmed toward the background. Drawing one flat grey made "grey" and
    "suggestion" the same feature."""
    bg = "#1e1e1e"
    keyword, string = "#569cd6", "#ce9178"
    dk, ds = _blend(keyword, bg, 0.35), _blend(string, bg, 0.35)
    assert dk != ds, "dimming collapsed two syntax colours into one"
    for dim, src in ((dk, keyword), (ds, string)):
        assert sum(dim) < sum(_blend(src, bg, 0.0)), "dimming did not darken"
    assert _blend(keyword, bg, 0.0) == (0x56, 0x9c, 0xd6)
    assert _blend(keyword, bg, 1.0) == (0x1e, 0x1e, 0x1e)


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


def test_no_dim_text_is_drawn_into_a_negative(monkeypatch):
    """A negative that shows dim trailing text looks positive to a human, and
    that is label noise on the class that can least afford it.

    The inlay-hint confuser used to append `: str` / `-> None` to the right end
    of arbitrary lines - comments and `continue` statements included - so 16 of
    40 clean negatives carried up to 15 apiece, each sitting exactly where a
    suggestion goes. Region labels never caught it: they stayed empty."""
    import comas.synthetic as S

    seen = []
    original = S._draw_code_line

    def spy(draw, x, y, tokens, theme, font, ghost=False, **kw):
        if ghost:
            seen.append("".join(tok for tok, _ in tokens))
        return original(draw, x, y, tokens, theme, font, ghost=ghost, **kw)

    monkeypatch.setattr(S, "_draw_code_line", spy)
    rng = random.Random(11)
    corpus = _corpus()
    for variant in ["clean", "hardneg"]:
        for _ in range(20):
            S.make_sample(corpus, rng, variant)
    assert not seen, f"dim text drawn into a negative: {seen[:5]}"


def test_neg_panels_are_not_chat():
    # the hard negatives must never render the chat panel itself
    assert "chat" not in NEG_PANELS


def test_wrap_respects_width():
    font = ImageFont.load_default()
    lines = _wrap("the quick brown fox jumps over the lazy dog", font, 60)
    assert len(lines) > 1
    assert all(font.getlength(ln) <= 60 for ln in lines if " " in ln)
