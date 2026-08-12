# Pure logic from the GUI, tested without opening a window.
import pytest

pytest.importorskip("tkinter")
from comas.gui import App


def _result(score, cop=None, bs=None, verdict=None, sites=""):
    return {"path": f"/x/{score}.png", "score": score, "copilot_score": cop,
            "bs_score": bs, "bs_verdict": verdict, "bs_sites": sites,
            "status": ""}


def test_detail_text_respects_threshold():
    r = _result(0.6, cop=0.6)
    assert App._detail_text(r, 0.5) == "AI coding assistant detected"
    assert App._detail_text(r, 0.7) == "no signal above threshold"


def test_detail_text_combines_both_detectors():
    r = _result(0.95, cop=0.95, bs=0.95, verdict="left_quiz", sites="chatgpt")
    text = App._detail_text(r, 0.5)
    assert "AI coding assistant detected" in text
    assert "left_quiz" in text and "chatgpt" in text


def test_detail_text_falls_back_to_verdict_when_no_sites():
    r = _result(0.5, bs=0.5, verdict="no_quiz_visible", sites="")
    assert "no_quiz_visible (no_quiz_visible)" in App._detail_text(r, 0.5)


def test_detail_text_ignores_missing_detectors():
    r = _result(0.9, cop=None, bs=0.9, verdict="left_quiz", sites="google")
    assert "AI coding" not in App._detail_text(r, 0.5)
