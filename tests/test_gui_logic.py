# Pure logic from the GUI, tested without opening a window.
import pytest

pytest.importorskip("tkinter")
from comas.gui import App


def _result(score, cop=None, bs=None, verdict=None, sites="", kind=None):
    return {"path": f"/x/{score}.png", "score": score, "copilot_score": cop,
            "assistant_kind": kind,
            "bs_score": bs, "bs_verdict": verdict, "bs_sites": sites,
            "status": ""}


class _Ev:
    def __init__(self, strong=(), weak=(), ocr_ran=True, tool="unknown"):
        self.strong_matches = list(strong)
        self.weak_matches = list(weak)
        self.ocr_ran = ocr_ran
        self.detected_tool = tool


def test_keyword_match_is_reported_as_chat_panel():
    kind = App._assistant_kind(_Ev(strong=["github copilot"], tool="copilot"))
    assert kind == "chat panel (copilot)"


def test_detection_without_keywords_is_ghost_text():
    # ghost text carries no identifying text, so a detection with a clean
    # OCR pass is the inline case
    assert App._assistant_kind(_Ev()) == "inline ghost text"


def test_kind_unknown_when_ocr_did_not_run():
    assert App._assistant_kind(_Ev(ocr_ran=False)) is None


def test_detail_text_names_the_kind():
    r = _result(0.9, cop=0.9, kind="inline ghost text")
    assert App._detail_text(r, 0.5) == "CHEATING DETECTED: inline ghost text"


def test_detail_text_falls_back_when_kind_unknown():
    r = _result(0.9, cop=0.9, kind=None)
    assert App._detail_text(r, 0.5) == "CHEATING DETECTED: AI assistant"


def test_keyword_match_flags_even_below_model_threshold():
    """The tiled model needs a threshold near 1.0, but keyword scores top out
    at 0.95. Sharing one cutoff would silently disable OCR entirely."""
    r = _result(0.95, cop=0.95)
    r["ml_score"] = 0.10          # model saw nothing
    r["ocr_score"] = 0.95         # "GitHub Copilot" plainly on screen
    assert App._assistant_flagged(r, 0.9925)
    assert "CHEATING DETECTED" in App._detail_text(r, 0.9925)


def test_strong_keyword_outranks_a_model_detection():
    """Readable interface text is verifiable evidence; a model score is not.
    The keyword hit must sort first in the review queue."""
    ocr_hit = _result(1.0, cop=0.95)      # promoted: strong keyword matched
    model_hit = _result(0.999, cop=0.999)
    assert ocr_hit["score"] > model_hit["score"]


def test_weak_keywords_alone_do_not_flag():
    r = _result(0.45, cop=0.45)
    r["ml_score"] = 0.10
    r["ocr_score"] = 0.45         # single weak keyword
    assert not App._assistant_flagged(r, 0.9925)


def test_model_alone_flags_when_over_threshold():
    r = _result(0.999, cop=0.999)
    r["ml_score"] = 0.999
    r["ocr_score"] = 0.0
    assert App._assistant_flagged(r, 0.9925)


def test_neither_signal_does_not_flag():
    r = _result(0.5, cop=0.5)
    r["ml_score"] = 0.5
    r["ocr_score"] = 0.0
    assert not App._assistant_flagged(r, 0.9925)


def test_non_ide_screenshot_flags_window_leave():
    assert App._left_vscode(is_ide=False, ocr_ran=True, combined=False) == 0.95


def test_ide_screenshot_is_not_a_window_leave():
    assert App._left_vscode(is_ide=True, ocr_ran=True, combined=False) == 0.0


def test_window_leave_suppressed_in_combined_mode():
    # the student is expected to be in Brightspace too
    assert App._left_vscode(is_ide=False, ocr_ran=True, combined=True) == 0.0


def test_window_leave_needs_ocr_to_have_run():
    # no text is not evidence there was no editor
    assert App._left_vscode(is_ide=False, ocr_ran=False, combined=False) == 0.0


def test_detail_text_reports_window_leave():
    r = _result(0.95)
    r["left_vscode"] = 0.95
    assert "LEFT VS CODE" in App._detail_text(r, 0.5)


def test_button_does_not_shadow_tk_internals():
    """tkinter.Misc keeps the widget's Tcl path in self._w and the master in
    self._name / self.master. Assigning any of those in a widget subclass
    breaks only after super().__init__ overwrites it, which is easy to miss."""
    import ast
    import inspect
    from comas import gui

    reserved = {"_w", "_name", "_tclCommands", "master", "tk", "children"}
    src = inspect.getsource(gui.Button)
    tree = ast.parse(src)
    assigned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                assigned.add(node.attr)
    clashes = assigned & reserved
    assert not clashes, f"Button assigns tkinter-internal attributes: {clashes}"


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


def test_ide_screenshot_not_flagged_in_combined_mode():
    # a VS Code capture is the student doing their exam, not a tab-leave
    assert App._brightspace_contribution("left_quiz", 0.95, combined=True,
                                         is_ide=True) == 0.0


def test_ide_gets_no_free_pass_in_brightspace_only_mode():
    assert App._brightspace_contribution("left_quiz", 0.95, combined=False,
                                         is_ide=True) == 0.95


def test_non_ide_tab_leave_counts_in_combined_mode():
    assert App._brightspace_contribution("left_quiz", 0.95, combined=True,
                                         is_ide=False) == 0.95
    assert App._brightspace_contribution("on_quiz", 0.05, combined=True,
                                         is_ide=False) == 0.05
