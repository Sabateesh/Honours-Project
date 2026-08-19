from pathlib import Path

from comas.copilot import (
    CopilotDetector,
    CopilotEvidence,
    find_keywords,
    infer_tool,
    score_from_keywords,
    write_csv,
)


class _CountingCache:
    """Stands in for OCRCache so tests can assert whether OCR was reached."""

    def __init__(self, text=""):
        self.text = text
        self.calls = 0

    def get_or_extract(self, path, region=None):
        self.calls += 1
        return self.text


def test_ocr_runs_however_confident_the_model_is():
    """OCR must never be skipped. The keywords are the only chat-panel
    detector and the only way to tell a panel from inline ghost text, and on
    real screenshots the model is confident on nearly everything."""
    for score in (0.02, 0.85, 0.95, 0.999):
        cache = _CountingCache("GitHub Copilot")
        det = CopilotDetector(ocr_cache=cache, ml_scorer=lambda p, s=score: s)
        ev = det.detect(Path("x.png"))
        assert cache.calls == 1, f"OCR skipped at model score {score}"
        assert ev.score == 0.95 and ev.method == "ocr_strong"


class _FakeTensor:
    def __init__(self, vals):
        self.vals = vals

    def to(self, device):
        return self

    def cpu(self):
        return self

    def __iter__(self):
        return iter(self.vals)


class _FakeTorch:
    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def no_grad(self):
        return self._NoGrad()

    def stack(self, xs):
        return _FakeTensor(xs)

    def zeros(self, n, d, device=None):
        return _FakeTensor([0.0] * n)

    def sigmoid(self, x):
        return x


def test_score_batch_chunks_and_keys_by_path(tmp_path):
    from PIL import Image
    from comas.copilot import MLScorer

    paths = []
    for i in range(5):
        p = tmp_path / f"s{i}.png"
        Image.new("RGB", (8, 8), (i, 0, 0)).save(p)
        paths.append(p)

    def model(x, t):
        return _FakeTensor([0.5] * len(list(x.vals)))

    scorer = MLScorer(model, lambda img: 0, device=None, torch_mod=_FakeTorch())
    scorer.BATCH = 2  # force chunking: 5 images -> 3 forward passes
    ticks = []
    out = scorer.score_batch(paths, progress=lambda d, t: ticks.append((d, t)))
    assert set(out) == {str(p) for p in paths}
    assert all(v == 0.5 for v in out.values())
    assert ticks == [(2, 5), (4, 5), (5, 5)]


def test_score_batch_survives_unreadable_file(tmp_path):
    from PIL import Image
    from comas.copilot import MLScorer

    good = tmp_path / "good.png"
    Image.new("RGB", (8, 8)).save(good)
    bad = tmp_path / "missing.png"  # never created

    def model(x, t):
        return _FakeTensor([0.9] * len(list(x.vals)))

    def tfms(img):
        return 0

    scorer = MLScorer(model, tfms, device=None, torch_mod=_FakeTorch())
    out = scorer.score_batch([good, bad])
    assert out[str(good)] == 0.9
    assert out[str(bad)] == 0.0  # scored safe-low, not crashed


def test_model_score_discarded_when_not_an_editor():
    """The model has never seen a browser or a desktop, so its opinion there
    is meaningless. Without this gate a stray high score would be reported to
    a proctor as 'AI coding assistant detected' on a browser window."""
    cache = _CountingCache("Gmail - Inbox (24)")   # no IDE chrome
    det = CopilotDetector(ocr_cache=cache, ml_scorer=lambda p: 0.97)
    ev = det.detect(Path("x.png"))
    assert ev.is_ide is False
    assert ev.ml_gated is True
    assert ev.score == 0.0          # model ignored
    assert ev.ml_score == 0.97      # but still recorded for transparency


def test_keywords_survive_the_gate():
    # reading the assistant's own name is evidence wherever it appears
    cache = _CountingCache("github.com/copilot - Gmail")
    det = CopilotDetector(ocr_cache=cache, ml_scorer=lambda p: 0.97)
    ev = det.detect(Path("x.png"))
    assert ev.is_ide is False
    assert ev.score > 0.0


def test_model_score_kept_inside_an_editor():
    cache = _CountingCache("EXPLORER  data.py  TERMINAL  UTF-8  Spaces: 4")
    det = CopilotDetector(ocr_cache=cache, ml_scorer=lambda p: 0.97)
    ev = det.detect(Path("x.png"))
    assert ev.is_ide is True
    assert ev.ml_gated is False
    assert ev.score == 0.97



def test_find_keywords_copilot_strong():
    text = "Status: GitHub Copilot is running. Tab to accept."
    strong, weak = find_keywords(text)
    assert "github copilot" in strong
    assert "tab to accept" in strong
    assert infer_tool(strong, weak) == "copilot"

def test_bare_copilot_does_not_trigger():
    text = "Today I'm reading about copilot in the news."
    strong, weak = find_keywords(text)
    assert strong == []
    assert weak == []

def test_filename_copilot_does_not_trigger():
    text = "EXPLORER  __init__.py  copilot.py  data.py  config.yaml"
    strong, weak = find_keywords(text)
    assert strong == []
    assert weak == []

def test_filename_plus_strong_keyword():
    text = "Open copilot.py to see my settings - uses GitHub Copilot."
    strong, weak = find_keywords(text)
    assert "github copilot" in strong

def test_find_keywords_cursor_plan_agent():
    text = "Tip: Try the Plan agent to research and plan before implementing changes."
    strong, weak = find_keywords(text)
    assert "plan agent" in strong
    assert "before implementing changes" in strong
    assert infer_tool(strong, weak) == "cursor"

def test_find_keywords_cursor_placeholder():
    text = "CHAT  SESSIONS  Explore and understand your code"
    strong, weak = find_keywords(text)
    assert "explore and understand your code" in strong
    assert infer_tool(strong, weak) == "cursor"

def test_no_matching_sessions_is_cursor():
    text = "SESSIONS  No matching sessions - Reset Filter"
    strong, weak = find_keywords(text)
    assert "no matching sessions" in strong
    assert infer_tool(strong, weak) == "cursor"

def test_panel_headers_flag_inside_the_editor():
    """CHAT and SESSIONS are the assistant panel's own tab headers. Read
    beside VS Code chrome they are the panel, so the capture is flagged
    outright rather than merely scored."""
    text = ("EXPLORER  main.py  TERMINAL  UTF-8  Spaces: 4  "
            "CHAT  SESSIONS  Add context")
    strong, weak = find_keywords(text)
    assert "chat" in strong
    assert "sessions" in strong
    s, m = score_from_keywords(strong, weak)
    assert s == 1.0 and m == "ocr_chat_panel"


def test_panel_headers_attribute_no_tool():
    """They name no vendor, so they must not turn a Copilot capture into
    'both' or invent a tool of their own."""
    text = "EXPLORER  app.py  TERMINAL  UTF-8  Chat  GitHub Copilot"
    strong, weak = find_keywords(text)
    assert "chat" in strong
    assert infer_tool(strong, weak) == "copilot"
    assert infer_tool(["chat", "sessions"], []) == "unknown"


def test_panel_headers_ignored_outside_the_editor():
    """The word is ordinary English everywhere else. A Brightspace page or a
    browser tab reading 'Chat' is not an assistant, and must not be flagged."""
    for text in ("Brightspace  Course Chat  Submit Quiz  (1 point)",
                 "Gmail - Inbox (24)  Chat  Spaces  Meet",
                 "Sessions - Conference programme"):
        strong, weak = find_keywords(text)
        assert "chat" not in strong
        assert "sessions" not in strong
        assert score_from_keywords(strong, weak)[0] == 0.0


def test_panel_headers_need_a_whole_word():
    """sessions.py sitting in the explorer is a filename, not a panel."""
    text = "EXPLORER  sessions.py  chatbot.py  TERMINAL  UTF-8  Spaces: 4"
    strong, _ = find_keywords(text)
    assert "sessions" not in strong
    assert "chat" not in strong


def test_panel_header_beats_a_confident_model_score():
    cache = _CountingCache("EXPLORER  main.py  TERMINAL  UTF-8  CHAT  SESSIONS")
    det = CopilotDetector(ocr_cache=cache, ml_scorer=lambda p: 0.97)
    ev = det.detect(Path("x.png"))
    assert ev.is_ide is True
    assert ev.score == 1.0 and ev.method == "ocr_chat_panel"
    assert ev.detected_tool == "unknown"


def test_both_tools_visible():
    text = "GitHub Copilot active. Also using Plan agent."
    strong, weak = find_keywords(text)
    assert infer_tool(strong, weak) == "both"

def test_find_keywords_empty():
    assert find_keywords("") == ([], [])
    assert find_keywords(None) == ([], [])

def test_infer_tool_no_matches():
    assert infer_tool([], []) == "unknown"

def test_score_from_keywords():
    s, m = score_from_keywords(["github copilot"], [])
    assert s == 0.95 and m == "ocr_strong"
    s, m = score_from_keywords([], ["ghcp", "ghost text"])
    assert s == 0.75 and m == "ocr_weak"
    s, m = score_from_keywords([], ["ghcp"])
    assert s == 0.45 and m == "ocr_weak"
    s, m = score_from_keywords([], [])
    assert s == 0.0 and m == "none"
    
def test_write_csv_round_trip(tmp_path: Path):
    out = tmp_path / "out.csv"
    results = [
        CopilotEvidence(
            path="/img/a.png", score=0.95, method="ocr_strong",
            detected_tool="cursor",
            strong_matches=["plan agent"], weak_matches=[],
            ocr_snippet="Plan agent to research\nand plan",
        ),
        CopilotEvidence(
            path="/img/b.png", score=0.82, method="template",
            detected_tool="copilot",
            template_name="dark.png", template_confidence=0.82,
            template_box=(100, 200, 120, 220),
        ),
    ]
    write_csv(results, out)
    text = out.read_text()
    assert "path,score,method" in text
    assert "detected_tool" in text
    assert "cursor" in text
    assert "copilot" in text
    assert "\nand plan" not in text
