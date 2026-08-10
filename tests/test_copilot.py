from pathlib import Path

from comas.copilot import (
    CopilotEvidence,
    find_keywords,
    infer_tool,
    score_from_keywords,
    write_csv,
)

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
