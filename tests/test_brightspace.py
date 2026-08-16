from comas.brightspace import find_markers, looks_like_ide, score_markers


def test_on_quiz_wins_even_with_off_task_words():
    # a quiz question may legitimately say "cite Wikipedia" or "google it"
    on, off = find_markers(
        "Brightspace  COMP 2402 Quiz  (1 point)  "
        "Which source is acceptable to cite: Wikipedia or a textbook?")
    assert on and off
    score, verdict = score_markers(on, off)
    assert verdict == "on_quiz"
    assert score == 0.05


def test_anything_without_quiz_markers_is_a_tab_leave():
    # binary rule: during a Brightspace exam the quiz should be on screen
    for text in ["ChatGPT  what is the time complexity of quicksort",
                 "My vacation photos - Google Photos",
                 ""]:
        on, off = find_markers(text)
        score, verdict = score_markers(on, off)
        assert verdict == "left_quiz"
        assert score == 0.95


def test_off_task_sites_still_named_for_the_evidence_line():
    on, off = find_markers("Stack Overflow - time complexity of binary search")
    assert "stack overflow" in off
    # the site is annotation, not the verdict driver
    _, verdict = score_markers(on, off)
    assert verdict == "left_quiz"


def test_quiz_markers_detected():
    on, off = find_markers("Submit Quiz   0:12:33 elapsed   Questions saved")
    assert "submit quiz" in on
    assert "elapsed" in on


def test_looks_like_ide_on_vscode_chrome():
    assert looks_like_ide(
        "EXPLORER  copilot.py  data.py  TERMINAL  Ln 42, Col 7  "
        "Spaces: 4  UTF-8  Python")
    assert not looks_like_ide("Brightspace  Submit Quiz  (1 point)")
    assert not looks_like_ide("")
    # one weak hit is not enough - a quiz about Python mentions .py too
    assert not looks_like_ide("Question 3: what does script.py print?")
