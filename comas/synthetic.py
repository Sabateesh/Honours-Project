# Generates synthetic IDE screenshots for the two ways an AI assistant shows up
# on screen: dimmed inline "ghost text" at the cursor, and an assistant chat
# panel docked to the right.
#
# Negatives are not just clean editors. Roughly half of them put some *other*
# panel in that same right hand slot (Outline, Extensions, Source Control,
# Search), because a model trained only against clean editors learns "panel on
# the right = cheating" and then flags every student with the Extensions view
# open. The chat panel has to be told apart by its structure - message bubbles
# stacked above a composer box - not by its position.
#
# Comments stay in the corpus on purpose: they are gray too, so the model can't
# just learn "gray = ghost text".
from __future__ import annotations

import argparse
import logging
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

PY_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield", "self",
}

# Words that would let the OCR keyword detector (or the fusion text branch)
# cheat instead of learning the visual ghost-text signal.
BANNED_WORDS = re.compile(r"copilot|cursor|ghost", re.IGNORECASE)

THEMES = [
    {
        "name": "dark_plus",
        "bg": "#1e1e1e", "sidebar": "#252526", "titlebar": "#3c3c3c",
        "statusbar": "#007acc", "tab_active": "#1e1e1e", "tab_inactive": "#2d2d2d",
        "gutter": "#858585", "gutter_active": "#c6c6c6",
        "default": "#d4d4d4", "keyword": "#569cd6", "string": "#ce9178",
        "comment": "#6a9955", "number": "#b5cea8", "func": "#dcdcaa",
        "ghost": "#6e6e6e", "cursor": "#aeafad", "line_hl": "#2a2a2a",
    },
    {
        "name": "one_dark",
        "bg": "#282c34", "sidebar": "#21252b", "titlebar": "#3a3f4b",
        "statusbar": "#21252b", "tab_active": "#282c34", "tab_inactive": "#21252b",
        "gutter": "#495162", "gutter_active": "#abb2bf",
        "default": "#abb2bf", "keyword": "#c678dd", "string": "#98c379",
        "comment": "#5c6370", "number": "#d19a66", "func": "#61afef",
        "ghost": "#6b7180", "cursor": "#528bff", "line_hl": "#2c313c",
    },
    {
        "name": "monokai",
        "bg": "#272822", "sidebar": "#1e1f1c", "titlebar": "#414339",
        "statusbar": "#414339", "tab_active": "#272822", "tab_inactive": "#34352f",
        "gutter": "#90908a", "gutter_active": "#c2c2bf",
        "default": "#f8f8f2", "keyword": "#f92672", "string": "#e6db74",
        "comment": "#75715e", "number": "#ae81ff", "func": "#a6e22e",
        "ghost": "#7a7a72", "cursor": "#f8f8f0", "line_hl": "#3e3d32",
    },
    {
        "name": "light_plus",
        "bg": "#ffffff", "sidebar": "#f3f3f3", "titlebar": "#dddddd",
        "statusbar": "#007acc", "tab_active": "#ffffff", "tab_inactive": "#ececec",
        "gutter": "#237893", "gutter_active": "#0b216f",
        "default": "#000000", "keyword": "#0000ff", "string": "#a31515",
        "comment": "#008000", "number": "#098658", "func": "#795e26",
        "ghost": "#9b9b9b", "cursor": "#000000", "line_hl": "#f5f5f5",
    },
]

FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]

WINDOW_SIZES = [(1280, 800), (1440, 900), (1512, 982), (1680, 1050)]

# Panel text is deliberately generic. None of it contains the OCR detector's
# keywords, so the CNN cannot pass by reading the panel - it has to see it.
CHAT_PROMPTS = [
    "how do I reverse a linked list in place",
    "why is this loop off by one",
    "explain what this function does",
    "write a unit test for this method",
    "refactor this to use a dictionary",
    "what is the time complexity here",
    "fix the type error on line 42",
]

CHAT_REPLIES = [
    "You can walk the list once and flip each next pointer as you go, "
    "keeping a reference to the previous node.",
    "The range is exclusive at the upper bound, so the final element is "
    "never visited. Widen it by one.",
    "This builds a lookup table first, then does a single pass over the "
    "input instead of a nested scan.",
    "Each comparison halves the remaining search space, so the work grows "
    "logarithmically with the input size.",
    "The argument is annotated as an integer but a string is passed in from "
    "the caller above.",
]

CHAT_SNIPPETS = [
    ["def reverse(head):", "    prev = None", "    while head:",
     "        head.next, prev, head = prev, head, head.next", "    return prev"],
    ["for i in range(len(items)):", "    total += items[i]", "return total"],
    ["seen = {}", "for x in values:", "    seen[x] = seen.get(x, 0) + 1"],
]

COMPOSER_HINTS = ["Ask anything", "Send a message", "Type a question",
                  "Describe what you want"]

NEG_PANELS = ["outline", "extensions", "scm", "search"]

OUTLINE_ROWS = ["main", "load_corpus", "tokenize_line", "render", "Config",
                "__init__", "run", "parse_args", "Detector", "detect", "flush"]
EXT_ROWS = [("Python", "IntelliSense, linting, debugging"),
            ("Pylance", "Fast type checking"),
            ("Jupyter", "Notebook support"),
            ("Docker", "Container tooling"),
            ("GitLens", "Repository insights"),
            ("Black Formatter", "Code formatter")]
SCM_ROWS = [("data.py", "M"), ("train.py", "M"), ("notes.txt", "U"),
            ("model.py", "M"), ("setup.cfg", "A"), ("utils.py", "M")]

TOKEN_RE = re.compile(
    r"""(\#.*$                       # comment to end of line
        |\"\"\"|'''                  # triple quotes
        |\"[^\"]*\"|'[^']*'          # strings
        |\b\d+(?:\.\d+)?\b           # numbers
        |\b[A-Za-z_][A-Za-z0-9_]*\b  # identifiers
        |\s+|.)""",
    re.VERBOSE,
)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def tokenize_line(line: str) -> list[tuple[str, str]]:
    tokens = []
    prev = ""
    for m in TOKEN_RE.finditer(line):
        tok = m.group(0)
        if tok.startswith("#"):
            kind = "comment"
        elif tok.startswith(("\"", "'")):
            kind = "string"
        elif re.fullmatch(r"\d+(?:\.\d+)?", tok):
            kind = "number"
        elif tok in PY_KEYWORDS:
            kind = "keyword"
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            kind = "func" if prev.strip() in {"def", "class"} or line[m.end():m.end() + 1] == "(" else "default"
        else:
            kind = "default"
        tokens.append((tok, kind))
        if tok.strip():
            prev = tok
    return tokens


def load_corpus(roots: list[Path]) -> list[tuple[str, list[str]]]:
    corpus = []
    for root in roots:
        for p in sorted(root.rglob("*.py")):
            if BANNED_WORDS.search(p.name):
                continue
            lines = [
                ln.rstrip()[:110] for ln in p.read_text(errors="ignore").splitlines()
                if not BANNED_WORDS.search(ln)
            ]
            if len(lines) >= 30:
                corpus.append((p.name, lines))
    if not corpus:
        raise RuntimeError(f"No usable .py files under {roots}")
    return corpus


def _draw_chrome(draw, W, H, theme, rng, filenames, active_file, font_ui):
    sidebar_w = rng.choice([0, 200, 230, 260])
    title_h, tab_h, status_h = 28, 34, 24

    draw.rectangle([0, 0, W, title_h], fill=theme["titlebar"])
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        draw.ellipse([12 + i * 20, 9, 22 + i * 20, 19], fill=c)

    if sidebar_w:
        draw.rectangle([0, title_h, sidebar_w, H - status_h], fill=theme["sidebar"])
        draw.text((14, title_h + 10), "EXPLORER", fill=theme["gutter"], font=font_ui)
        y = title_h + 36
        for name in filenames[:18]:
            color = theme["default"] if name != active_file else theme["func"]
            draw.ellipse([16, y + 4, 22, y + 10], fill=theme["keyword"])
            draw.text((30, y), name, fill=color, font=font_ui)
            y += 22

    tab_x = sidebar_w
    tabs = [active_file] + rng.sample([f for f in filenames if f != active_file],
                                      k=min(rng.randint(1, 4), len(filenames) - 1))
    rng.shuffle(tabs)
    draw.rectangle([sidebar_w, title_h, W, title_h + tab_h], fill=theme["tab_inactive"])
    for name in tabs:
        w = int(font_ui.getlength(name)) + 36
        active = name == active_file
        draw.rectangle([tab_x, title_h, tab_x + w, title_h + tab_h],
                       fill=theme["tab_active"] if active else theme["tab_inactive"])
        draw.text((tab_x + 14, title_h + 9), name,
                  fill=theme["default"] if active else theme["gutter"], font=font_ui)
        tab_x += w

    draw.rectangle([0, H - status_h, W, H], fill=theme["statusbar"])
    status_fg = "#ffffff" if theme["name"] != "one_dark" else theme["default"]
    draw.text((12, H - status_h + 5), "main*", fill=status_fg, font=font_ui)
    right = (f"Ln {rng.randint(1, 200)}, Col {rng.randint(1, 60)}   Spaces: 4   "
             f"UTF-8   LF   Python")
    draw.text((W - font_ui.getlength(right) - 16, H - status_h + 5),
              right, fill=status_fg, font=font_ui)

    return sidebar_w, title_h + tab_h, H - status_h


def _wrap(text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if font.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _panel_header(draw, x0, top, W, theme, font_ui, title):
    draw.rectangle([x0, top, W, top + 34], fill=theme["sidebar"])
    draw.text((x0 + 14, top + 10), title, fill=theme["gutter"], font=font_ui)
    draw.line([x0, top + 34, W, top + 34], fill=theme["titlebar"])


def _draw_chat_panel(draw, x0, top, bottom, W, theme, rng, font_ui):
    """Assistant chat: stacked message bubbles above a composer box."""
    draw.rectangle([x0, top, W, bottom], fill=theme["sidebar"])
    _panel_header(draw, x0, top, W, theme, font_ui, "CHAT")

    inner_w = W - x0 - 30
    composer_top = bottom - rng.randint(58, 74)
    y = top + 46

    while y < composer_top - 60:
        prompt = rng.choice(CHAT_PROMPTS)
        lines = _wrap(prompt, font_ui, inner_w - 24)
        h = len(lines) * 16 + 14
        if y + h > composer_top - 30:
            break
        draw.rounded_rectangle([x0 + 12, y, W - 16, y + h], radius=8,
                               fill=theme["line_hl"])
        yy = y + 7
        for ln in lines:
            draw.text((x0 + 22, yy), ln, fill=theme["default"], font=font_ui)
            yy += 16
        y += h + 12

        reply = rng.choice(CHAT_REPLIES)
        for ln in _wrap(reply, font_ui, inner_w):
            if y > composer_top - 24:
                break
            draw.text((x0 + 16, y), ln, fill=theme["default"], font=font_ui)
            y += 16
        y += 8

        if rng.random() < 0.55:
            snippet = rng.choice(CHAT_SNIPPETS)
            bh = len(snippet) * 15 + 12
            if y + bh < composer_top - 20:
                draw.rectangle([x0 + 12, y, W - 16, y + bh], fill=theme["bg"])
                yy = y + 6
                for ln in snippet:
                    draw.text((x0 + 20, yy), ln, fill=theme["string"], font=font_ui)
                    yy += 15
                y += bh + 12
        y += 6

    draw.rounded_rectangle([x0 + 12, composer_top, W - 16, bottom - 14],
                           radius=8, fill=theme["bg"], outline=theme["titlebar"])
    draw.text((x0 + 22, composer_top + 10), rng.choice(COMPOSER_HINTS),
              fill=theme["gutter"], font=font_ui)
    bx = x0 + 22
    for _ in range(rng.randint(2, 3)):
        bw = rng.randint(34, 52)
        draw.rounded_rectangle([bx, bottom - 38, bx + bw, bottom - 22],
                               radius=5, fill=theme["line_hl"])
        bx += bw + 8


def _draw_neg_panel(draw, kind, x0, top, bottom, W, theme, rng, font_ui):
    """A different panel in the same slot - same shape, no chat structure."""
    draw.rectangle([x0, top, W, bottom], fill=theme["sidebar"])
    titles = {"outline": "OUTLINE", "extensions": "EXTENSIONS",
              "scm": "SOURCE CONTROL", "search": "SEARCH"}
    _panel_header(draw, x0, top, W, theme, font_ui, titles[kind])
    y = top + 46

    if kind == "search":
        draw.rounded_rectangle([x0 + 12, y, W - 16, y + 24], radius=4,
                               fill=theme["bg"], outline=theme["titlebar"])
        draw.text((x0 + 20, y + 5), rng.choice(["def ", "self.", "return", "import"]),
                  fill=theme["default"], font=font_ui)
        y += 38
        for _ in range(rng.randint(4, 9)):
            if y > bottom - 20:
                break
            name = rng.choice(OUTLINE_ROWS) + ".py"
            draw.text((x0 + 16, y), name, fill=theme["func"], font=font_ui)
            y += 18
            for _ in range(rng.randint(1, 3)):
                if y > bottom - 20:
                    break
                draw.text((x0 + 32, y), rng.choice(OUTLINE_ROWS),
                          fill=theme["gutter"], font=font_ui)
                y += 16
            y += 4
    elif kind == "extensions":
        for name, desc in rng.sample(EXT_ROWS, k=min(len(EXT_ROWS), rng.randint(3, 6))):
            if y > bottom - 44:
                break
            draw.rectangle([x0 + 14, y, x0 + 40, y + 26], fill=theme["line_hl"])
            draw.text((x0 + 50, y), name, fill=theme["default"], font=font_ui)
            draw.text((x0 + 50, y + 15), desc[:34], fill=theme["gutter"], font=font_ui)
            draw.rounded_rectangle([W - 70, y + 4, W - 20, y + 22], radius=4,
                                   fill=theme["statusbar"])
            y += 40
    elif kind == "scm":
        for name, badge in rng.sample(SCM_ROWS, k=min(len(SCM_ROWS), rng.randint(3, 6))):
            if y > bottom - 24:
                break
            draw.text((x0 + 20, y), name, fill=theme["default"], font=font_ui)
            draw.text((W - 30, y), badge, fill=theme["keyword"], font=font_ui)
            y += 22
    else:
        for _ in range(rng.randint(6, 14)):
            if y > bottom - 20:
                break
            indent = rng.choice([0, 14, 28])
            draw.ellipse([x0 + 16 + indent, y + 4, x0 + 24 + indent, y + 12],
                         fill=theme["keyword"])
            draw.text((x0 + 32 + indent, y), rng.choice(OUTLINE_ROWS),
                      fill=theme["default"], font=font_ui)
            y += 20


def _draw_code_line(draw, x, y, tokens, theme, font, ghost=False):
    for tok, kind in tokens:
        color = theme["ghost"] if ghost else theme[kind]
        draw.text((x, y), tok, fill=color, font=font)
        x += font.getlength(tok)
    return x


def render_screenshot(file_name, lines, start, cursor_line, ghost_lines,
                      theme, rng, filenames, panel=None) -> Image.Image:
    W, H = rng.choice(WINDOW_SIZES)
    font_size = rng.randint(12, 15)
    font = _load_font(font_size)
    font_ui = _load_font(12)
    line_h = font_size + 7

    img = Image.new("RGB", (W, H), theme["bg"])
    draw = ImageDraw.Draw(img)
    sidebar_w, top, bottom = _draw_chrome(draw, W, H, theme, rng,
                                          filenames, file_name, font_ui)

    # A docked panel takes width away from the editor rather than covering it.
    panel_w = rng.randint(300, 430) if panel else 0
    panel_x = W - panel_w
    edit_right = panel_x if panel else W

    gutter_w = 58
    code_x = sidebar_w + gutter_w + 12
    minimap_w = 0 if panel and panel_w > 360 else 90
    n_visible = (bottom - top - 16) // line_h

    y = top + 12
    row = 0
    idx = start
    cursor_x = None
    while row < n_visible and idx < len(lines):
        ghost_here = ghost_lines and idx == cursor_line
        lineno = str(idx + 1)
        is_cursor = idx == cursor_line
        if is_cursor:
            draw.rectangle([sidebar_w, y - 2, edit_right - minimap_w, y + line_h - 2],
                           fill=theme["line_hl"])
        draw.text((sidebar_w + gutter_w - font.getlength(lineno), y), lineno,
                  fill=theme["gutter_active"] if is_cursor else theme["gutter"],
                  font=font)

        if ghost_here:
            # typed prefix in normal colors, suggestion continues dimmed
            prefix, first_ghost = ghost_lines[0]
            x_end = _draw_code_line(draw, code_x, y, tokenize_line(prefix), theme, font)
            cursor_x = x_end
            _draw_code_line(draw, x_end, y, [(first_ghost, "default")],
                            theme, font, ghost=True)
            for cont in ghost_lines[1:]:
                row += 1
                y += line_h
                if row >= n_visible:
                    break
                _draw_code_line(draw, code_x, y, [(cont, "default")],
                                theme, font, ghost=True)
        else:
            x_end = _draw_code_line(draw, code_x, y, tokenize_line(lines[idx]),
                                    theme, font)
            if is_cursor:
                cursor_x = x_end

        row += 1
        y += line_h
        idx += 1

    if cursor_x is not None:
        cy = top + 12 + (cursor_line - start) * line_h
        draw.rectangle([cursor_x, cy - 1, cursor_x + 2, cy + line_h - 3],
                       fill=theme["cursor"])
    if minimap_w:
        mm_x = edit_right - minimap_w
        for i, ln in enumerate(lines[max(0, start - 40):start + 120]):
            my = top + 4 + i * 3
            if my > bottom - 6:
                break
            w = min(int(len(ln) * 0.7), minimap_w - 12)
            if w > 0:
                draw.rectangle([mm_x + 6, my, mm_x + 6 + w, my + 1],
                               fill=theme["gutter"])

    if panel == "chat":
        _draw_chat_panel(draw, panel_x, top, bottom, W, theme, rng, font_ui)
    elif panel:
        _draw_neg_panel(draw, panel, panel_x, top, bottom, W, theme, rng, font_ui)
    return img


def make_sample(corpus, rng, variant: str):
    """variant: ghost | panel | both | clean | hardneg"""
    file_name, lines = rng.choice(corpus)
    filenames = [name for name, _ in corpus]
    theme = rng.choice(THEMES)

    start = rng.randint(0, max(0, len(lines) - 40))
    cursor_line = min(start + rng.randint(3, 30), len(lines) - 2)

    with_ghost = variant in ("ghost", "both")
    if variant in ("panel", "both"):
        panel = "chat"
    elif variant == "hardneg":
        panel = rng.choice(NEG_PANELS)
    else:
        panel = None

    ghost = []
    if with_ghost:
        full = lines[cursor_line]
        stripped = len(full) - len(full.lstrip())
        cut = rng.randint(stripped, max(stripped, len(full) - 1)) if full.strip() else stripped
        ghost = [(full[:cut], full[cut:] or " ")]
        n_more = rng.choices([0, 1, 2, 3, 5], weights=[2, 3, 3, 2, 1])[0]
        if n_more:
            _, other = rng.choice(corpus)
            at = rng.randint(0, len(other) - n_more - 1)
            indent = " " * (len(full) - len(full.lstrip()))
            for extra in other[at:at + n_more]:
                ghost.append(indent + extra.strip() if extra.strip() else " ")
    return render_screenshot(file_name, lines, start, cursor_line, ghost,
                             theme, rng, filenames, panel=panel)


def _plan(n_active: int, n_clean: int, hard_neg_frac: float, rng):
    """Positives split across the two ways an assistant shows up; a chunk of the
    negatives carry a non-chat panel so position alone is not a giveaway."""
    jobs = []
    for i in range(n_active):
        r = i / max(1, n_active)
        variant = "ghost" if r < 0.40 else ("panel" if r < 0.80 else "both")
        jobs.append(("copilot_active", variant))
    n_hard = int(n_clean * hard_neg_frac)
    jobs += [("no_copilot", "hardneg")] * n_hard
    jobs += [("no_copilot", "clean")] * (n_clean - n_hard)
    rng.shuffle(jobs)
    return jobs


def generate(out_dir: Path, n_active: int, n_clean: int, sessions: int,
             seed: int, corpus_roots: list[Path], hard_neg_frac: float = 0.5):
    rng = random.Random(seed)
    corpus = load_corpus(corpus_roots)
    log.info("Corpus: %d files", len(corpus))

    jobs = _plan(n_active, n_clean, hard_neg_frac, rng)

    counters: dict[str, int] = {}
    tally: dict[str, int] = {}
    for i, (label, variant) in enumerate(jobs):
        session = f"synth{i % sessions:02d}"
        counters[label] = counters.get(label, 0) + 1
        tally[variant] = tally.get(variant, 0) + 1
        folder = out_dir / label
        folder.mkdir(parents=True, exist_ok=True)
        img = make_sample(corpus, rng, variant)
        img.save(folder / f"{session}_{counters[label]:04d}.png")
        if (i + 1) % 50 == 0:
            log.info("%d / %d", i + 1, len(jobs))

    log.info("Done: %d active / %d clean under %s", n_active, n_clean, out_dir)
    log.info("Variants: %s", ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/"))
    ap.add_argument("--n-active", type=int, default=150)
    ap.add_argument("--n-clean", type=int, default=150)
    ap.add_argument("--sessions", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", type=Path, nargs="*",
                    default=[Path("comas"), Path("tests")])
    ap.add_argument("--hard-neg-frac", type=float, default=0.5,
                    help="share of negatives that show a non-chat side panel")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    generate(args.out, args.n_active, args.n_clean, args.sessions,
             args.seed, args.corpus, args.hard_neg_frac)


if __name__ == "__main__":
    main()
