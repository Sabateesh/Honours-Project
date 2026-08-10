from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

RESOLUTIONS = [(1280, 800), (1440, 900), (1680, 1050), (1920, 1080)]

QUIZ_TITLES = ["COVID-19 Course Quiz", "COMP 2402 Midterm", "Residence Orientation Quiz",
               "MATH 1104 Quiz 3", "PSYC 1001 Weekly Quiz"]

QUIZ_QUESTIONS = [
    ("If you test positive for COVID-19, what should you do?",
     ["Take a nap - I'll feel better in the morning.",
      "Complete the cuScreen Symptom Reporting Tool, call my doctor",
      "Stay in my room and contact the Reception Desk",
      "Tell my Residence Fellow and continue my day"]),
    ("What is the running time of binary search?",
     ["O(n)", "O(log n)", "O(n log n)", "O(1)"]),
    ("True or False: Residents must follow all posted guidelines.",
     ["True", "False"]),
    ("Which data structure uses LIFO ordering?",
     ["Queue", "Stack", "Heap", "Graph"]),
    ("Where can I check updated information about protocols?",
     ["Instagram page", "Residence Standards", "Facebook group",
      "Housing and Residence Life Services website"]),
]

GOOGLE_RESULTS = [
    ("What should you do if you test positive for covid",
     ["cuScreen Symptom Reporting Tool - Carleton University",
      "COVID-19 protocols and guidelines | Ottawa Public Health",
      "What to do if you test positive - Reddit r/CarletonU",
      "COVID-19 self-isolation rules explained"]),
    ("binary search running time",
     ["Binary search algorithm - Wikipedia",
      "Time complexity of binary search - Stack Overflow",
      "Big O notation explained with examples",
      "Binary Search - GeeksforGeeks"]),
    ("which data structure uses lifo",
     ["Stack (abstract data type) - Wikipedia",
      "LIFO vs FIFO differences - Stack Overflow",
      "Stacks and Queues tutorial",
      "Data structures cheat sheet"]),
]

CHATGPT_CONVOS = [
    [("You", "what should i do if i test positive for covid at carleton residence"),
     ("ChatGPT", "You should stay in your room, complete the cuScreen"),
     ("ChatGPT", "Symptom Reporting Tool, and contact the Residence"),
     ("ChatGPT", "Reception Desk for next steps.")],
    [("You", "whats the time complexity of binary search"),
     ("ChatGPT", "Binary search runs in O(log n) time because it"),
     ("ChatGPT", "halves the search space on every comparison.")],
    [("You", "which data structure is LIFO"),
     ("ChatGPT", "A stack uses Last-In-First-Out (LIFO) ordering."),
     ("ChatGPT", "The most recently pushed element is popped first.")],
]

SO_QUESTIONS = [
    ("Time complexity of binary search on sorted array",
     "The complexity is O(log n) since each step halves the range..."),
    ("Difference between stack and queue in practice",
     "A stack is LIFO, a queue is FIFO. Use a stack when..."),
    ("How does lru_cache memoization work in Python",
     "functools.lru_cache stores results keyed by arguments..."),
]

WIKI_ARTICLES = [
    ("Binary search algorithm",
     "In computer science, binary search is a search algorithm that finds "
     "the position of a target value within a sorted array."),
    ("Stack (abstract data type)",
     "A stack is an abstract data type that serves as a collection of "
     "elements with two main operations: push and pop."),
]


def _font(size, bold=False):
    names = ["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_browser_chrome(draw, w, rng, url, active_title, extra_tabs=None):
    draw.rectangle([(0, 0), (w, 40)], fill=(50, 52, 56))
    tabs = [(active_title, True)] + [(t, False) for t in (extra_tabs or [])]
    x = 80
    for title, active in tabs:
        tab_w = min(220, 30 + len(title) * 7)
        fill = (70, 72, 78) if active else (55, 57, 62)
        draw.rectangle([(x, 8), (x + tab_w, 40)], fill=fill)
        draw.text((x + 10, 16), title[:26], fill=(230, 230, 230), font=_font(11))
        x += tab_w + 4
    draw.rectangle([(0, 40), (w, 76)], fill=(60, 62, 68))
    draw.rounded_rectangle([(90, 46), (w - 200, 70)], radius=12,
                           fill=(40, 42, 46))
    draw.text((104, 51), url, fill=(200, 200, 205), font=_font(12))
    return 76


def _draw_quiz_page(draw, w, h, top, rng):
    draw.rectangle([(0, top), (w, h)], fill=(255, 255, 255))
    title = rng.choice(QUIZ_TITLES)
    draw.text((w // 4, top + 24), title, fill=(20, 20, 20), font=_font(20, bold=True))
    mins = rng.randint(0, 40)
    draw.text((w - 320, top + 28), f"0:{mins:02d}:{rng.randint(0,59):02d} elapsed",
              fill=(60, 60, 60), font=_font(13))
    draw.text((120, top + 80), "Page 1:", fill=(20, 20, 20), font=_font(14, bold=True))
    for i in range(5):
        bx = 120 + (i % 3) * 52
        by = top + 110 + (i // 3) * 60
        draw.rectangle([(bx, by), (bx + 40, by + 46)], outline=(150, 150, 150))
        draw.text((bx + 15, by + 6), str(i + 1), fill=(37, 99, 235), font=_font(13))
    draw.text((120, top + 250), "Quiz Information", fill=(37, 99, 235), font=_font(12))

    x = w // 3
    y = top + 90
    n_q = rng.randint(2, 3)
    for qi, (q, opts) in enumerate(rng.sample(QUIZ_QUESTIONS, k=n_q)):
        draw.text((x, y), f"Question {qi + 1} (1 point)", fill=(20, 20, 20),
                  font=_font(15, bold=True))
        y += 30
        draw.text((x + 10, y), q[: (w - x - 60) // 8], fill=(30, 30, 30), font=_font(13))
        y += 34
        for opt in opts:
            draw.ellipse([(x + 14, y + 2), (x + 28, y + 16)], outline=(120, 120, 120))
            draw.text((x + 38, y), opt[: (w - x - 80) // 8], fill=(30, 30, 30),
                      font=_font(13))
            y += 28
        y += 22
        if y > h - 160:
            break

    by = min(h - 70, y + 20)
    draw.rounded_rectangle([(x, by), (x + 120, by + 34)], radius=4,
                           fill=(37, 99, 235))
    draw.text((x + 22, by + 8), "Submit Quiz", fill=(255, 255, 255), font=_font(12))
    draw.text((x + 140, by + 10), f"{rng.randint(0, n_q)} of 5 questions saved",
              fill=(80, 80, 80), font=_font(12))


def _draw_google(draw, w, h, top, rng):
    draw.rectangle([(0, top), (w, h)], fill=(255, 255, 255))
    query, results = rng.choice(GOOGLE_RESULTS)
    draw.text((160, top + 20), "Google", fill=(66, 103, 244), font=_font(22, bold=True))
    draw.rounded_rectangle([(280, top + 16), (w - 300, top + 52)], radius=18,
                           outline=(200, 200, 200))
    draw.text((300, top + 26), query, fill=(40, 40, 40), font=_font(13))
    draw.text((160, top + 80), f"About {rng.randint(1, 90)},{rng.randint(100, 999)},000 results",
              fill=(110, 110, 110), font=_font(11))
    y = top + 120
    for r in results:
        draw.text((160, y), r, fill=(26, 13, 171), font=_font(15))
        draw.text((160, y + 22), "https://" + r.split(" ")[0].lower() + ".example.com",
                  fill=(0, 102, 33), font=_font(11))
        draw.text((160, y + 40),
                  "Learn more about this topic with detailed explanations and examples...",
                  fill=(80, 80, 80), font=_font(12))
        y += 84


def _draw_chatgpt(draw, w, h, top, rng):
    draw.rectangle([(0, top), (w, h)], fill=(52, 53, 65))
    draw.rectangle([(0, top), (240, h)], fill=(32, 33, 40))
    draw.text((20, top + 16), "ChatGPT", fill=(230, 230, 230), font=_font(15, bold=True))
    draw.text((20, top + 50), "+ New chat", fill=(200, 200, 200), font=_font(12))
    convo = rng.choice(CHATGPT_CONVOS)
    y = top + 40
    for who, msg in convo:
        col = (230, 230, 230) if who == "You" else (190, 220, 190)
        draw.text((280, y), f"{who}:", fill=col, font=_font(13, bold=True))
        draw.text((360, y), msg[: (w - 400) // 8], fill=(220, 220, 225), font=_font(13))
        y += 44
    draw.rounded_rectangle([(280, h - 70), (w - 60, h - 30)], radius=8,
                           outline=(120, 120, 130))
    draw.text((296, h - 60), "Message ChatGPT", fill=(140, 140, 150), font=_font(12))


def _draw_stackoverflow(draw, w, h, top, rng):
    draw.rectangle([(0, top), (w, h)], fill=(255, 255, 255))
    draw.rectangle([(0, top), (w, top + 50)], fill=(244, 128, 36))
    draw.text((40, top + 14), "stack overflow", fill=(255, 255, 255),
              font=_font(16, bold=True))
    q, a = rng.choice(SO_QUESTIONS)
    draw.text((60, top + 80), q, fill=(59, 64, 69), font=_font(18, bold=True))
    draw.text((60, top + 116), f"Asked {rng.randint(2, 9)} years ago    "
              f"Viewed {rng.randint(10, 900)}k times",
              fill=(110, 110, 110), font=_font(11))
    draw.text((100, top + 170), str(rng.randint(50, 4000)), fill=(59, 64, 69),
              font=_font(18))
    draw.text((160, top + 170), a[: (w - 220) // 8], fill=(40, 40, 40), font=_font(13))
    draw.rectangle([(160, top + 210), (w - 100, top + 300)], fill=(246, 246, 246))
    draw.text((175, top + 226), "lo, hi = 0, len(arr) - 1", fill=(30, 30, 30),
              font=_font(12))
    draw.text((175, top + 246), "while lo <= hi:", fill=(30, 30, 30), font=_font(12))
    draw.text((175, top + 266), "    mid = (lo + hi) // 2", fill=(30, 30, 30),
              font=_font(12))


def _draw_wikipedia(draw, w, h, top, rng):
    draw.rectangle([(0, top), (w, h)], fill=(255, 255, 255))
    title, body = rng.choice(WIKI_ARTICLES)
    draw.text((180, top + 30), title, fill=(20, 20, 20), font=_font(24))
    draw.line([(180, top + 66), (w - 180, top + 66)], fill=(200, 200, 200))
    draw.text((180, top + 80), "From Wikipedia, the free encyclopedia",
              fill=(100, 100, 100), font=_font(11))
    words = body.split()
    y = top + 120
    line = ""
    for word in words:
        if len(line) + len(word) > (w - 400) // 8:
            draw.text((180, y), line, fill=(30, 30, 30), font=_font(14))
            y += 24
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        draw.text((180, y), line, fill=(30, 30, 30), font=_font(14))


OFF_TASK_SITES = [
    ("google.com/search?q=quiz+answer", "quiz answer - Google Search", _draw_google),
    ("chat.openai.com", "ChatGPT", _draw_chatgpt),
    ("stackoverflow.com/questions/48591", "algorithms - Stack Overflow", _draw_stackoverflow),
    ("en.wikipedia.org/wiki/Binary_search", "Binary search - Wikipedia", _draw_wikipedia),
]

QUIZ_URL = "brightspace.carleton.ca/d2l/lms/quizzing/user/attempt/quiz_start_frame"


def generate_screenshot(on_quiz: bool, rng: random.Random) -> Image.Image:
    w, h = rng.choice(RESOLUTIONS)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if on_quiz:
        extra = rng.sample(["Gmail", "Carleton Central", "Calendar"],
                           k=rng.randint(0, 2))
        top = _draw_browser_chrome(draw, w, rng, QUIZ_URL,
                                   "Quizzes - 2022 Residence", extra)
        _draw_quiz_page(draw, w, h, top, rng)
    else:
        url, title, renderer = rng.choice(OFF_TASK_SITES)
        extra = []
        if rng.random() < 0.6:
            extra.append("Quizzes - 2022 Residence")
        if rng.random() < 0.3:
            extra.append("Gmail")
        top = _draw_browser_chrome(draw, w, rng, url, title, extra)
        renderer(draw, w, h, top, rng)

    return img


def generate(out_dir: Path, n_on: int, n_off: int, seed: int = 0) -> None:
    rng = random.Random(seed)
    on_dir = out_dir / "on_quiz"
    off_dir = out_dir / "left_quiz"
    on_dir.mkdir(parents=True, exist_ok=True)
    off_dir.mkdir(parents=True, exist_ok=True)

    n_sess_on = max(1, n_on // 5)
    n_sess_off = max(1, n_off // 5)

    for i in range(n_on):
        img = generate_screenshot(True, rng)
        img.save(on_dir / f"bq{i % n_sess_on}_{i:04d}.png")

    for i in range(n_off):
        img = generate_screenshot(False, rng)
        img.save(off_dir / f"bl{i % n_sess_off}_{i:04d}.png")

    log.info("Generated %d on_quiz, %d left_quiz in %s", n_on, n_off, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data_brightspace/"))
    ap.add_argument("--n-on", type=int, default=200)
    ap.add_argument("--n-off", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate(args.out, args.n_on, args.n_off, args.seed)


if __name__ == "__main__":
    main()