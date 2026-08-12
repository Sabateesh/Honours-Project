import csv
import queue
import shutil
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

from PIL import Image, ImageTk

from .brightspace import BrightspaceDetector
from .config import load_config
from .copilot import CopilotDetector, build_ml_scorer, ml_is_confident
from .ocr import DEFAULT_WORKERS as OCR_WORKERS
from .ocr import OCRCache

# Carleton University brand palette (Pantone 185 red, black wordmark, white ground)
BG = "#ffffff"
SURFACE = "#f4f4f5"
BORDER = "#e0e0e0"
FG = "#111111"
DIM = "#6e6e6e"
RED = "#e91c24"
RED_TINT = "#fbe6e7"
BLACK = "#000000"

HEADING_FONT = ("Georgia", 20, "bold")
SUBHEAD_FONT = ("Georgia", 12)
BODY_FONT = ("Helvetica", 12)
SMALL_FONT = ("Helvetica", 10)

MODES = ["VSCODE Cheating", "Brightspace Cheating", "VSCODE + Brightspace Cheating"]
FLAG_THRESHOLD = 0.5
VIEW_W, VIEW_H = 860, 480


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CoMas Screenshot Triage")
        self.geometry("1040x740")
        self.configure(bg=BG)

        cfg_path = Path("config.yaml")
        self.cfg = load_config(cfg_path if cfg_path.exists() else None)
        self.ocr_cache = OCRCache(self.cfg.paths.ocr_cache)

        self.mode = None
        self.results = []
        self.result_idx = 0
        self._queue = queue.Queue()
        self._photo = None

        # ghost-text CNN scorer, built lazily on first VS Code scan
        self._ml_scorer = None
        self._ml_scorer_built = False
        self._ml_status = "not loaded"

        tk.Frame(self, bg=RED, height=5).pack(fill="x")
        self._build_header()
        self.body = tk.Frame(self, bg=BG)
        self.body.pack(fill="both", expand=True)
        self._build_home()

    # ----- header ----------------------------------------------------------

    def _build_header(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x")

        inner = tk.Frame(header, bg=BG)
        inner.pack(fill="x", padx=24, pady=(18, 14))

        logo_path = Path(__file__).parent / "assets" / "carleton_logo.png"
        if logo_path.exists():
            img = Image.open(logo_path)
            img.thumbnail((230, 50))
            self._logo = ImageTk.PhotoImage(img)
            tk.Label(inner, image=self._logo, bg=BG).pack(side="left")
        else:
            wordmark = tk.Frame(inner, bg=BG)
            wordmark.pack(side="left")
            tk.Label(wordmark, text="Carleton", bg=BG, fg=BLACK,
                     font=("Georgia", 22, "bold")).pack(side="left")
            tk.Label(wordmark, text=" University", bg=BG, fg=BLACK,
                     font=("Georgia", 22)).pack(side="left")
            tk.Frame(wordmark, bg=RED, width=4, height=26).pack(
                side="left", padx=(10, 0))

        title = tk.Frame(inner, bg=BG)
        title.pack(side="right")
        tk.Label(title, text="CoMas Screenshot Triage", bg=BG, fg=FG,
                 font=("Helvetica", 13, "bold"), anchor="e").pack(anchor="e")
        tk.Label(title, text="AI-assistant & tab-leave detection", bg=BG,
                 fg=DIM, font=SMALL_FONT, anchor="e").pack(anchor="e")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

    # ----- home screen ------------------------------------------------------

    def _build_home(self):
        self._clear_body()

        tk.Label(self.body, text="Select what to detect", bg=BG, fg=FG,
                 font=HEADING_FONT).pack(pady=(64, 4))
        tk.Label(self.body, text="Choose a mode, then upload the screenshots "
                                  "CoMas captured during the exam.",
                 bg=BG, fg=DIM, font=BODY_FONT).pack(pady=(0, 28))

        row = tk.Frame(self.body, bg=BG)
        row.pack()
        self.mode_buttons = {}
        for m in MODES:
            b = tk.Button(row, text=m, command=lambda m=m: self._select_mode(m),
                          bg=SURFACE, fg=FG, activebackground=RED,
                          activeforeground="#ffffff", font=("Helvetica", 12, "bold"),
                          padx=20, pady=14, borderwidth=1, relief="solid",
                          highlightthickness=0, cursor="hand2")
            b.pack(side="left", padx=8)
            b.bind("<Enter>", lambda e, b=b: self._mode_hover(b, True))
            b.bind("<Leave>", lambda e, b=b: self._mode_hover(b, False))
            self.mode_buttons[m] = b

        self.upload_btn = tk.Button(
            self.body, text="Upload CoMas Screenshots", command=self._pick_files,
            bg=SURFACE, fg="#aaaaaa", activebackground=RED,
            activeforeground="#ffffff", font=("Helvetica", 13, "bold"),
            padx=26, pady=13, borderwidth=0, state=tk.DISABLED)
        self.upload_btn.pack(pady=(44, 16))

        self.status = tk.Label(self.body, text="Choose a detection mode first",
                               bg=BG, fg=DIM, font=BODY_FONT)
        self.status.pack()

    def _mode_hover(self, button, entering):
        if button["bg"] == RED:
            return
        button.config(bg=RED_TINT if entering else SURFACE)

    def _select_mode(self, mode):
        self.mode = mode
        for m, b in self.mode_buttons.items():
            active = (m == mode)
            b.config(bg=RED if active else SURFACE,
                     fg="#ffffff" if active else FG)
        self.upload_btn.config(state=tk.NORMAL, bg=BLACK, fg="#ffffff",
                               cursor="hand2")
        self.status.config(text=f"Mode: {mode} — ready to upload", fg=FG)

    # ----- detection --------------------------------------------------------

    def _pick_files(self):
        paths = filedialog.askopenfilenames(
            title="Select CoMas screenshots",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if not paths:
            return
        paths = [Path(p) for p in paths]
        self.upload_btn.config(state=tk.DISABLED, bg=SURFACE, fg="#aaaaaa")
        for b in self.mode_buttons.values():
            b.config(state=tk.DISABLED)
        signal_note = "OCR + ghost-text model" if "VSCODE" in self.mode else "OCR"
        self.status.config(text=f"Analyzing {len(paths)} screenshot(s)... "
                                f"first run is slow ({signal_note})", fg=DIM)
        threading.Thread(target=self._run_detection, args=(paths,),
                         daemon=True).start()
        self.after(100, self._poll_queue)

    def _get_ml_scorer(self):
        if not self._ml_scorer_built:
            self._ml_scorer_built = True
            cfg_path = Path("config.yaml")
            try:
                self._ml_scorer = build_ml_scorer(cfg_path if cfg_path.exists() else None)
                self._ml_status = "loaded" if self._ml_scorer else "no trained checkpoint found"
            except Exception as e:
                self._ml_scorer = None
                self._ml_status = f"failed to load ({e})"
        return self._ml_scorer

    def _run_detection(self, paths):
        use_copilot = "VSCODE" in self.mode
        use_brightspace = "Brightspace" in self.mode

        copilot = None
        if use_copilot:
            ml_scorer = self._get_ml_scorer()
            self._queue.put(("model_status", self._ml_status))
            copilot = CopilotDetector(ocr_cache=self.ocr_cache, ml_scorer=ml_scorer)
        brightspace = BrightspaceDetector(self.ocr_cache) if use_brightspace else None

        # OCR the batch in parallel up front; the loop below then reads from cache.
        def _ocr_progress(done, total):
            self._queue.put(("ocr", done, total))

        ml_scores = {}
        if copilot:
            # Score with the model first: anything it is already sure about does
            # not need the expensive OCR pass. One forward pass per 16 images.
            if copilot.ml_scorer is not None:
                try:
                    raw = copilot.ml_scorer.score_batch(
                        paths, progress=lambda d, t: self._queue.put(("model", d, t)))
                    ml_scores = {p: raw.get(str(p)) for p in paths}
                except Exception as e:
                    self._queue.put(("model_status", f"scoring failed ({e})"))
            needs_ocr = [p for p in paths if not ml_is_confident(ml_scores.get(p))]
            # Assistant panels can sit anywhere on screen, so this one needs full frames.
            self.ocr_cache.extract_many(needs_ocr, progress=_ocr_progress)
        if brightspace:
            brightspace.prefetch(paths, progress=None if copilot else _ocr_progress)

        results = []
        for i, p in enumerate(paths):
            r = {"path": p, "score": 0.0, "copilot_score": None,
                 "bs_score": None, "bs_verdict": None, "bs_sites": "",
                 "status": ""}
            if copilot:
                ev = copilot.detect(p, ml_score=ml_scores.get(p))
                r["copilot_score"] = ev.score
                r["score"] = max(r["score"], ev.score)
            if brightspace:
                ev = brightspace.detect(p)
                r["bs_score"] = ev.score
                r["bs_verdict"] = ev.verdict
                r["bs_sites"] = ",".join(ev.off_task_sites)
                r["score"] = max(r["score"], ev.score)
            results.append(r)
            self._queue.put(("progress", i + 1, len(paths)))

        self.ocr_cache.flush()
        results.sort(key=lambda r: r["score"], reverse=True)
        self._queue.put(("done", results))

    @staticmethod
    def _detail_text(r, threshold):
        bits = []
        if r["copilot_score"] is not None and r["copilot_score"] >= threshold:
            bits.append("AI coding assistant detected")
        if r["bs_score"] is not None and r["bs_score"] >= threshold:
            sites = r["bs_sites"] or r["bs_verdict"]
            bits.append(f"Brightspace: {r['bs_verdict']} ({sites})")
        return "; ".join(bits) or "no signal above threshold"

    def _poll_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg[0] == "model_status":
                    suffix = f" — ghost-text model: {msg[1]}" if "VSCODE" in (self.mode or "") else ""
                    self.status.config(text=f"Analyzing...{suffix}")
                elif msg[0] == "model":
                    self.status.config(text=f"Scoring image {msg[1]} of {msg[2]}...")
                elif msg[0] == "ocr":
                    self.status.config(text=f"Reading text {msg[1]} of {msg[2]} "
                                            f"(using {OCR_WORKERS} cores)...")
                elif msg[0] == "progress":
                    self.status.config(text=f"Analyzed {msg[1]} of {msg[2]}...")
                elif msg[0] == "done":
                    self._show_results(msg[1])
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ----- results viewer ---------------------------------------------------

    def _show_results(self, results):
        self.all_results = results
        self.threshold = FLAG_THRESHOLD
        self.result_idx = 0
        self._clear_body()

        self.summary = tk.Label(self.body, text="", bg=BG, fg=FG,
                                font=("Helvetica", 15, "bold"))
        self.summary.pack(pady=(14, 2))

        thr_row = tk.Frame(self.body, bg=BG)
        thr_row.pack()
        tk.Label(thr_row, text="Flag threshold", bg=BG, fg=DIM,
                 font=SMALL_FONT).pack(side="left", padx=(0, 8))
        self.thr_slider = tk.Scale(
            thr_row, from_=0.05, to=0.95, resolution=0.05,
            orient="horizontal", length=220, bg=BG, fg=FG,
            troughcolor=SURFACE, highlightthickness=0, bd=0,
            font=SMALL_FONT, command=self._on_threshold)
        self.thr_slider.set(FLAG_THRESHOLD)
        self.thr_slider.pack(side="left")

        self.viewer_info = tk.Label(self.body, text="", bg=BG, fg=DIM,
                                    font=BODY_FONT)
        self.viewer_info.pack()

        frame = tk.Frame(self.body, bg=BORDER, bd=0)
        frame.pack(pady=10)
        self.canvas = tk.Label(frame, bg=SURFACE)
        self.canvas.pack(padx=1, pady=1)

        self.detail = tk.Label(self.body, text="", bg=BG, fg=RED,
                               font=("Helvetica", 12, "bold"), wraplength=860)
        self.detail.pack(pady=(2, 0))

        triage = tk.Frame(self.body, bg=BG)
        triage.pack(pady=(8, 0))
        tk.Button(triage, text="Mark reviewed", command=lambda: self._set_status("reviewed"),
                  bg=SURFACE, fg=FG, font=SMALL_FONT, padx=14, pady=6,
                  borderwidth=1, relief="solid",
                  activebackground="#dcfce7", activeforeground=FG).pack(side="left", padx=5)
        tk.Button(triage, text="Dismiss", command=lambda: self._set_status("dismissed"),
                  bg=SURFACE, fg=FG, font=SMALL_FONT, padx=14, pady=6,
                  borderwidth=1, relief="solid",
                  activebackground=RED_TINT, activeforeground=FG).pack(side="left", padx=5)
        tk.Button(triage, text="Clear mark", command=lambda: self._set_status(""),
                  bg=BG, fg=DIM, font=SMALL_FONT, padx=10, pady=6,
                  borderwidth=0, activebackground=BG,
                  activeforeground=FG).pack(side="left", padx=5)

        nav = tk.Frame(self.body, bg=BG)
        nav.pack(pady=10)
        tk.Button(nav, text="< Prev", command=lambda: self._step(-1),
                  bg=SURFACE, fg=FG, font=("Helvetica", 12), padx=18, pady=8,
                  borderwidth=1, relief="solid",
                  activebackground=RED_TINT, activeforeground=FG).pack(side="left", padx=6)
        tk.Button(nav, text="Next >", command=lambda: self._step(1),
                  bg=RED, fg="#ffffff", font=("Helvetica", 12, "bold"), padx=18, pady=8,
                  borderwidth=0, activebackground="#c8121a",
                  activeforeground="#ffffff").pack(side="left", padx=6)
        tk.Button(nav, text="Export report", command=self._export,
                  bg=BLACK, fg="#ffffff", font=("Helvetica", 12), padx=16, pady=8,
                  borderwidth=0, activebackground="#333333",
                  activeforeground="#ffffff").pack(side="left", padx=(24, 6))
        tk.Button(nav, text="Start over", command=self._build_home,
                  bg=BG, fg=DIM, font=SMALL_FONT, padx=12, pady=8,
                  borderwidth=0, activebackground=BG,
                  activeforeground=RED, cursor="hand2").pack(side="left", padx=6)

        self._refilter()

    def _flagged(self):
        return [r for r in self.all_results if r["score"] >= self.threshold]

    def _on_threshold(self, value):
        self.threshold = float(value)
        self._refilter()

    def _refilter(self):
        flagged = self._flagged()
        current = self.results[self.result_idx]["path"] if getattr(self, "results", None) else None
        self.results = flagged if flagged else self.all_results
        # keep showing the same image across threshold changes when possible
        self.result_idx = next((i for i, r in enumerate(self.results)
                                if r["path"] == current), 0)
        n = len(self.all_results)
        if flagged:
            self.summary.config(
                text=f"{len(flagged)} of {n} screenshots flagged as likely cheating",
                fg=RED)
        else:
            self.summary.config(
                text=f"No screenshots flagged — showing all {n} for review", fg=FG)
        self._render_current()

    def _step(self, delta):
        if not self.results:
            return
        self.result_idx = (self.result_idx + delta) % len(self.results)
        self._render_current()

    def _set_status(self, status):
        if not self.results:
            return
        self.results[self.result_idx]["status"] = status
        self._render_current()

    def _render_current(self):
        r = self.results[self.result_idx]
        img = Image.open(r["path"])
        img.thumbnail((VIEW_W, VIEW_H))
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=self._photo)
        score = r["score"]
        mark = f"  —  [{r['status'].upper()}]" if r["status"] else ""
        self.viewer_info.config(
            text=f"{self.result_idx + 1} of {len(self.results)}  —  "
                 f"{Path(r['path']).name}  —  score {score:.2f}{mark}",
            fg=RED if score >= self.threshold else DIM)
        self.detail.config(text=self._detail_text(r, self.threshold))

    def _export(self):
        if not self.all_results:
            return
        dest = filedialog.askdirectory(title="Choose export folder")
        if not dest:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(dest) / f"comas_report_{stamp}"
        img_dir = out / "flagged"
        img_dir.mkdir(parents=True, exist_ok=True)

        flagged = self._flagged()
        for r in flagged:
            if r["status"] != "dismissed":
                shutil.copy2(r["path"], img_dir / Path(r["path"]).name)

        with open(out / "report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "score", "flagged", "status",
                        "assistant_score", "brightspace_score",
                        "brightspace_verdict", "off_task_sites", "evidence"])
            for r in self.all_results:
                w.writerow([
                    Path(r["path"]).name, f"{r['score']:.2f}",
                    "yes" if r["score"] >= self.threshold else "no",
                    r["status"],
                    "" if r["copilot_score"] is None else f"{r['copilot_score']:.2f}",
                    "" if r["bs_score"] is None else f"{r['bs_score']:.2f}",
                    r["bs_verdict"] or "", r["bs_sites"],
                    self._detail_text(r, self.threshold),
                ])

        with open(out / "summary.txt", "w", encoding="utf-8") as f:
            kept = [r for r in flagged if r["status"] != "dismissed"]
            f.write(f"CoMas screenshot triage report\n"
                    f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n"
                    f"Mode: {self.mode}\n"
                    f"Flag threshold: {self.threshold:.2f}\n\n"
                    f"Screenshots analyzed: {len(self.all_results)}\n"
                    f"Flagged: {len(flagged)}"
                    f" (of which dismissed by reviewer: {len(flagged) - len(kept)})\n"
                    f"Reviewed: {sum(1 for r in self.all_results if r['status'] == 'reviewed')}\n")

        self.summary.config(text=f"Report saved to {out}", fg=FG)

    # ----- utils ------------------------------------------------------------

    def _clear_body(self):
        for child in self.body.winfo_children():
            child.destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
