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

# ---- palette ---------------------------------------------------------------
# Carleton brand red on a clean, flat, near-white surface. The image stage is
# near-black so screenshots read like slides on a projector.
BG = "#fafafa"
CARD = "#ffffff"
BORDER = "#e5e7eb"
INK = "#0f172a"
DIM = "#64748b"
FAINT = "#94a3b8"
RED = "#e91c24"
RED_DARK = "#c8121a"
RED_TINT = "#fdecec"
GREEN_TINT = "#dcfce7"
BLACK = "#111111"
STAGE = "#111111"

FONT = "Helvetica Neue"
H1 = (FONT, 24, "bold")
H2 = (FONT, 15, "bold")
BODY = (FONT, 13)
SMALL = (FONT, 11)
TINY = (FONT, 10)

MODES = ["VSCODE Cheating", "Brightspace Cheating", "VSCODE + Brightspace Cheating"]
# Recall-first default, set aggressively low on purpose: reviewers dismiss a
# benign flag in one click, but a missed cheat is invisible. Ghost text is the
# model's weakest signal and much of it scores in the 0.2-0.6 band; false
# positives are the accepted cost and the slider raises the bar per run.
FLAG_THRESHOLD = 0.20
VIEW_W, VIEW_H = 1160, 620


def flat(btn, bg, fg, hover_bg, font=BODY, padx=18, pady=10):
    btn.config(bg=bg, fg=fg, font=font, padx=padx, pady=pady,
               borderwidth=0, relief="flat", highlightthickness=0,
               activebackground=hover_bg, activeforeground=fg,
               cursor="hand2")
    btn.bind("<Enter>", lambda e: btn["state"] == "normal" and btn.config(bg=hover_bg))
    btn.bind("<Leave>", lambda e: btn["state"] == "normal" and btn.config(bg=bg))
    return btn


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CoMas Screenshot Checker")
        self.geometry("1280x960")
        self.minsize(1080, 800)
        self.configure(bg=BG)

        cfg_path = Path("config.yaml")
        self.cfg = load_config(cfg_path if cfg_path.exists() else None)
        self.ocr_cache = OCRCache(self.cfg.paths.ocr_cache)

        self.mode = None
        self.results = []
        self.all_results = []
        self.result_idx = 0
        self.threshold = FLAG_THRESHOLD
        self._queue = queue.Queue()
        self._photo = None

        self._ml_scorer = None
        self._ml_scorer_built = False
        self._ml_status = "not loaded"
        # bumped whenever the user leaves a screen, so a scan that finishes
        # after they navigate away is ignored instead of hijacking the view
        self._run_id = 0

        self._build_header()
        self.body = tk.Frame(self, bg=BG)
        self.body.pack(fill="both", expand=True)
        self._build_home()

    # ----- header ----------------------------------------------------------

    def _build_header(self):
        bar = tk.Frame(self, bg=CARD)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=CARD)
        inner.pack(fill="x", padx=28, pady=14)

        logo_path = Path(__file__).parent / "assets" / "carleton_logo.png"
        if logo_path.exists():
            img = Image.open(logo_path)
            img.thumbnail((260, 58), Image.LANCZOS)
            self._logo = ImageTk.PhotoImage(img)
            home = tk.Button(inner, image=self._logo, command=self._go_home,
                             bg=CARD, activebackground=CARD, borderwidth=0,
                             relief="flat", highlightthickness=0,
                             cursor="hand2", takefocus=0)
        else:
            home = tk.Button(inner, text="Carleton University",
                             command=self._go_home, bg=CARD, fg=INK,
                             activebackground=CARD, activeforeground=RED,
                             font=(FONT, 20, "bold"), borderwidth=0,
                             relief="flat", highlightthickness=0,
                             cursor="hand2", takefocus=0)
        home.pack(side="left")
        self._home_btn = home

        tip = tk.Label(inner, text="", bg=CARD, fg=FAINT, font=TINY)
        tip.pack(side="left", padx=(12, 0))
        home.bind("<Enter>", lambda e: tip.config(text="↩  Back to start"))
        home.bind("<Leave>", lambda e: tip.config(text=""))

        right = tk.Frame(inner, bg=CARD)
        right.pack(side="right")
        tk.Label(right, text="CoMas Screenshot Checker", bg=CARD, fg=INK,
                 font=H2, anchor="e").pack(anchor="e")

        tk.Frame(self, bg=RED, height=3).pack(fill="x")

    # ----- home screen ------------------------------------------------------

    def _go_home(self):
        """Logo click. Abandons any in-flight scan's results (the worker thread
        is a daemon and its OCR work stays cached, so nothing is wasted)."""
        self._run_id += 1
        self._build_home()

    def _build_home(self):
        self._clear_body()
        self.unbind_all_keys()

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(expand=True)

        tk.Label(wrap, text="What should be detected?", bg=BG, fg=INK,
                 font=H1).pack(pady=(0, 6))
        tk.Label(wrap, text="Choose a mode, then upload the screenshots "
                            "CoMas captured during the exam.",
                 bg=BG, fg=DIM, font=BODY).pack(pady=(0, 30))

        row = tk.Frame(wrap, bg=BG)
        row.pack()
        self.mode_buttons = {}
        for m in MODES:
            card = tk.Frame(row, bg=CARD, highlightbackground=BORDER,
                            highlightthickness=1)
            card.pack(side="left", padx=9)
            b = tk.Button(card, text=m, command=lambda m=m: self._select_mode(m))
            flat(b, CARD, INK, "#f1f5f9", font=(FONT, 13, "bold"),
                 padx=26, pady=18)
            b.pack()
            self.mode_buttons[m] = (card, b)

        self.upload_btn = tk.Button(wrap, text="Upload CoMas Screenshots",
                                    command=self._pick_files, state=tk.DISABLED)
        flat(self.upload_btn, "#e2e8f0", FAINT, "#e2e8f0",
             font=(FONT, 14, "bold"), padx=34, pady=14)
        self.upload_btn.config(cursor="arrow")
        self.upload_btn.pack(pady=(44, 14))

        self.status = tk.Label(wrap, text="Choose a detection mode first",
                               bg=BG, fg=FAINT, font=SMALL)
        self.status.pack()

    def _select_mode(self, mode):
        self.mode = mode
        for m, (card, b) in self.mode_buttons.items():
            active = (m == mode)
            card.config(highlightbackground=RED if active else BORDER,
                        highlightthickness=2 if active else 1)
            b.config(bg=RED if active else CARD,
                     fg="#ffffff" if active else INK,
                     activebackground=RED_DARK if active else "#f1f5f9",
                     activeforeground="#ffffff" if active else INK)
            b.unbind("<Enter>")
            b.unbind("<Leave>")
            if active:
                b.bind("<Enter>", lambda e, b=b: b.config(bg=RED_DARK))
                b.bind("<Leave>", lambda e, b=b: b.config(bg=RED))
            else:
                b.bind("<Enter>", lambda e, b=b: b.config(bg="#f1f5f9"))
                b.bind("<Leave>", lambda e, b=b: b.config(bg=CARD))
        self.upload_btn.config(state=tk.NORMAL)
        flat(self.upload_btn, RED, "#ffffff", RED_DARK,
             font=(FONT, 14, "bold"), padx=34, pady=14)
        self.status.config(text=f"Mode: {mode} — ready to upload", fg=DIM)

    # ----- detection --------------------------------------------------------

    def _pick_files(self):
        paths = filedialog.askopenfilenames(
            title="Select CoMas screenshots",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if not paths:
            return
        paths = [Path(p) for p in paths]
        self.upload_btn.config(state=tk.DISABLED, bg="#e2e8f0", fg=FAINT,
                               cursor="arrow")
        for _, b in self.mode_buttons.values():
            b.config(state=tk.DISABLED)
        signal_note = "OCR + ghost-text model" if "VSCODE" in self.mode else "OCR"
        self.status.config(text=f"Analyzing {len(paths)} screenshot(s)... "
                                f"first run is slow ({signal_note})", fg=DIM)
        rid = self._run_id
        threading.Thread(target=self._run_detection, args=(paths,),
                         daemon=True).start()
        self.after(100, lambda: self._poll_queue(rid))

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

        def _ocr_progress(done, total):
            self._queue.put(("ocr", done, total))

        ml_scores = {}
        if copilot:
            if copilot.ml_scorer is not None:
                try:
                    raw = copilot.ml_scorer.score_batch(
                        paths, progress=lambda d, t: self._queue.put(("model", d, t)))
                    ml_scores = {p: raw.get(str(p)) for p in paths}
                except Exception as e:
                    self._queue.put(("model_status", f"scoring failed ({e})"))
            needs_ocr = [p for p in paths if not ml_is_confident(ml_scores.get(p))]
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
                r["bs_score"] = self._brightspace_contribution(
                    ev.verdict, ev.score, combined=use_copilot,
                    is_ide=ev.is_ide)
                r["bs_verdict"] = ev.verdict
                r["bs_sites"] = ",".join(ev.off_task_sites)
                r["score"] = max(r["score"], r["bs_score"])
            results.append(r)
            self._queue.put(("progress", i + 1, len(paths)))

        self.ocr_cache.flush()
        results.sort(key=lambda r: r["score"], reverse=True)
        self._queue.put(("done", results))

    @staticmethod
    def _brightspace_contribution(verdict, score, combined, is_ide=False):
        """Not-on-the-quiz means cheating - except in combined mode, where a
        capture that is clearly the IDE is the student doing their exam. The
        IDE gets no free pass in Brightspace-only mode."""
        if combined and is_ide:
            return 0.0
        return score

    @staticmethod
    def _detail_text(r, threshold):
        bits = []
        if r["copilot_score"] is not None and r["copilot_score"] >= threshold:
            bits.append("CHEATING DETECTED:")
        if r["bs_score"] is not None and r["bs_score"] >= threshold:
            sites = r["bs_sites"] or "unrecognized page"
            bits.append(f"Brightspace: {r['bs_verdict']} ({sites})")
        return "; ".join(bits) or "no signal above threshold"

    def _poll_queue(self, rid=None):
        if rid is not None and rid != self._run_id:
            return  # user navigated away; drop this scan's updates
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
        self.after(100, lambda: self._poll_queue(rid))

    # ----- results viewer ---------------------------------------------------

    def _show_results(self, results):
        self.all_results = results
        self.threshold = FLAG_THRESHOLD
        self.result_idx = 0
        self._clear_body()

        # toolbar: summary on the left, threshold on the right
        toolbar = tk.Frame(self.body, bg=BG)
        toolbar.pack(fill="x", padx=32, pady=(16, 8))

        self.summary = tk.Label(toolbar, text="", bg=BG, fg=INK, font=H2,
                                anchor="w")
        self.summary.pack(side="left")

        thr = tk.Frame(toolbar, bg=BG)
        thr.pack(side="right")
        tk.Label(thr, text="FLAG THRESHOLD", bg=BG, fg=FAINT,
                 font=(FONT, 9, "bold")).pack(side="left", padx=(0, 10))
        self.thr_slider = tk.Scale(
            thr, from_=0.05, to=0.95, resolution=0.05,
            orient="horizontal", length=200, bg=BG, fg=INK,
            troughcolor="#e2e8f0", highlightthickness=0, bd=0,
            font=TINY, command=self._on_threshold)
        self.thr_slider.set(FLAG_THRESHOLD)
        self.thr_slider.pack(side="left")

        # image stage
        stage_wrap = tk.Frame(self.body, bg=STAGE)
        stage_wrap.pack(fill="both", expand=True, padx=32)
        self.canvas = tk.Label(stage_wrap, bg=STAGE)
        self.canvas.place(relx=0.5, rely=0.5, anchor="center")
        self._stage = stage_wrap

        # info strip under the image
        info = tk.Frame(self.body, bg=BG)
        info.pack(fill="x", padx=32, pady=(10, 0))
        self.viewer_info = tk.Label(info, text="", bg=BG, fg=DIM, font=SMALL,
                                    anchor="w")
        self.viewer_info.pack(side="left")
        self.mark_label = tk.Label(info, text="", bg=BG, font=(FONT, 10, "bold"))
        self.mark_label.pack(side="right")

        self.detail = tk.Label(self.body, text="", bg=BG, fg=RED,
                               font=(FONT, 13, "bold"), wraplength=1100,
                               anchor="w", justify="left")
        self.detail.pack(fill="x", padx=32, pady=(2, 0))

        # controls
        controls = tk.Frame(self.body, bg=BG)
        controls.pack(fill="x", padx=32, pady=14)

        nav = tk.Frame(controls, bg=BG)
        nav.pack(side="left")
        prev_b = tk.Button(nav, text="←  Prev", command=lambda: self._step(-1))
        flat(prev_b, CARD, INK, "#f1f5f9", padx=20, pady=9)
        prev_b.config(highlightbackground=BORDER, highlightthickness=1)
        prev_b.pack(side="left", padx=(0, 6))
        next_b = tk.Button(nav, text="Next  →", command=lambda: self._step(1))
        flat(next_b, RED, "#ffffff", RED_DARK, font=(FONT, 13, "bold"),
             padx=20, pady=9)
        next_b.pack(side="left")

        triage = tk.Frame(controls, bg=BG)
        triage.pack(side="left", padx=28)
        rev_b = tk.Button(triage, text="Mark reviewed",
                          command=lambda: self._set_status("reviewed"))
        flat(rev_b, CARD, INK, GREEN_TINT, font=SMALL, padx=14, pady=9)
        rev_b.config(highlightbackground=BORDER, highlightthickness=1)
        rev_b.pack(side="left", padx=3)
        dis_b = tk.Button(triage, text="Dismiss",
                          command=lambda: self._set_status("dismissed"))
        flat(dis_b, CARD, INK, RED_TINT, font=SMALL, padx=14, pady=9)
        dis_b.config(highlightbackground=BORDER, highlightthickness=1)
        dis_b.pack(side="left", padx=3)
        clr_b = tk.Button(triage, text="Clear",
                          command=lambda: self._set_status(""))
        flat(clr_b, BG, FAINT, "#f1f5f9", font=SMALL, padx=10, pady=9)
        clr_b.pack(side="left", padx=3)

        actions = tk.Frame(controls, bg=BG)
        actions.pack(side="right")
        exp_b = tk.Button(actions, text="Export report", command=self._export)
        flat(exp_b, BLACK, "#ffffff", "#2d2d2d", padx=18, pady=9)
        exp_b.pack(side="left", padx=(0, 10))
        over_b = tk.Button(actions, text="Start over", command=self._go_home)
        flat(over_b, BG, FAINT, "#f1f5f9", font=SMALL, padx=12, pady=9)
        over_b.pack(side="left")

        # keyboard: arrows to navigate, r/d/c to triage, e to export
        self.bind("<Left>", lambda e: self._step(-1))
        self.bind("<Right>", lambda e: self._step(1))
        self.bind("r", lambda e: self._set_status("reviewed"))
        self.bind("d", lambda e: self._set_status("dismissed"))
        self.bind("c", lambda e: self._set_status(""))
        self.bind("e", lambda e: self._export())
        self._stage.bind("<Configure>", lambda e: self._render_current())

        self._refilter()

    def unbind_all_keys(self):
        for key in ("<Left>", "<Right>", "r", "d", "c", "e"):
            self.unbind(key)

    def _flagged(self):
        return [r for r in self.all_results if r["score"] >= self.threshold]

    def _on_threshold(self, value):
        self.threshold = float(value)
        self._refilter()

    def _refilter(self):
        flagged = self._flagged()
        current = self.results[self.result_idx]["path"] if getattr(self, "results", None) else None
        self.results = flagged if flagged else self.all_results
        self.result_idx = next((i for i, r in enumerate(self.results)
                                if r["path"] == current), 0)
        n = len(self.all_results)
        if flagged:
            self.summary.config(
                text=f"{len(flagged)} of {n} flagged as likely cheating", fg=RED)
        else:
            self.summary.config(
                text=f"No screenshots flagged — showing all {n}", fg=INK)
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
        if not self.results:
            return
        r = self.results[self.result_idx]
        img = Image.open(r["path"])
        # fit to the live stage size; LANCZOS keeps small UI text legible
        sw = max(self._stage.winfo_width() - 24, 200)
        sh = max(self._stage.winfo_height() - 24, 200)
        img.thumbnail((min(sw, VIEW_W * 2), min(sh, VIEW_H * 2)), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=self._photo)

        score = r["score"]
        self.viewer_info.config(
            text=f"{self.result_idx + 1} / {len(self.results)}    "
                 f"{Path(r['path']).name}    score {score:.2f}",
            fg=RED if score >= self.threshold else DIM)
        if r["status"] == "reviewed":
            self.mark_label.config(text="✓ REVIEWED", fg="#16a34a")
        elif r["status"] == "dismissed":
            self.mark_label.config(text="✕ DISMISSED", fg=FAINT)
        else:
            self.mark_label.config(text="")
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

        self.summary.config(text=f"Report saved to {out}", fg=INK)

    # ----- utils ------------------------------------------------------------

    def _clear_body(self):
        for child in self.body.winfo_children():
            child.destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
