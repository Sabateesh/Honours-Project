import csv
import queue
import shutil
import threading
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

from PIL import Image, ImageDraw, ImageTk

from .brightspace import BrightspaceDetector
from .copilot import CopilotDetector, build_ml_scorer, score_from_keywords
from .resources import load_app_config
from .ocr import DEFAULT_WORKERS as OCR_WORKERS
from .ocr import OCRCache

# Carleton brand palette.
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
FLAG_THRESHOLD = 0.45

OCR_FLAG_THRESHOLD = 0.75
VIEW_W, VIEW_H = 1160, 620


def _pill(w, h, radius, fill, outline, supersample=3):
    """A rounded rectangle drawn with PIL and downsampled, which anti-aliases
    the corners. Tk cannot draw a smooth rounded rect itself."""
    s = supersample
    img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle(
        [0, 0, w * s - 1, h * s - 1], radius=radius * s,
        fill=fill, outline=outline, width=s)
    return ImageTk.PhotoImage(img.resize((w, h), Image.LANCZOS))


class Button(tk.Label):
    """A flat, rounded button.

    macOS renders tk.Button with the native Aqua theme and quietly ignores the
    background colour, so a styled tk.Button comes out as a stock grey pill no
    matter how it is configured. A Label accepts an image and centred text
    together (compound="center"), so painting the shape with PIL gives full
    control over colour, radius and hover state on every platform."""

    def __init__(self, parent, text, command=None, bg=CARD, fg=INK,
                 hover=None, border=None, font=BODY, padx=20, pady=11,
                 radius=9, width=None):
        self._command = command
        self._font = font
        self._radius = radius
        self._enabled = True

        f = tkfont.Font(family=font[0], size=font[1],
                        weight=font[2] if len(font) > 2 else "normal")
        # not _w: tkinter.Misc stores the widget's Tcl path there
        self._pill_w = width or f.measure(text) + padx * 2
        self._pill_h = f.metrics("linespace") + pady * 2

        self._paint(bg, fg, hover, border)
        super().__init__(parent, image=self._img_normal, text=text,
                         compound="center", fg=fg, font=font,
                         bg=parent.cget("bg"), bd=0, highlightthickness=0,
                         cursor="hand2")
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _paint(self, bg, fg, hover, border):
        self._bg, self._fg = bg, fg
        self._hover = hover or bg
        self._border = border or bg
        self._img_normal = _pill(self._pill_w, self._pill_h, self._radius,
                                 self._bg, self._border)
        self._img_hover = _pill(self._pill_w, self._pill_h, self._radius,
                                self._hover, self._border)

    def _on_enter(self, _=None):
        if self._enabled:
            self.config(image=self._img_hover)

    def _on_leave(self, _=None):
        if self._enabled:
            self.config(image=self._img_normal)

    def _on_click(self, _=None):
        if self._enabled and self._command:
            self._command()

    def restyle(self, bg=None, fg=None, hover=None, border=None):
        self._paint(bg or self._bg, fg or self._fg,
                    hover or self._hover, border or self._border)
        self.config(image=self._img_normal, fg=self._fg)

    def set_enabled(self, enabled, bg=None, fg=None, hover=None):
        self._enabled = enabled
        self.config(cursor="hand2" if enabled else "arrow")
        if bg or fg or hover:
            self.restyle(bg=bg, fg=fg, hover=hover)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CoMas Screenshot Checker")
        self.geometry("1280x960")
        self.minsize(1080, 800)
        self.configure(bg=BG)
        self.cfg = load_app_config()
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
            home = tk.Label(inner, image=self._logo, bg=CARD, cursor="hand2")
        else:
            home = tk.Label(inner, text="Carleton University", bg=CARD, fg=INK,
                            font=(FONT, 20, "bold"), cursor="hand2")
        home.bind("<Button-1>", lambda e: self._go_home())
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
        # size every mode button to the longest label so the row is even
        _f = tkfont.Font(family=FONT, size=13, weight="bold")
        card_w = _f.measure(max(MODES, key=len)) + 40
        for m in MODES:
            b = Button(row, m, command=lambda m=m: self._select_mode(m),
                       bg=CARD, fg=INK, hover="#f1f5f9", border=BORDER,
                       font=(FONT, 13, "bold"), pady=20, width=card_w)
            b.pack(side="left", padx=7)
            self.mode_buttons[m] = b

        self.upload_btn = Button(wrap, "Upload CoMas Screenshots",
                                 command=self._pick_files,
                                 bg="#e9edf2", fg=FAINT, hover="#e9edf2",
                                 border="#e9edf2",
                                 font=(FONT, 14, "bold"), padx=34, pady=15)
        self.upload_btn.set_enabled(False)
        self.upload_btn.pack(pady=(44, 14))

        self.status = tk.Label(wrap, text="Choose a detection mode first",
                               bg=BG, fg=FAINT, font=SMALL)
        self.status.pack()

        if not self.cfg.paths.checkpoint.exists():
            warn = tk.Frame(wrap, bg="#fff4e5", highlightbackground="#f0c890",
                            highlightthickness=1)
            warn.pack(pady=(22, 0), ipadx=16, ipady=12)
            tk.Label(warn, text="Detection model not installed",
                     bg="#fff4e5", fg="#8a5a00",
                     font=(FONT, 12, "bold")).pack()
            tk.Label(warn,
                     text=f"Expected at:  {self.cfg.paths.checkpoint}\n"
                          "Download it from the project's GitHub Releases page "
                          "and place it there.\n"
                          "Without it only text-based detection runs, which "
                          "misses most inline suggestions.",
                     bg="#fff4e5", fg="#8a5a00", font=SMALL,
                     justify="center").pack(pady=(4, 0))

    def _select_mode(self, mode):
        self.mode = mode
        for m, b in self.mode_buttons.items():
            if m == mode:
                b.restyle(bg=RED, fg="#ffffff", hover=RED_DARK, border=RED)
            else:
                b.restyle(bg=CARD, fg=INK, hover="#f1f5f9", border=BORDER)
        self.upload_btn.set_enabled(True, bg=RED, fg="#ffffff", hover=RED_DARK)
        self.upload_btn.restyle(border=RED)
        self.status.config(text=f"Mode: {mode} — ready to upload", fg=DIM)

    # ----- detection --------------------------------------------------------

    def _pick_files(self):
        paths = filedialog.askopenfilenames(
            title="Select CoMas screenshots",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if not paths:
            return
        paths = [Path(p) for p in paths]
        self.upload_btn.set_enabled(False, bg="#e9edf2", fg=FAINT,
                                    hover="#e9edf2")
        self.upload_btn.restyle(border="#e9edf2")
        for b in self.mode_buttons.values():
            b.set_enabled(False)
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
            try:
                self._ml_scorer = build_ml_scorer(self.cfg)
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

            copilot = CopilotDetector(ocr_cache=self.ocr_cache,
                                      ml_scorer=ml_scorer)
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
            self.ocr_cache.extract_many(paths, progress=_ocr_progress)
        if brightspace:
            brightspace.prefetch(paths, progress=None if copilot else _ocr_progress)

        results = []
        for i, p in enumerate(paths):
            r = {"path": p, "score": 0.0, "copilot_score": None,
                 "ml_score": None, "ocr_score": 0.0,
                 "assistant_kind": None, "left_vscode": 0.0,
                 "bs_score": None, "bs_verdict": None, "bs_sites": "",
                 "status": "", "box": None}
            if copilot:
                ev = copilot.detect(p, ml_score=ml_scores.get(p))
                r["copilot_score"] = ev.score
                r["ml_score"] = ev.ml_score
                r["ocr_score"] = score_from_keywords(ev.strong_matches,
                                                     ev.weak_matches)[0]
                r["assistant_kind"] = self._assistant_kind(ev)
                r["score"] = max(r["score"], ev.score)

                if ev.strong_matches:
                    r["score"] = 1.0
                r["left_vscode"] = self._left_vscode(
                    ev.is_ide, ev.ocr_ran, combined=use_brightspace)
                r["score"] = max(r["score"], r["left_vscode"])
                if hasattr(copilot.ml_scorer, "box_for"):
                    r["box"] = copilot.ml_scorer.box_for(p)
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
    def _assistant_kind(ev):
        """Which of the two ways an assistant shows up did we see?

        The model is a single binary classifier and cannot say. The keywords
        can: they only appear in an assistant's own interface, so a match means
        a panel is open. A detection with no keywords anywhere on screen is
        the inline case - ghost text carries no identifying text at all.

        Which assistant it was is not named. The reviewer's decision is the
        same either way, and the vendor guess rests on whichever keyword OCR
        happened to read - a claim in an integrity case should not."""
        if ev.strong_matches or ev.weak_matches:
            return "chat panel"
        if not ev.ocr_ran:
            return None            # cannot tell without the text
        return "inline ghost text"

    @staticmethod
    def _left_vscode(is_ide, ocr_ran, combined):
        """During a VS Code exam every capture should show the editor, so one
        that does not is a window-leave. Suppressed in combined mode, where
        the student is also expected to be in Brightspace, and when OCR did
        not run - absence of text is not evidence of absence of an editor."""
        if combined or not ocr_ran or is_ide:
            return 0.0
        return 0.95

    @staticmethod
    def _assistant_flagged(r, threshold):
        """Each detector is judged on its own scale. The model threshold is
        calibrated near 1.0 for the tiled checkpoint; keyword scores top out
        at 0.95, so comparing them to the same number would mean OCR could
        never raise a flag."""
        if (r.get("ml_score") or 0.0) >= threshold:
            return True
        if r.get("ocr_score", 0.0) >= OCR_FLAG_THRESHOLD:
            return True
        # no model available: fall back to the combined score
        if r.get("ml_score") is None and (r.get("copilot_score") or 0.0) >= threshold:
            return True
        return False

    @staticmethod
    def _detail_text(r, threshold):
        bits = []
        if r.get("left_vscode", 0.0) >= threshold:
            bits.append("LEFT VS CODE: screenshot is not the editor")
        if r["copilot_score"] is not None and App._assistant_flagged(r, threshold):
            kind = r.get("assistant_kind")
            bits.append(f"CHEATING DETECTED: {kind}" if kind
                        else "CHEATING DETECTED: AI assistant")
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
                 font=(FONT, 9, "bold")).pack(side="left", padx=(0, 8))

       
        self.thr_slider = tk.Scale(
            thr, from_=0.50, to=1.0, resolution=0.0001,
            orient="horizontal", length=300, bg=BG, fg=INK,
            troughcolor="#e2e8f0", highlightthickness=0, bd=0,
            font=TINY, showvalue=False, command=self._on_threshold)
        self.thr_slider.set(FLAG_THRESHOLD)
        self.thr_slider.pack(side="left")

        self.thr_entry = tk.Entry(thr, width=7, font=(FONT, 11), justify="center",
                                  bd=0, highlightthickness=1,
                                  highlightbackground=BORDER,
                                  highlightcolor=RED, bg=CARD, fg=INK)
        self.thr_entry.insert(0, f"{FLAG_THRESHOLD:.4f}")
        self.thr_entry.pack(side="left", padx=(10, 0), ipady=4)
        self.thr_entry.bind("<Return>", self._on_threshold_typed)
        self.thr_entry.bind("<FocusOut>", self._on_threshold_typed)

        # measured operating points, see FLAG_THRESHOLD
        for label, value in (("strict", 0.90), ("default", FLAG_THRESHOLD),
                             ("loose", 0.25)):
            Button(thr, label, command=lambda v=value: self._set_threshold(v),
                   bg=CARD, fg=DIM, hover="#f1f5f9", border=BORDER,
                   font=(FONT, 10), padx=10, pady=5).pack(side="left", padx=3)

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
        Button(nav, "←  Prev", command=lambda: self._step(-1),
               bg=CARD, fg=INK, hover="#f1f5f9", border=BORDER,
               padx=22).pack(side="left", padx=(0, 8))
        Button(nav, "Next  →", command=lambda: self._step(1),
               bg=RED, fg="#ffffff", hover=RED_DARK, border=RED,
               font=(FONT, 13, "bold"), padx=22).pack(side="left")

        triage = tk.Frame(controls, bg=BG)
        triage.pack(side="left", padx=30)
        Button(triage, "✓  Reviewed", command=lambda: self._set_status("reviewed"),
               bg=CARD, fg=INK, hover=GREEN_TINT, border=BORDER,
               font=SMALL, padx=16).pack(side="left", padx=4)
        Button(triage, "✕  Dismiss", command=lambda: self._set_status("dismissed"),
               bg=CARD, fg=INK, hover=RED_TINT, border=BORDER,
               font=SMALL, padx=16).pack(side="left", padx=4)
        Button(triage, "Clear", command=lambda: self._set_status(""),
               bg=BG, fg=FAINT, hover="#eef2f6", border=BG,
               font=SMALL, padx=12).pack(side="left", padx=4)

        actions = tk.Frame(controls, bg=BG)
        actions.pack(side="right")
        Button(actions, "Export report", command=self._export,
               bg=BLACK, fg="#ffffff", hover="#2d2d2d", border=BLACK,
               padx=20).pack(side="left", padx=(0, 10))
        Button(actions, "Start over", command=self._go_home,
               bg=BG, fg=FAINT, hover="#eef2f6", border=BG,
               font=SMALL, padx=14).pack(side="left")

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
        return [r for r in self.all_results
                if r["score"] >= self.threshold
                or self._assistant_flagged(r, self.threshold)]

    def _on_threshold(self, value):
        self.threshold = float(value)
        if hasattr(self, "thr_entry"):
            self.thr_entry.delete(0, tk.END)
            self.thr_entry.insert(0, f"{self.threshold:.4f}")
        self._refilter()

    def _on_threshold_typed(self, _event=None):
        """Typed values are the only way to reach thresholds the slider's step
        size cannot land on exactly."""
        try:
            v = float(self.thr_entry.get())
        except ValueError:
            self.thr_entry.delete(0, tk.END)
            self.thr_entry.insert(0, f"{self.threshold:.4f}")
            return
        self._set_threshold(max(0.0, min(1.0, v)))

    def _set_threshold(self, value):
        self.thr_slider.set(value)

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
        img = Image.open(r["path"]).convert("RGB")
        full_w, full_h = img.size
        sw = max(self._stage.winfo_width() - 24, 200)
        sh = max(self._stage.winfo_height() - 24, 200)
        img.thumbnail((min(sw, VIEW_W * 2), min(sh, VIEW_H * 2)), Image.LANCZOS)

        if r.get("box") and r["score"] >= self.threshold:
            scale = img.size[0] / full_w
            x0, y0, x1, y1 = (int(v * scale) for v in r["box"])
            d = ImageDraw.Draw(img)
            for w in range(3):
                d.rectangle([x0 - w, y0 - w, x1 + w, y1 + w], outline=RED)

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
                        "assistant_score", "assistant_kind", "left_vscode",
                        "brightspace_score",
                        "brightspace_verdict", "off_task_sites", "evidence"])
            for r in self.all_results:
                w.writerow([
                    Path(r["path"]).name, f"{r['score']:.2f}",
                    "yes" if r["score"] >= self.threshold else "no",
                    r["status"],
                    "" if r["copilot_score"] is None else f"{r['copilot_score']:.2f}",
                    r.get("assistant_kind") or "",
                    "yes" if r.get("left_vscode", 0.0) >= self.threshold else "",
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


    def _clear_body(self):
        for child in self.body.winfo_children():
            child.destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
