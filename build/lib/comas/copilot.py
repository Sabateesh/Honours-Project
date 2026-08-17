# OCR based detection of GitHub Copilot and Cursor in screenshots. Uses the test shots and looks for keywords in the OCR text, as well as template matching and an optional ML model. Results are written to a CSV file for analysis.

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .config import load_config
from .ocr import OCRCache

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

STRONG_KEYWORDS: dict[str, str] = {
    # GitHub Copilot
    "github copilot": "copilot",
    "copilot chat": "copilot",
    "copilot suggestions": "copilot",
    "tab to accept": "copilot",
    "accept suggestion": "copilot",
    "next suggestion": "copilot",
    "open copilot": "copilot",
    "ask copilot": "copilot",
    # Cursor
    "plan agent": "cursor",
    "explore and understand your code": "cursor",
    "no matching sessions": "cursor",
    "ask cursor": "cursor",
    "cursor tab": "cursor",
    "before implementing changes": "cursor",
    
}

WEAK_KEYWORDS: dict[str, str] = {
    "ghcp": "copilot",
    "github.com/copilot": "copilot",
    "ghost text": "copilot",
}

# VS Code chrome that OCR reads reliably. Two matches are required so a quiz
# question mentioning a .py file does not register as an editor.
IDE_KEYWORDS = [
    "explorer", "debug console", "problems", "output", "terminal",
    "spaces: 4", "utf-8", ".py", "outline", "timeline",
]


def looks_like_ide(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return sum(1 for k in IDE_KEYWORDS if k in low) >= 2


@dataclass
class CopilotEvidence:
    path: str
    score: float
    method: str
    detected_tool: str = "unknown"
    strong_matches: list[str] = field(default_factory=list)
    weak_matches: list[str] = field(default_factory=list)
    template_name: Optional[str] = None
    template_confidence: Optional[float] = None
    template_box: Optional[tuple[int, int, int, int]] = None
    ml_score: Optional[float] = None
    ocr_ran: bool = False
    is_ide: bool = False
    ml_gated: bool = False
    ocr_snippet: Optional[str] = None
    


def find_keywords(text: str) -> tuple[list[str], list[str]]:
    if not text:
        return [], []
    low = text.lower()
    strong = [k for k in STRONG_KEYWORDS if k in low]
    weak = [k for k in WEAK_KEYWORDS if k in low]
    return strong, weak
    

def infer_tool(strong: list[str], weak: list[str]) -> str:
    tools = set()
    for k in strong:
        tools.add(STRONG_KEYWORDS[k])
    for k in weak:
        if k in WEAK_KEYWORDS:
            tools.add(WEAK_KEYWORDS[k])
    if not tools:
        return "unknown"
    if len(tools) == 1:
        return tools.pop()
    return "both"
    


def score_from_keywords(strong: list[str], weak: list[str]) -> tuple[float, str]:
    if strong:
        return 0.95, "ocr_strong"
    if len(weak) >= 2:
        return 0.75, "ocr_weak"
    if weak:
        return 0.45, "ocr_weak"
    return 0.0, "none"
    


def _match_one_template(img_gray, tpl_gray, scales):
    import cv2

    best = None
    h_t, w_t = tpl_gray.shape
    img_h, img_w = img_gray.shape
    
    for scale in scales:
        new_w = max(5, int(w_t * scale))
        new_h = max(5, int(h_t * scale))
        if new_w >= img_w or new_h >= img_h:
            continue
        scaled = cv2.resize(tpl_gray, (new_w, new_h),
                            interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(img_gray, scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if best is None or max_val > best["confidence"]:
            best = {
                "confidence": float(max_val),
                "scale": scale,
                "box": (int(max_loc[0]), int(max_loc[1]),
                        int(max_loc[0] + new_w), int(max_loc[1] + new_h)),
            }
    return best
    


def detect_via_template(image_path, templates_dir,
                        scales=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                        threshold=0.78):
    try:
        import cv2
    except ImportError:
        log.warning("opencv-python not installed; skipping template matching")
        return {"score": 0.0, "best_match": None, "templates_searched": 0}

    if not templates_dir.exists():
        return {"score": 0.0, "best_match": None, "templates_searched": 0}

    img = cv2.imread(str(image_path))
    if img is None:
        log.warning("Could not read image: %s", image_path)
        return {"score": 0.0, "best_match": None, "templates_searched": 0}
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    templates = sorted(templates_dir.glob("*.png"))
    if not templates:
        return {"score": 0.0, "best_match": None, "templates_searched": 0}

    best = None
    for tpl_path in templates:
        tpl = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        match = _match_one_template(img_gray, tpl, scales)
        if match and (best is None or match["confidence"] > best["confidence"]):
            best = {**match, "name": tpl_path.name}

    if best is None:
        return {"score": 0.0, "best_match": None, "templates_searched": len(templates)}

    score = best["confidence"] if best["confidence"] >= threshold else 0.0
    
    return {"score": score, "best_match": best, "templates_searched": len(templates)}


# Above this the model is confident enough on its own: the screenshot is going
# to be flagged either way, so the expensive OCR pass adds nothing. Deliberately
# one sided - a low model score does NOT allow skipping OCR, because the model
# only learned inline ghost text and would miss an open chat panel.
ML_CONFIDENT = 0.90


def ml_is_confident(ml_score: Optional[float]) -> bool:
    return ml_score is not None and ml_score >= ML_CONFIDENT


class CopilotDetector:
    def __init__(self, ocr_cache=None, templates_dir=None, ml_scorer=None,
                 template_threshold=0.78,
                 template_scales=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                 skip_ocr_when_confident=True, gate_ml_on_ide=True):
        self.ocr_cache = ocr_cache
        self.templates_dir = templates_dir
        self.ml_scorer = ml_scorer
        self.template_threshold = template_threshold
        self.template_scales = template_scales
        self.skip_ocr_when_confident = skip_ocr_when_confident
        self.gate_ml_on_ide = gate_ml_on_ide

    def detect(self, image_path: Path,
               ml_score: Optional[float] = None) -> CopilotEvidence:
        ev = CopilotEvidence(path=str(image_path), score=0.0, method="none")

        # Model first, so a confident result can short circuit the OCR pass.
        if ml_score is None and self.ml_scorer is not None:
            try:
                ml_score = float(self.ml_scorer(image_path))
            except Exception as e:
                log.warning("ML scorer failed on %s: %s", image_path, e)
        if ml_score is not None:
            ev.ml_score = ml_score
            ev.score, ev.method = ml_score, "ml"

        skip_ocr = self.skip_ocr_when_confident and ml_is_confident(ml_score)
        if self.ocr_cache is not None and not skip_ocr:
            ev.ocr_ran = True
            text = self.ocr_cache.get_or_extract(image_path)
            ev.is_ide = looks_like_ide(text)
            strong, weak = find_keywords(text)
            ocr_score, ocr_method = score_from_keywords(strong, weak)
            ev.strong_matches = strong
            ev.weak_matches = weak
            ev.ocr_snippet = text[:300] if text else None
            ev.detected_tool = infer_tool(strong, weak)
            if ocr_score > ev.score:
                ev.score, ev.method = ocr_score, ocr_method

        if self.templates_dir is not None:
            tpl = detect_via_template(image_path, self.templates_dir,
                                       self.template_scales,
                                       self.template_threshold)
            if tpl["best_match"]:
                ev.template_name = tpl["best_match"]["name"]
                ev.template_confidence = tpl["best_match"]["confidence"]
                ev.template_box = tpl["best_match"]["box"]
            if tpl["score"] > ev.score:
                ev.score, ev.method = tpl["score"], "template"

        # An AI coding assistant lives inside a code editor. If the capture is
        # not an editor, the model is being asked about something it was never
        # trained on and its answer is not meaningful - so it is discarded and
        # the screenshot is judged on text alone. Keyword matches survive:
        # reading "GitHub Copilot" on screen is direct evidence wherever it
        # appears. This is what makes browser training negatives unnecessary.
        if self.gate_ml_on_ide and ev.ocr_ran and not ev.is_ide:
            ocr_score, ocr_method = score_from_keywords(ev.strong_matches,
                                                        ev.weak_matches)
            ev.score, ev.method = ocr_score, ocr_method
            ev.ml_gated = True

        return ev


def scan_folder(folder: Path, detector: CopilotDetector) -> list[CopilotEvidence]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise RuntimeError(f"No images in {folder}")
    results = [detector.detect(p) for p in paths]
    if detector.ocr_cache is not None:
        detector.ocr_cache.flush()
    results.sort(key=lambda e: e.score, reverse=True)
    return results


def write_csv(results: list[CopilotEvidence], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    for r in rows:
        r["strong_matches"] = "|".join(r["strong_matches"])
        r["weak_matches"] = "|".join(r["weak_matches"])
        if r.get("template_box"):
            r["template_box"] = ",".join(str(x) for x in r["template_box"])
        if r.get("ocr_snippet"):
            r["ocr_snippet"] = r["ocr_snippet"].replace("\n", " ")
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        


class MLScorer:
    """Wraps the trained CNN. Callable per image for compatibility, but
    score_batch is what the GUI uses - one forward pass per BATCH images
    instead of per screenshot."""

    BATCH = 16

    def __init__(self, model, eval_tfms, device, torch_mod,
                 text_encoder=None, ocr_cache=None):
        self._model = model
        self._tfms = eval_tfms
        self._device = device
        self._torch = torch_mod
        self._text_encoder = text_encoder
        self._ocr_cache = ocr_cache

    def __call__(self, path: Path) -> float:
        return self.score_batch([path])[str(path)]

    def score_images(self, images) -> list[float]:
        """Score already-loaded PIL images. Used by tiled inference, where the
        crops exist in memory and never touch disk."""
        torch = self._torch
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(images), self.BATCH):
                chunk = images[i:i + self.BATCH]
                x = torch.stack([self._tfms(im) for im in chunk]).to(self._device)
                if self._text_encoder is not None:
                    t = self._text_encoder.encode([""] * len(chunk)).to(self._device)
                else:
                    t = torch.zeros(len(chunk), 0, device=self._device)
                probs = torch.sigmoid(self._model(x, t)).cpu()
                out.extend(float(v) for v in probs)
        return out

    def score_batch(self, paths, progress=None) -> dict[str, float]:
        from PIL import Image
        torch = self._torch
        out: dict[str, float] = {}
        done = 0
        with torch.no_grad():
            for i in range(0, len(paths), self.BATCH):
                chunk = [Path(p) for p in paths[i:i + self.BATCH]]
                xs, kept = [], []
                for p in chunk:
                    try:
                        xs.append(self._tfms(Image.open(p).convert("RGB")))
                        kept.append(p)
                    except Exception as e:
                        log.warning("Could not read %s: %s", p, e)
                        out[str(p)] = 0.0
                if not xs:
                    done += len(chunk)
                    if progress:
                        progress(done, len(paths))
                    continue
                x = torch.stack(xs).to(self._device)
                if self._text_encoder is not None:
                    texts = [self._ocr_cache.get_or_extract(p) for p in kept]
                    t = self._text_encoder.encode(texts).to(self._device)
                else:
                    t = torch.zeros(len(kept), 0, device=self._device)
                logits = self._model(x, t)
                probs = torch.sigmoid(logits).cpu()
                for p, prob in zip(kept, probs):
                    out[str(p)] = float(prob)
                done += len(chunk)
                if progress:
                    progress(done, len(paths))
        return out


def build_ml_scorer(cfg_path: Optional[Path]) -> Optional[MLScorer]:
    cfg = load_config(cfg_path)
    if not cfg.paths.checkpoint.exists():
        log.info("No checkpoint at %s; ML signal disabled.", cfg.paths.checkpoint)
        return None

    try:
        import json
        import torch
        from .data import build_transforms
        from .model import TextEncoder, build_model
    except ImportError as e:
        log.warning("ML deps not available (%s); ML signal disabled.", e)
        return None

    sidecar = cfg.paths.checkpoint.with_suffix(".meta.json")
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = build_model(cfg.model, text_emb_dim=meta.get("text_emb_dim", 0),
                        pretrained=False).to(device)
    model.load_state_dict(torch.load(cfg.paths.checkpoint, map_location=device))
    model.eval()
    _, eval_tfms = build_transforms(cfg.data.img_size)

    text_encoder = TextEncoder(cfg.model.text_encoder, device=str(device)) \
        if cfg.model.use_ocr else None
    ocr_cache = OCRCache(cfg.paths.ocr_cache) if cfg.model.use_ocr else None

    scorer = MLScorer(model, eval_tfms, device, torch,
                      text_encoder=text_encoder, ocr_cache=ocr_cache)

    # A tile-trained checkpoint is applied tile-by-tile at native resolution
    # instead of resizing the whole screenshot down to the input size.
    tile_cfg = getattr(cfg, "tile", None)
    if tile_cfg is not None and tile_cfg.enabled:
        from .tiling import TiledScorer
        log.info("Tiled inference enabled (tile_frac=%.2f, overlap=%.2f).",
                 tile_cfg.tile_frac, tile_cfg.overlap)
        return TiledScorer(scorer, tile_frac=tile_cfg.tile_frac,
                           overlap=tile_cfg.overlap, min_px=tile_cfg.min_px)
    return scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, required=True)
    ap.add_argument("--templates", type=Path, default=Path("assets/copilot_templates"))
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--out", type=Path, default=Path("copilot_ranked.csv"))
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--no-templates", action="store_true")
    ap.add_argument("--template-threshold", type=float, default=0.78)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    cfg = load_config(args.config) if args.config.exists() else load_config(None)
    ocr_cache = OCRCache(cfg.paths.ocr_cache)

    templates_dir = None if args.no_templates else args.templates
    ml_scorer = None if args.no_ml else build_ml_scorer(args.config)

    detector = CopilotDetector(
        ocr_cache=ocr_cache,
        templates_dir=templates_dir,
        ml_scorer=ml_scorer,
        template_threshold=args.template_threshold,
    )

    log.info("Scanning %s ...", args.folder)
    results = scan_folder(args.folder, detector)
    write_csv(results, args.out)

    flagged = sum(1 for r in results if r.score >= 0.5)
    log.info("Scanned %d images; %d flagged (score>=0.5). Wrote %s",
             len(results), flagged, args.out)
    print("\nTop 10:")
    for r in results[:10]:
        ev_bits = []
        if r.strong_matches:
            ev_bits.append(f"strong=[{','.join(r.strong_matches[:2])}]")
        if r.weak_matches:
            ev_bits.append(f"weak={len(r.weak_matches)}")
        if r.template_name:
            ev_bits.append(f"tpl={r.template_name}@{r.template_confidence:.2f}")
        if r.ml_score is not None:
            ev_bits.append(f"ml={r.ml_score:.2f}")
        ev_str = " ".join(ev_bits) or "no signal"
        print(f"  {r.score:.3f}  [{r.method:11s}]  tool={r.detected_tool:8s}  "
              f"{Path(r.path).name:40s} {ev_str}")


if __name__ == "__main__":
    main()
