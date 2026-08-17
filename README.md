# CoMas Screenshot Triage

Flags suspicious screenshots from CoMas online exams so a proctor reviews the
most likely cases first instead of clicking through thousands of images.

Two detection tasks:

1. **AI coding assistant in VS Code** — an open chat panel, or an inline
   "ghost text" suggestion at the cursor.
2. **Brightspace tab-leave** — the student navigated away from their quiz.

![Carleton University](comas/assets/carleton_logo.png)

---

## Install

### Quick install

Download the repository ([ZIP](https://github.com/Sabateesh/Honours-Project/archive/refs/heads/main.zip)
or `git clone`), then:

| | |
|---|---|
| **macOS / Linux** | open a terminal in the folder and run `./install.sh` |
| **Windows** | double-click `install.bat` |

The installer checks your Python and Tesseract, creates an isolated
environment, installs the app, downloads the detection model, and leaves a
double-clickable launcher next to it. Re-running it is safe.

It needs to download PyTorch (~2 GB), so allow ten minutes or so.

### Manual install

**Requirements:** Python 3.10+ and Tesseract. On macOS you must *not* use the
Python bundled with Xcode — it ships an old Tk that renders the window
incorrectly.

### macOS

```bash
brew install python@3.11 python-tk@3.11 tesseract
git clone https://github.com/Sabateesh/Honours-Project.git
cd Honours-Project
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Windows

Install [Python 3.11+](https://www.python.org/downloads/) — tick "Add python.exe
to PATH" and keep the "tcl/tk and IDLE" component.

Then install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Its
installer has no "add to PATH" option and the location depends on the install
type — `%LOCALAPPDATA%\Tesseract-OCR` for "just me", `C:\Program
Files\Tesseract-OCR` for "all users". `install.bat` checks both, so prefer it.
Installing manually, add whichever folder holds `tesseract.exe` to Path
(Environment Variables → User variables for a per-user install), open a new
command prompt, and confirm `tesseract --version` works. Then:

```bat
git clone https://github.com/Sabateesh/Honours-Project.git
cd Honours-Project
python -m venv .venv && .venv\Scripts\activate
pip install -e .
```

### Linux

```bash
sudo apt install python3.11 python3-tk tesseract-ocr
git clone https://github.com/Sabateesh/Honours-Project.git
cd Honours-Project
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Download the detection model

The trained model is ~90 MB, too large for the repository. Download
`vscode_tiles_2x.pt` and `vscode_tiles_2x.meta.json` from the
[Releases page](https://github.com/Sabateesh/Honours-Project/releases) and put
both in `checkpoints/`:

```
Honours-Project/
  checkpoints/
    vscode_tiles_2x.pt
    vscode_tiles_2x.meta.json
```

The app runs without it but falls back to text-only detection, which misses
most inline suggestions.

### Verify

```bash
python -c "import tkinter; print(tkinter.TkVersion)"   # must be 8.6, not 8.5
pytest -q                                              # 57 passed, 2 skipped
```

---

## Use

```bash
comas-triage          # or: python -m comas.gui
```

1. Pick a detection mode.
2. Upload the screenshots CoMas captured.
3. Review the ranked queue — most suspicious first, with the suspected region
   boxed in red.
4. Mark each **Reviewed** or **Dismissed**, then **Export report**.

**Keyboard:** `←` `→` navigate, `r` reviewed, `d` dismiss, `c` clear, `e` export.

The threshold slider adjusts sensitivity live. The export writes a timestamped
folder with copies of the flagged images, a CSV of all scores, and a summary
suitable for attaching to an academic integrity case.

---

## How it works

Two independent detectors, because the two things being detected look nothing
alike.

**Chat panels** contain the assistant's own interface text, so OCR keyword
matching finds them exactly and reports which tool it was. A keyword hit is
evidence a human can verify at a glance, so it sorts to the top of the queue.

**Inline ghost text** contains no identifying text at all — it is ordinary code
rendered in dim italic. Only a vision model can see it. A ResNet50 is applied
to overlapping native-resolution tiles of the screenshot, and the highest tile
score wins; the winning tile is what the GUI boxes in red.

The Brightspace detector uses no machine learning. It recognises the one
legitimate state (the quiz is on screen) and treats anything else as a
tab-leave, which catches sites nobody thought to blocklist.

---

## Measured performance

On 27 **real** screenshots (14 with ghost text, 13 without):

| detector | detected | false positives |
|---|---|---|
| OCR keywords only | 3/14 | 2/13 |
| **shipped model + OCR** | **11/14** | **1/13** |

Synthetic evaluation did not just overstate real performance, it inverted model
selection. The shipped checkpoint scores 0.554 AUROC on held-out *synthetic*
data and 0.868 on real screenshots; the checkpoint with the best synthetic score
(0.999) reached only 0.786 on real ones. See `report/` for the full analysis.

Small sample: 27 images. Treat these as counts, not precise percentages.

---

## Development

```bash
pip install -e ".[train,dev]"

# regenerate training data (Retina scale, ghost text only)
python -m comas.synthetic --out data_vscode --n-active 300 --n-clean 300 \
    --scale 2 --ghost-only
python -m comas.tiling --src data_vscode --out data_tiles

# train
python -m comas.train

# evaluate on real screenshots - the number that matters
python -m comas.diagnose --folder test_shots_positive --neg-folder test_shots

# compare every checkpoint on real data
python -m comas.compare_checkpoints
```

`config.yaml` is the shipped configuration. Thresholds belong to a specific
checkpoint *and* inference mode — re-derive them with `compare_checkpoints`
after any retrain rather than reusing an old value.

### Layout

```
comas/
  gui.py                    desktop reviewer interface
  copilot.py                VS Code detector: keywords, CNN, signal fusion
  brightspace.py            quiz-marker rule, IDE recognition
  tiling.py                 tile geometry, tiled inference
  ocr.py                    Tesseract wrapper, cache, parallel batching
  model.py  data.py  train.py  evaluate.py  config.py
  synthetic.py              VS Code screenshot generator
  synthetic_brightspace.py  browser screenshot generator
  diagnose.py               per-image explanation of a verdict
  compare_checkpoints.py    rank checkpoints on real data
  variant_report.py         per-variant synthetic breakdown
tests/                      59 tests
test_shots_positive/        14 real screenshots with ghost text
test_shots/                 13 real screenshots without
report/                     honours project report (LaTeX)
```

---

## Limitations

- Evaluated on 27 real screenshots — a small sample.
- The model is trained on synthetic data and does not transfer perfectly;
  roughly 3 in 14 real ghost-text screenshots are missed.
- Keyword detection breaks when a vendor renames interface text, so the list
  needs reviewing each term.
- Only GitHub Copilot and Cursor are covered by name.
- Each screenshot is judged alone; patterns across a student's session are not
  modelled.

## Licence

MIT. Built as an honours project at Carleton University, School of Computer
Science, supervised by Prof. Darryl Hill.
