# CoMas Screenshot Checker

![Carleton University](comas/assets/carleton_logo.png)

## What this is

CoMas is the proctoring tool Carleton uses for online exams. It takes mutiple
screenshot of each student's desktop druing an exam. One exam with a few hundred students
produces tens of thousands of screenshots, so nobody can look at them all
properly and review ends up shallow and inconsistent.

This desktop app reads a folder of those screenshots and **sorts them
by how suspicious they look**, so a TA reviews the images most worth
their attention instead of clicking through ten thousand.

It looks for two specific things:

1. **An AI coding assistant in VS Code** — either an open Copilot or Cursor
   chat panel, or an inline "ghost text" suggestion.

2. **Leaving the Brightspace quiz** — the student navigated away from the quiz
   page they were supposed to be on.

You give it a folder of screenshots; it gives you a ranked queue with the
suspicious region boxed in red, buttons to mark each one reviewed or dismissed,
and an export you can attach to an academic integrity case.

---

# Getting it running

You need three things: **Python**, **Tesseract**,
and the **project itself**.


## macOS — step by step

**1. Install Homebrew** Open Terminal and paste:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**2. Install Python and Tesseract:**

```bash
brew install python@3.11 python-tk@3.11 tesseract
```

**3. Download the project.** 

```bash
git clone https://github.com/Sabateesh/Honours-Project.git
cd Honours-Project
```
**4. Run the installer:**

```bash
./install.sh
```

**5. Start it.** Double-click **`Run CoMas Triage.command`**, which the
installer leaves in the folder. 

## Windows — step by step

**1. Install Python** from [python.org/downloads](https://www.python.org/downloads/).

**2. Install Tesseract** from the
[UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki).

**3. Download the project** as a
[ZIP](https://github.com/Sabateesh/Honours-Project/archive/refs/heads/main.zip).

**4. Extract it 

**5. Run the installer.** Open and double-click **`install.bat`**.

**6. Start it.** Double-click **`Run CoMas Triage.bat`**, which the installer
leaves next to it.

## The detection model

The trained model ships **inside the repository**, so a clone or a ZIP download
already has everything and the install needs no network access for it:

```
Honours-Project/
  checkpoints/
    vscode_tiles_2x.pt          90 MB, the ResNet50 ghost-text detector
    vscode_tiles_2x.meta.json   backbone and input size the app reads on load
```

---

# Using it

**1. Pick a detection mode** on the opening screen:

| mode | use it for |
|---|---|
| **VSCODE Cheating** | a programming exam written in an editor |
| **Brightspace Cheating** | a quiz taken in the browser |
| **VSCODE + Brightspace Cheating** | an exam involving both |


**2. Upload the screenshots** 

**3. Review the queue.** The most suspicious images come first. Each shows its
score, a plain-language reason, and a red box around the region that triggered
the flag.

**4. Check each one** as **Reviewed** or **Dismissed**.

**5. Export.** You get a timestamped folder with copies of the flagged images,
a CSV of every score.


The threshold slider adjusts sensitivity live without re-running the analysis.
Lower it to see marginal cases, raise it is noisy.

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
score wins; the winning tile is what the app boxes in red.

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


## Licence

MIT. Built as an honours project at Carleton University, School of Computer
Science, supervised by Prof. Darryl Hill.
