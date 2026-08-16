# CoMas Screenshot Triage

Flags suspicious CoMas exam screenshots (AI coding assistants in VS Code, tab-leaves
from Brightspace quizzes) so proctors review the most suspicious images first.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tesseract must be installed separately (`brew install tesseract`).

### macOS Tk warning

Don't use Xcode's bundled `python3` for the GUI. It ships an old Tk (8.5) with
known rendering bugs on modern macOS — windows can render half-blank until
resized. Use Homebrew Python (`brew install python-tk@3.11`) or the python.org
installer instead, then create your venv from that interpreter.

## Running

```
python3 -m comas.gui          # desktop app
python3 -m pytest tests/      # test suite
```

## Detection signals

VS Code screenshots are scored by a CNN first; OCR keyword matching only runs on
images the model is not already confident about. The two catch different things:

| | ghost text | chat panel | unseen assistant |
|---|---|---|---|
| CNN | yes | yes | maybe |
| OCR keywords | no | yes | yes, if named |

OCR is kept as the fallback because it degrades differently: keyword matching
breaks when a vendor renames a button, the CNN breaks on unfamiliar themes or a
redesigned UI. It is also the baseline the CNN is measured against.

## Regenerating training data

```
python3 -m comas.synthetic --out data_vscode --n-active 200 --n-clean 200
python3 -m comas.synthetic_brightspace --out data_brightspace
# non-IDE negatives: a real CoMas stream is mostly browsers, not editors
cp data_brightspace/on_quiz/*.png data_brightspace/left_quiz/*.png data_vscode/no_copilot/
```

After training, break the headline number down by failure mode:

```
python3 -m comas.variant_report          # recall for ghost/panel/both,
                                         # FPR for clean/hardneg/browser negatives
```

Half the VS Code negatives deliberately show a non-chat side panel (Outline,
Extensions, Source Control, Search). Without them the model learns "panel on the
right = cheating" and flags anyone with the Extensions view open. Tune with
`--hard-neg-frac`.
