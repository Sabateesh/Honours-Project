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
