#!/usr/bin/env bash
# One-step installer for macOS and Linux.
#
#   ./install.sh
#
# Creates an isolated environment, installs the app, fetches the detection
# model, and writes a double-clickable launcher. Safe to re-run.
set -euo pipefail

REPO_URL="https://github.com/Sabateesh/Honours-Project"
MODEL_TAG="v1.0.0"
MODEL_FILES=("vscode_tiles_2x.pt" "vscode_tiles_2x.meta.json")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
die()  { printf '\033[31mError: %s\033[0m\n' "$*" >&2; exit 1; }

say "CoMas Screenshot Triage - installer"
echo

# ---------------------------------------------------------------- Python ---
# Needs 3.10+ with a working Tk 8.6. macOS ships a Python whose Tk 8.5 draws
# the window incorrectly, so the version is checked rather than assumed.
PY=""
for c in python3.12 python3.11 python3.10 python3; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" - <<'EOF' >/dev/null 2>&1 || continue
import sys, tkinter
assert sys.version_info >= (3, 10)
assert float(str(tkinter.TkVersion)) >= 8.6
EOF
    PY="$c"; break
done

if [ -z "$PY" ]; then
    warn "No suitable Python found (need 3.10+ with Tk 8.6)."
    echo
    if [[ "$OSTYPE" == darwin* ]]; then
        echo "Install it with Homebrew, then re-run this script:"
        echo "    brew install python@3.11 python-tk@3.11"
        echo
        echo "Note: the Python bundled with macOS/Xcode will not work - its"
        echo "Tk 8.5 renders the window half-blank."
    else
        echo "Install it with your package manager, then re-run this script:"
        echo "    sudo apt install python3.11 python3.11-venv python3-tk"
    fi
    exit 1
fi
say "Python:    $($PY --version) with Tk $($PY -c 'import tkinter;print(tkinter.TkVersion)')"

# ------------------------------------------------------------- Tesseract ---
if command -v tesseract >/dev/null 2>&1; then
    say "Tesseract: $(tesseract --version 2>&1 | head -1)"
else
    warn "Tesseract is not installed - text-based detection will not work."
    if [[ "$OSTYPE" == darwin* ]]; then
        echo "    brew install tesseract"
    else
        echo "    sudo apt install tesseract-ocr"
    fi
    echo
    read -r -p "Continue without it? [y/N] " a
    [[ "$a" =~ ^[Yy]$ ]] || exit 1
fi

# ----------------------------------------------------------- environment ---
echo
say "Creating environment in .venv ..."
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

say "Installing (this downloads PyTorch, about 2 GB - it takes a while) ..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e .

# ----------------------------------------------------------------- model ---
echo
mkdir -p checkpoints
missing=0
for f in "${MODEL_FILES[@]}"; do [ -f "checkpoints/$f" ] || missing=1; done

if [ "$missing" -eq 0 ]; then
    say "Model:     already present"
else
    say "Downloading the detection model (~90 MB) ..."
    ok=1
    for f in "${MODEL_FILES[@]}"; do
        url="$REPO_URL/releases/download/$MODEL_TAG/$f"
        curl -fL --progress-bar -o "checkpoints/$f.part" "$url" \
            && mv "checkpoints/$f.part" "checkpoints/$f" \
            || { rm -f "checkpoints/$f.part"; ok=0; }
    done
    if [ "$ok" -eq 0 ]; then
        warn "Could not download the model."
        echo "Get it manually from $REPO_URL/releases and put both files in"
        echo "  $HERE/checkpoints/"
        echo "The app still runs without it, but only text-based detection"
        echo "works and most inline suggestions will be missed."
    fi
fi

# -------------------------------------------------------------- launcher ---
cat > "Run CoMas Triage.command" <<EOF
#!/usr/bin/env bash
cd "\$(dirname "\$0")"
source .venv/bin/activate
exec comas-triage
EOF
chmod +x "Run CoMas Triage.command"

echo
say "Done."
echo
echo "  Double-click 'Run CoMas Triage.command'"
echo "  or from a terminal:  source .venv/bin/activate && comas-triage"
echo
