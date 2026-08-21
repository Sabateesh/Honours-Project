# PyInstaller spec for the Windows build.  Build ON Windows:
#
#   py -3.11 -m venv .venv-build
#   .venv-build\Scripts\activate
#   pip install -e .
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   pip install pyinstaller
#   pyinstaller comas.spec --noconfirm
#
# Output lands in dist\CoMas\CoMas.exe.  There is no cross-compiling: a macOS
# PyInstaller produces a macOS app, so this has to run on the target OS.
#
# onedir, not onefile.  onefile unpacks the whole bundle - including a 94 MB
# model and several hundred MB of torch - into a temp directory on EVERY
# launch, which turns startup into a minute of disk churn.  onedir starts
# instantly and is what an installer wants anyway.
import shutil
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH)

# Read-only files the app needs at runtime; comas/resources.py resolves these
# against sys._MEIPASS.  The checkpoint is the big one - keep it in step with
# paths.checkpoint in config.yaml.
datas = [
    (str(ROOT / "config.yaml"), "."),
    (str(ROOT / "checkpoints" / "vscode_tiles_2x.pt"), "checkpoints"),
    (str(ROOT / "checkpoints" / "vscode_tiles_2x.meta.json"), "checkpoints"),
    (str(ROOT / "comas" / "assets"), "comas/assets"),
]

# Tesseract is a separate executable, not a Python package: pip does not
# install it and PyInstaller cannot discover it.  Copy an installed one into
# tesseract\ beside this spec and it ships with the app; leave it out and the
# app falls back to a system install, and OCR silently does nothing without
# either.  See comas/resources.find_tesseract.
tess = ROOT / "tesseract"
if tess.exists():
    datas.append((str(tess), "tesseract"))
else:
    print("WARNING: no tesseract\\ directory - the build will depend on the "
          "user having Tesseract-OCR installed. Copy "
          "C:\\Program Files\\Tesseract-OCR (binary + tessdata) to "
          f"{tess} to bundle it.")

binaries = collect_dynamic_libs("torch")

a = Analysis(
    ["comas/__main__.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "PIL._tkinter_finder",   # tkinter image support, missed by the scanner
        "pytesseract",
        "tqdm",                  # comas.data disables the model without it
    ],
    hookspath=[],
    runtime_hooks=[],
    # Trimming.  Do NOT add tqdm here: comas/data.py imports it inside the try
    # that sets _ML_AVAILABLE, so dropping it disables the model with no error.
    excludes=[
        "matplotlib", "scipy", "pandas", "sklearn", "scikit-learn",
        "sentence_transformers", "transformers", "IPython", "notebook",
        "pytest", "sympy.plotting", "torch.distributed", "torch.testing",
        "torchvision.datasets",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="CoMas",
    console=False,           # no terminal window behind the GUI
    icon=None,               # point at a .ico here once you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,               # UPX corrupts some torch DLLs
    name="CoMas",
)
