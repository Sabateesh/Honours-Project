@echo off
REM One-step installer for Windows. Double-click this file.
REM
REM Creates an isolated environment, installs the app, fetches the detection
REM model, and writes a launcher. Safe to re-run.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set REPO_URL=https://github.com/Sabateesh/Honours-Project
set MODEL_TAG=v1.0.0

echo CoMas Screenshot Triage - installer
echo.

REM ------------------------------------------------------------- Python ---
set PY=
for %%C in (python3.12 python3.11 python) do (
    where %%C >nul 2>&1 && (
        %%C -c "import sys,tkinter; sys.exit(0 if sys.version_info>=(3,10) and float(str(tkinter.TkVersion))>=8.6 else 1)" >nul 2>&1 && (
            set PY=%%C
            goto :found_python
        )
    )
)
:found_python

if "%PY%"=="" (
    echo Could not find Python 3.10+ with Tk support.
    echo.
    echo Install it from https://www.python.org/downloads/
    echo IMPORTANT: tick "Add python.exe to PATH" during setup, and leave
    echo the "tcl/tk and IDLE" component enabled.
    echo.
    pause
    exit /b 1
)
for /f "delims=" %%V in ('%PY% --version') do echo Python:    %%V

REM ---------------------------------------------------------- Tesseract ---
where tesseract >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Tesseract is not installed or not on PATH.
    echo Text-based detection will not work without it.
    echo Get it from https://github.com/UB-Mannheim/tesseract/wiki
    echo and tick "Add to PATH" during setup.
    echo.
    set /p CONT="Continue without it? [y/N] "
    if /i not "!CONT!"=="y" exit /b 1
) else (
    echo Tesseract: found
)

REM -------------------------------------------------------- environment ---
echo.
echo Creating environment in .venv ...
if not exist .venv\Scripts\python.exe %PY% -m venv .venv
call .venv\Scripts\activate.bat

echo Installing (this downloads PyTorch, about 2 GB - it takes a while) ...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e .
if errorlevel 1 (
    echo.
    echo Installation failed. Scroll up for the error.
    pause
    exit /b 1
)

REM -------------------------------------------------------------- model ---
echo.
if not exist checkpoints mkdir checkpoints
set NEEDMODEL=0
if not exist checkpoints\vscode_tiles_2x.pt set NEEDMODEL=1
if not exist checkpoints\vscode_tiles_2x.meta.json set NEEDMODEL=1

if "%NEEDMODEL%"=="0" (
    echo Model:     already present
) else (
    echo Downloading the detection model ^(~90 MB^) ...
    powershell -NoProfile -Command ^
      "$ErrorActionPreference='Stop';" ^
      "foreach ($f in 'vscode_tiles_2x.pt','vscode_tiles_2x.meta.json') {" ^
      "  Invoke-WebRequest -Uri \"%REPO_URL%/releases/download/%MODEL_TAG%/$f\" -OutFile \"checkpoints/$f\" }"
    if errorlevel 1 (
        echo.
        echo Could not download the model.
        echo Get it manually from %REPO_URL%/releases and put both files in
        echo   %CD%\checkpoints\
        echo The app still runs without it, but only text-based detection works.
    )
)

REM ----------------------------------------------------------- launcher ---
> "Run CoMas Triage.bat" echo @echo off
>> "Run CoMas Triage.bat" echo cd /d "%%~dp0"
>> "Run CoMas Triage.bat" echo call .venv\Scripts\activate.bat
>> "Run CoMas Triage.bat" echo comas-triage

echo.
echo Done.
echo.
echo   Double-click "Run CoMas Triage.bat"
echo.
pause
