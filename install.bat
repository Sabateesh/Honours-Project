@echo off
REM One-step installer for Windows. Double-click this file.
REM
REM Creates an isolated environment, installs the app, fetches the detection
REM model, and writes a launcher. Safe to re-run.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set REPO_URL=https://github.com/Sabateesh/Honours-Project
set MODEL_TAG=v1.1.0

echo CoMas Screenshot Triage - installer
echo.

REM ------------------------------------------------------- path sanity ---
REM PyTorch ships headers nested nine levels deep. Windows caps paths at 260
REM characters, so a long folder makes pip fail with a confusing "No such
REM file or directory" after a 2 GB download. Running the .bat straight out
REM of a ZIP is the usual cause: Explorer unpacks to a Temp folder with a
REM GUID in the name and there is no room left.
set "HERE=%CD%"
set "PATHWARN="
echo !HERE! | findstr /i "\\Temp\\" >nul && set PATHWARN=1
if not "!HERE:~70!"=="" set PATHWARN=1

if defined PATHWARN (
    echo This folder is too long, or is a temporary folder:
    echo   !HERE!
    echo.
    echo Installing PyTorch here will fail on Windows' 260-character path
    echo limit. If you double-clicked install.bat inside the ZIP, Windows
    echo unpacked it somewhere temporary - that is the usual cause.
    echo.
    echo Fix: right-click the ZIP, Extract All, set the destination to
    echo      C:\CoMas   then run install.bat from there.
    echo.
    pause
    exit /b 1
)

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
REM The UB-Mannheim installer has no "add to PATH" option, and installs to
REM AppData for a per-user install or Program Files for an all-users one.
REM Rather than making the user edit PATH, look in the known locations and
REM bake whatever we find into the launcher.
set TESSDIR=
where tesseract >nul 2>&1
if not errorlevel 1 (
    echo Tesseract: found on PATH
) else (
    for %%D in (
        "%LOCALAPPDATA%\Tesseract-OCR"
        "%LOCALAPPDATA%\Programs\Tesseract-OCR"
        "%ProgramFiles%\Tesseract-OCR"
        "%ProgramFiles(x86)%\Tesseract-OCR"
    ) do (
        if exist "%%~D\tesseract.exe" if not defined TESSDIR set "TESSDIR=%%~D"
    )
)

if defined TESSDIR (
    echo Tesseract: found in !TESSDIR!
    set "PATH=!TESSDIR!;%PATH%"
)

where tesseract >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Tesseract not found. Text-based detection will not work,
    echo so chat panels and Brightspace tab-leaves will be missed.
    echo.
    echo Install it from https://github.com/UB-Mannheim/tesseract/wiki
    echo then run this installer again - it looks in the default locations,
    echo so you do not need to edit PATH yourself.
    echo.
    set /p CONT="Continue without it? [y/N] "
    if /i not "!CONT!"=="y" exit /b 1
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
REM Tesseract is not on PATH for a per-user install, so pin it in the launcher
if defined TESSDIR >> "Run CoMas Triage.bat" echo set "PATH=!TESSDIR!;%%PATH%%"
>> "Run CoMas Triage.bat" echo comas-triage

echo.
echo Done.
echo.
echo   Double-click "Run CoMas Triage.bat"
echo.
pause
