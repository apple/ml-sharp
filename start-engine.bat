@echo off
REM SHARP Engine launcher (Windows).
REM Sets up a local Python environment, installs dependencies on first run, and
REM starts the engine on http://localhost:8000. Keep this window open while using
REM the SHARP website. Close it (or Ctrl+C) to stop.

setlocal
cd /d "%~dp0"
set "VENV_DIR=.venv"

echo ==^> SHARP Engine launcher

where uv >nul 2>nul
if %ERRORLEVEL%==0 (
  echo ==^> Using uv
  if not exist "%VENV_DIR%" uv venv "%VENV_DIR%"
  call "%VENV_DIR%\Scripts\activate.bat"
  set "INSTALL=uv pip install"
) else (
  where python >nul 2>nul
  if not %ERRORLEVEL%==0 (
    echo ERROR: Python 3 not found. Install Python 3.10+ from https://www.python.org/downloads/ and re-run.
    pause
    exit /b 1
  )
  if not exist "%VENV_DIR%" python -m venv "%VENV_DIR%"
  call "%VENV_DIR%\Scripts\activate.bat"
  python -m pip install --upgrade pip >nul
  set "INSTALL=pip install"
)

python -c "import torch, fastapi, uvicorn, sharp" >nul 2>nul
if not %ERRORLEVEL%==0 (
  echo ==^> Installing dependencies ^(first run only; this can take several minutes^)
  %INSTALL% -e .
  %INSTALL% -r backend/requirements.txt
)

echo ==^> Starting engine ^(the model auto-downloads on first run^)
python -m backend
pause
