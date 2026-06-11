#!/usr/bin/env bash
#
# SHARP Engine launcher (macOS / Linux).
#
# Sets up a local Python environment, installs dependencies on first run, and
# starts the engine on http://localhost:8000. Keep the window open while using
# the SHARP website. Press Ctrl+C to stop.
#
set -euo pipefail

# Resolve to the project root (directory containing this script).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

echo "==> SHARP Engine launcher"

# Prefer uv (fast; can bootstrap Python itself). Fall back to system python3.
if command -v uv >/dev/null 2>&1; then
  echo "==> Using uv"
  [ -d "$VENV_DIR" ] || uv venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  INSTALL_CMD="uv pip install"
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "ERROR: Python 3 not found." >&2
    echo "Install Python 3.10+ from https://www.python.org/downloads/ (or 'brew install python'), then re-run." >&2
    exit 1
  fi
  [ -d "$VENV_DIR" ] || { echo "==> Creating virtual environment"; "$PYTHON_BIN" -m venv "$VENV_DIR"; }
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip >/dev/null
  INSTALL_CMD="pip install"
fi

# Install dependencies only when something is missing (keeps later launches fast).
if ! python -c "import torch, fastapi, uvicorn, sharp" >/dev/null 2>&1; then
  echo "==> Installing dependencies (first run only; this can take several minutes)"
  $INSTALL_CMD -e .
  $INSTALL_CMD -r backend/requirements.txt
fi

echo "==> Starting engine (the model auto-downloads on first run)"
exec python -m backend
