"""Run the SHARP Engine: `python -m backend`.

Starts the FastAPI app (backend.main:app) with uvicorn. The model is loaded and
auto-downloaded on first startup (see backend/main.py lifespan). Host/port are
configurable via SHARP_HOST / SHARP_PORT.
"""

import os
import sys
from pathlib import Path


def main() -> None:
    import uvicorn

    # Ensure the project root is importable so "backend.main" resolves regardless
    # of the current working directory.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    host = os.environ.get("SHARP_HOST", "127.0.0.1")
    port = int(os.environ.get("SHARP_PORT", "8000"))

    print(f"\n  SHARP Engine starting on http://{host}:{port}")
    print("  Keep this window open while using the SHARP website.")
    print("  Press Ctrl+C to stop.\n")

    uvicorn.run("backend.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
