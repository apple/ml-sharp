# Running the SHARP Engine

The SHARP website runs in your browser, but the actual 3D processing happens on
**your own computer** using its GPU/CPU. To do that, a small local "engine" must
be running. You start it once and leave it running while you use the site.

> **Browser note:** Use **Chrome, Edge, or Firefox**. Safari blocks websites from
> talking to programs running on your own machine, so the engine will not connect
> there.

## 1. Get the code

Download or clone this project, then open a terminal/Finder in the project folder.

## 2. Start the engine

**macOS** — double-click `start-engine.command` (or run `./start-engine.sh` in a terminal).

**Linux** — run:

```bash
./start-engine.sh
```

**Windows** — double-click `start-engine.bat`.

The first launch creates a local environment, installs dependencies, and downloads
the model (a few minutes, one time). After that it starts in seconds. When you see

```
SHARP Engine starting on http://127.0.0.1:8000
```

the engine is ready. **Leave this window open** while using the site.

## 3. Use the site

Open the SHARP website. The status pill at the top right should turn green
("Engine connected"). Your browser may ask to **"Look for and connect to devices on
your local network"** — click **Allow** (this lets the site reach the engine on your
own machine). Then drop in a photo.

## Requirements

- Python 3.10+ (the launcher uses [uv](https://docs.astral.sh/uv/) if installed,
  otherwise the system `python3`).
- ~3 GB free disk for dependencies + model.
- A CUDA GPU or Apple Silicon (MPS) is much faster, but CPU works too.

## Troubleshooting

- **Pill stays red:** make sure the engine window is still open and shows it started
  on port 8000. Click **Retry connection** in the site's engine panel.
- **Different port:** set `SHARP_PORT=9000` before launching, then in the site's
  engine panel → **Advanced**, set the Engine URL to `http://localhost:9000`.
- **"Python not found":** install Python 3.10+ from <https://www.python.org/downloads/>.
