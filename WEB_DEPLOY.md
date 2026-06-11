# Deploying the SHARP web app

This deploys the **website** (the React frontend) to the public web (Vercel). The
website itself does no ML work — when someone uses it, the processing runs on **their
own machine** via the local [SHARP Engine](ENGINE_SETUP.md). You host the site once;
each visitor runs the engine on their own computer.

```
  Visitor's browser ──HTTPS──▶  sharp.vercel.app  (static site you host)
        │
        └──http://localhost:8000──▶  SHARP Engine  (runs on the visitor's own machine)
```

## Why an engine is still required

A browser tab cannot reach a GPU or run the PyTorch model on its own. "Use the
visitor's local compute" therefore requires a small program (the engine) running on
that visitor's machine. The site auto-detects it and guides users through starting it.

## Deploy the frontend to Vercel

The frontend lives in `web-client/`.

1. Push this repo to GitHub.
2. In Vercel: **New Project** → import the repo.
3. Set **Root Directory** to `web-client`. Vercel auto-detects Vite
   (`web-client/vercel.json` pins the build command, output dir, and SPA rewrite).
4. Deploy. You get a public URL like `https://sharp-xyz.vercel.app`.

CLI alternative:

```bash
cd web-client
npm install
vercel --prod   # set Root Directory to the current dir when prompted
```

### Optional build-time env vars (Vercel → Project → Settings → Environment Variables)

- `VITE_API_BASE` — default backend URL. Leave unset; the default
  `http://localhost:8000` is correct because every visitor runs their own engine.
- `VITE_ENGINE_REPO` — link shown in the in-app setup card for downloading the engine.
  Set this to your repo URL (e.g. your GitHub fork) so users get the right code.

## What visitors do

1. Open your Vercel URL in **Chrome, Edge, or Firefox** (not Safari — see below).
2. Follow the in-app **SHARP Engine** card (mirrors [ENGINE_SETUP.md](ENGINE_SETUP.md)):
   download the engine and run `./start-engine.command` / `.sh` / `.bat`.
3. Click **Allow** on the browser's "Local Network Access" prompt.
4. Drop in a photo — it processes on their machine and renders in the browser.

## Browser support

| Browser            | Works? | Notes |
|--------------------|--------|-------|
| Chrome / Edge 142+ | ✅     | One-time "Local Network Access" permission prompt; click Allow. |
| Firefox            | ✅     | Reaches `http://localhost` from HTTPS. |
| Safari             | ❌     | WebKit blocks HTTPS pages from contacting `localhost`. The site shows a notice telling users to switch browsers. |

## Local development (unchanged)

```bash
# Terminal 1 — engine
python -m backend          # or ./start-engine.sh

# Terminal 2 — frontend dev server
cd web-client
npm install
npm run dev                # http://localhost:5173
```

Both are same-origin-ish on localhost, so there's no permission prompt during dev.

## Security note

The engine listens on `127.0.0.1` (loopback) only — it is **not** exposed to the
network or internet, so no one else can reach a visitor's engine. CORS allows the
website's origin to call it; override the allowed origins with `SHARP_ALLOWED_ORIGINS`
(comma-separated) if you want to restrict it to just your Vercel domain.
