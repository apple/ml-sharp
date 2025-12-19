# Sharp Web Interface

This is a web interface for the Sharp 3D prediction model.

## Prerequisites

Make sure you have the `sharp` package installed (see root README).
Install the web dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

Run the following command from the `web` directory:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Usage

1. Open your browser and navigate to `http://localhost:8000`.
2. Drag and drop images or click to select them.
3. Click "Predict 3D Gaussians".
4. A zip file containing the resulting `.ply` files will be downloaded automatically.
