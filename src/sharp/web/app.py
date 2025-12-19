import sys
from pathlib import Path
import logging
import shutil
import tempfile
import zipfile
import io as python_io
import base64

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import numpy as np

# Add src to path so we can import sharp
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io as sharp_io
from sharp.utils.gaussians import save_ply
from sharp.cli.predict import predict_image, DEFAULT_MODEL_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = FastAPI()

# Mount static files if needed (we created the dir)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Global variables for the model
predictor: RGBGaussianPredictor = None
device: torch.device = None

@app.on_event("startup")
async def startup_event():
    global predictor, device
    
    # Determine device
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    
    device = torch.device(device_str)
    LOGGER.info(f"Using device: {device}")

    # Load model
    LOGGER.info("Loading model...")
    try:
        # Try to load from cache or download
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True, map_location=device)
        
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        predictor.to(device)
        LOGGER.info("Model loaded successfully.")
    except Exception as e:
        LOGGER.error(f"Failed to load model: {e}")
        raise e

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    """Process images and return PLY data for viewing or download."""
    if not predictor:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    # Create a temporary directory to process files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        results = []
        
        for file in files:
            try:
                # Save uploaded file
                file_path = temp_path / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                LOGGER.info(f"Processing {file.filename}")
                
                # Load image using sharp's IO to get focal length and handle rotation
                image, _, f_px = sharp_io.load_rgb(file_path)
                
                # Run prediction
                gaussians = predict_image(predictor, image, f_px, device)
                
                # Save PLY
                ply_filename = f"{file_path.stem}.ply"
                ply_path = temp_path / ply_filename
                
                height, width = image.shape[:2]
                save_ply(gaussians, f_px, (height, width), ply_path)
                
                # Read PLY file and encode as base64
                with open(ply_path, "rb") as f:
                    ply_data = base64.b64encode(f.read()).decode("utf-8")
                
                results.append({
                    "filename": file.filename,
                    "ply_filename": ply_filename,
                    "ply_data": ply_data,
                    "width": width,
                    "height": height,
                    "focal_length": f_px,
                })
                
            except Exception as e:
                LOGGER.error(f"Error processing {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                })
        
        return JSONResponse({"results": results})


@app.post("/predict/download")
async def predict_download(files: list[UploadFile] = File(...)):
    """Process images and return a ZIP file for download."""
    if not predictor:
        return HTMLResponse("Model not loaded", status_code=500)

    # Create a temporary directory to process files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_zip = python_io.BytesIO()
        
        with zipfile.ZipFile(output_zip, "w") as zf:
            for file in files:
                try:
                    # Save uploaded file
                    file_path = temp_path / file.filename
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    
                    LOGGER.info(f"Processing {file.filename}")
                    
                    # Load image using sharp's IO to get focal length and handle rotation
                    image, _, f_px = sharp_io.load_rgb(file_path)
                    
                    # Run prediction
                    gaussians = predict_image(predictor, image, f_px, device)
                    
                    # Save PLY
                    ply_filename = f"{file_path.stem}.ply"
                    ply_path = temp_path / ply_filename
                    
                    height, width = image.shape[:2]
                    save_ply(gaussians, f_px, (height, width), ply_path)
                    
                    # Add to zip
                    zf.write(ply_path, ply_filename)
                    
                except Exception as e:
                    LOGGER.error(f"Error processing {file.filename}: {e}")
                    continue
        
        output_zip.seek(0)
        return StreamingResponse(
            output_zip, 
            media_type="application/zip", 
            headers={"Content-Disposition": "attachment; filename=gaussians.zip"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
