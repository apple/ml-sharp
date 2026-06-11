import sys
import os
from pathlib import Path
import logging
import torch
import shutil
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import json
import tempfile

# Add project root and src to sys.path to import sharp and backend
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.cli.predict import predict_image
from backend.convert import gaussians_to_splat
from backend.equirect_to_cubemap import (
    equirect_to_cubemap,
    get_cubemap_focal_length,
    FACE_NAMES,
)
from backend.merge_gaussians import (
    transform_gaussians,
    merge_gaussians,
    apply_rigid_transform,
)
from backend.station_transform import (
    build_station_transform,
    parse_stations_json,
    match_stations_to_files,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Global variables
gaussian_predictor = None
device = "cpu"
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

# Job Store
JOBS: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown."""
    global gaussian_predictor, device
    
    # Startup
    try:
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        LOGGER.info(f"Using device: {device}")

        # Load model
        LOGGER.info("Loading SHARP model...")
        cache_dir = Path.home() / ".cache" / "sharp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / "sharp_model.pt"

        if not checkpoint_path.exists():
            LOGGER.info(f"Downloading model from {DEFAULT_MODEL_URL}")
            # Download and save to specific path
            state_dict = torch.hub.load_state_dict_from_url(
                DEFAULT_MODEL_URL, 
                progress=True, 
                model_dir=str(cache_dir), 
                map_location=device,
                file_name="sharp_model.pt"
            )
        else:
            LOGGER.info(f"Loading cached model from {checkpoint_path}")
            state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)

        gaussian_predictor = create_predictor(PredictorParams())
        gaussian_predictor.load_state_dict(state_dict)
        gaussian_predictor.eval()
        gaussian_predictor.to(device)
        LOGGER.info("Model loaded successfully.")
    except Exception as e:
        LOGGER.error(f"Failed to load model: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown (if needed)
    LOGGER.info("Shutting down...")


app = FastAPI(lifespan=lifespan)


class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    """Answer legacy Chromium Private Network Access (PNA) preflights.

    When the website is served over HTTPS (e.g. Vercel) but the engine runs at
    http://localhost on the visitor's machine, older Chromium versions send a
    preflight with `Access-Control-Request-Private-Network: true` and require the
    response to echo `Access-Control-Allow-Private-Network: true`. Newer Chrome
    (142+) uses a user permission prompt instead and ignores this header, so adding
    it is harmless and only helps older browsers.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.headers.get("Access-Control-Request-Private-Network") == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response


# Allowed origins for the hosted frontend. The deployed site's origin must be
# listed (CORS), plus the Vite dev server. Override with SHARP_ALLOWED_ORIGINS
# (comma-separated) to add your Vercel URL, e.g.
#   SHARP_ALLOWED_ORIGINS="https://sharp.vercel.app"
_default_origins = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:4173",  # Vite preview server
    "http://127.0.0.1:4173",
]
_env_origins = os.environ.get("SHARP_ALLOWED_ORIGINS", "").strip()
if _env_origins == "*":
    ALLOWED_ORIGINS = ["*"]
elif _env_origins:
    ALLOWED_ORIGINS = _default_origins + [
        o.strip() for o in _env_origins.split(",") if o.strip()
    ]
else:
    # Default: allow any origin. We send no cookies, so this is safe and lets the
    # engine work from any deployed frontend without per-user configuration.
    ALLOWED_ORIGINS = ["*"]

# Enable CORS for the frontend. allow_credentials must be False when origins is
# "*" (the wildcard + credentials combination is rejected by browsers); we use no
# cookies, so credentials are not needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Added LAST so it is the OUTERMOST middleware: CORSMiddleware answers (and
# short-circuits) preflight OPTIONS requests, so the PNA header must be appended
# on the way back out, after CORS has produced the preflight response.
app.add_middleware(PrivateNetworkAccessMiddleware)

@app.get("/")
async def root():
    return {"status": "running", "service": "sharp-backend", "model_loaded": gaussian_predictor is not None}

async def process_job(job_id: str, file_path: Path, original_filename: str):
    """Background worker to process the image.

    Runs heavy CPU/GPU work in a thread pool so the event loop stays
    responsive for status-polling requests.
    """
    try:
        LOGGER.info(f"Starting job {job_id}")
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["progress"] = 10
        JOBS[job_id]["message"] = "Preprocessing Image..."

        # 1. Load Image (in thread to avoid blocking event loop)
        image, _, f_px = await asyncio.to_thread(io.load_rgb, file_path)
        height, width = image.shape[:2]

        JOBS[job_id]["progress"] = 30
        JOBS[job_id]["message"] = "Running AI Inference..."
        await asyncio.sleep(0)  # yield to event loop

        # 2. Inference
        gaussians = await asyncio.to_thread(
            predict_image, gaussian_predictor, image, f_px, torch.device(device)
        )

        JOBS[job_id]["progress"] = 80
        JOBS[job_id]["message"] = "Converting to .splat format..."
        await asyncio.sleep(0)

        # 3. Convert directly to SPLAT (bypasses PLY round-trip)
        splat_bytes = await asyncio.to_thread(gaussians_to_splat, gaussians)
        del gaussians

        JOBS[job_id]["progress"] = 95
        JOBS[job_id]["message"] = "Writing output file..."
        await asyncio.sleep(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".splat", mode='wb') as tmp_splat:
            tmp_splat.write(splat_bytes)
            tmp_splat_path = Path(tmp_splat.name)
        del splat_bytes

        JOBS[job_id]["status"] = "complete"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["message"] = "Ready!"
        JOBS[job_id]["result_path"] = str(tmp_splat_path)
        JOBS[job_id]["filename"] = f"{Path(original_filename).stem}.splat"

        LOGGER.info(f"Job {job_id} completed successfully.")

    except Exception as e:
        LOGGER.error(f"Job {job_id} failed: {e}")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
    finally:
        # Cleanup input file
        if file_path.exists():
            os.remove(file_path)

async def process_job_360(job_id: str, file_path: Path, original_filename: str):
    """Background worker to process a 360 equirectangular image.

    Pipeline:
    1. Load equirectangular image
    2. Validate ~2:1 aspect ratio
    3. Extract 6 cubemap faces (90-degree FOV each)
    4. Run SHARP on each face
    5. Transform each face's Gaussians to shared world frame
    6. Merge all Gaussians
    7. Save combined PLY and convert to .splat

    Heavy CPU/GPU work runs in a thread pool via asyncio.to_thread()
    so the event loop stays responsive for status-polling requests.
    """
    try:
        LOGGER.info(f"Starting 360 job {job_id}")
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["progress"] = 5
        JOBS[job_id]["message"] = "Loading 360 image..."

        # 1. Load Image
        image, _, _ = await asyncio.to_thread(io.load_rgb, file_path)
        height, width = image.shape[:2]

        # 2. Validate aspect ratio (~2:1 for equirectangular)
        aspect_ratio = width / height
        if aspect_ratio < 1.8 or aspect_ratio > 2.2:
            raise ValueError(
                f"Image aspect ratio {aspect_ratio:.2f} is not close to 2:1. "
                f"Expected an equirectangular 360 image."
            )

        JOBS[job_id]["progress"] = 10
        JOBS[job_id]["message"] = "Extracting cubemap faces..."
        await asyncio.sleep(0)

        # 3. Extract 6 cubemap faces
        face_size = width // 4
        faces = await asyncio.to_thread(equirect_to_cubemap, image, face_size)
        f_px = get_cubemap_focal_length(face_size)

        LOGGER.info(
            f"360 job {job_id}: Extracted {len(faces)} cubemap faces "
            f"({face_size}x{face_size}, f_px={f_px:.1f})"
        )

        # 4 & 5. Run SHARP on each face and transform to world frame
        all_gaussians = []
        for i, face_name in enumerate(FACE_NAMES):
            face_progress = 15 + i * 10  # 15, 25, 35, 45, 55, 65
            JOBS[job_id]["progress"] = face_progress
            JOBS[job_id]["message"] = f"Processing face {i + 1}/6 ({face_name})..."
            await asyncio.sleep(0)  # yield so polls can be served

            LOGGER.info(f"360 job {job_id}: Processing face '{face_name}'")

            face_image = faces[face_name]
            gaussians = await asyncio.to_thread(
                predict_image,
                gaussian_predictor, face_image, f_px, torch.device(device),
            )

            # Transform to world frame
            gaussians_world = await asyncio.to_thread(
                transform_gaussians, gaussians, face_name, torch.device(device),
            )
            all_gaussians.append(gaussians_world)

        # 6. Merge all Gaussians
        JOBS[job_id]["progress"] = 80
        JOBS[job_id]["message"] = "Merging Gaussians from all faces..."
        await asyncio.sleep(0)

        merged_gaussians = await asyncio.to_thread(merge_gaussians, all_gaussians)
        total_count = merged_gaussians.mean_vectors.shape[1]
        LOGGER.info(
            f"360 job {job_id}: Merged {total_count} total Gaussians "
            f"from {len(all_gaussians)} faces"
        )

        # 7. Convert directly to SPLAT (bypasses PLY round-trip)
        JOBS[job_id]["progress"] = 85
        JOBS[job_id]["message"] = "Converting to .splat format..."
        await asyncio.sleep(0)

        splat_bytes = await asyncio.to_thread(gaussians_to_splat, merged_gaussians)
        del merged_gaussians

        JOBS[job_id]["progress"] = 95
        JOBS[job_id]["message"] = "Writing output file..."
        await asyncio.sleep(0)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".splat", mode="wb"
        ) as tmp_splat:
            tmp_splat.write(splat_bytes)
            tmp_splat_path = Path(tmp_splat.name)
        del splat_bytes

        JOBS[job_id]["status"] = "complete"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["message"] = "Ready!"
        JOBS[job_id]["result_path"] = str(tmp_splat_path)
        JOBS[job_id]["filename"] = f"{Path(original_filename).stem}_360.splat"

        LOGGER.info(f"360 job {job_id} completed successfully.")

    except Exception as e:
        LOGGER.error(f"360 job {job_id} failed: {e}", exc_info=True)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
    finally:
        # Cleanup input file
        if file_path.exists():
            os.remove(file_path)


async def process_job_multistation(
    job_id: str,
    stations_data: List[dict],
    image_paths: Dict[str, Path],
    job_name: str,
):
    """Background worker to process multiple 360 stations into a single scene.

    Pipeline per station:
    1. Load 360 equirectangular image
    2. Extract 6 cubemap faces
    3. Run SHARP on each face
    4. Transform faces to station-local world frame
    5. Merge faces into station gaussians
    6. Apply station world-space transform (position + orientation)

    Then:
    7. Merge all station gaussians into a single scene
    8. Save PLY and convert to .splat
    """
    total_stations = len(stations_data)
    temp_files_to_cleanup: List[Path] = []

    try:
        LOGGER.info(
            f"Starting multi-station job {job_id} with {total_stations} stations"
        )
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["total_stations"] = total_stations

        # Progress allocation:
        #   0-5%   : setup
        #   5-85%  : per-station processing (each station gets equal share)
        #   85-95% : final merge + save PLY
        #   95-100%: convert to splat
        per_station_progress = 80.0 / total_stations  # share of 5%-85%

        all_station_gaussians = []

        for station_idx, station in enumerate(stations_data):
            station_name = station.get("name", station["id"])
            station_basename = Path(station["path_to_image"]).name

            LOGGER.info(
                f"Multi-station job {job_id}: Processing station "
                f"{station_idx + 1}/{total_stations} ({station_name})"
            )

            base_progress = 5 + station_idx * per_station_progress
            JOBS[job_id]["current_station"] = station_idx + 1
            JOBS[job_id]["progress"] = int(base_progress)
            JOBS[job_id]["message"] = (
                f"Station {station_idx + 1}/{total_stations} ({station_name}): "
                f"Loading image..."
            )
            await asyncio.sleep(0)

            # 1. Load the 360 image for this station
            image_path = image_paths[station_basename]
            image, _, _ = await asyncio.to_thread(io.load_rgb, image_path)
            height, width = image.shape[:2]

            # Validate aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < 1.8 or aspect_ratio > 2.2:
                raise ValueError(
                    f"Station '{station_name}': image aspect ratio "
                    f"{aspect_ratio:.2f} is not ~2:1 (expected equirectangular)"
                )

            # 2. Extract cubemap faces
            face_size = width // 4
            faces = await asyncio.to_thread(equirect_to_cubemap, image, face_size)
            f_px = get_cubemap_focal_length(face_size)

            # Free the full equirectangular image to save memory
            del image

            # 3 & 4. SHARP on each face + transform to station-local world frame
            face_gaussians = []
            for face_idx, face_name in enumerate(FACE_NAMES):
                face_progress = base_progress + (face_idx / 6.0) * per_station_progress * 0.8
                JOBS[job_id]["progress"] = int(face_progress)
                JOBS[job_id]["message"] = (
                    f"Station {station_idx + 1}/{total_stations} ({station_name}): "
                    f"Processing face {face_idx + 1}/6 ({face_name})..."
                )
                await asyncio.sleep(0)

                face_image = faces[face_name]
                gaussians = await asyncio.to_thread(
                    predict_image,
                    gaussian_predictor, face_image, f_px, torch.device(device),
                )

                gaussians_world = await asyncio.to_thread(
                    transform_gaussians, gaussians, face_name, torch.device(device),
                )
                face_gaussians.append(gaussians_world)

            # Free face images
            del faces

            # 5. Merge all 6 faces into station gaussians
            merge_progress = base_progress + per_station_progress * 0.85
            JOBS[job_id]["progress"] = int(merge_progress)
            JOBS[job_id]["message"] = (
                f"Station {station_idx + 1}/{total_stations} ({station_name}): "
                f"Merging faces..."
            )
            await asyncio.sleep(0)

            station_gaussians = await asyncio.to_thread(merge_gaussians, face_gaussians)
            del face_gaussians

            station_count = station_gaussians.mean_vectors.shape[1]
            LOGGER.info(
                f"Multi-station job {job_id}: Station '{station_name}' produced "
                f"{station_count} gaussians"
            )

            # 6. Apply station world-space transform (rigid -- no SVD needed)
            JOBS[job_id]["message"] = (
                f"Station {station_idx + 1}/{total_stations} ({station_name}): "
                f"Applying world transform..."
            )
            await asyncio.sleep(0)

            station_transform = build_station_transform(
                station["position_3d"],
                station["orientation_3d"],
                torch.device(device),
            )
            station_gaussians_global = await asyncio.to_thread(
                apply_rigid_transform, station_gaussians, station_transform,
            )
            del station_gaussians

            # Move to CPU immediately to free GPU memory for the next station
            station_gaussians_global = station_gaussians_global.to(
                torch.device("cpu")
            )
            all_station_gaussians.append(station_gaussians_global)

        # 7. Merge all stations into a single scene
        JOBS[job_id]["progress"] = 85
        JOBS[job_id]["message"] = (
            f"Merging {total_stations} stations into single scene..."
        )
        await asyncio.sleep(0)

        merged_gaussians = await asyncio.to_thread(
            merge_gaussians, all_station_gaussians
        )
        del all_station_gaussians

        total_count = merged_gaussians.mean_vectors.shape[1]
        LOGGER.info(
            f"Multi-station job {job_id}: Merged {total_count} total gaussians "
            f"from {total_stations} stations"
        )

        # 8. Convert directly to SPLAT (bypasses PLY entirely)
        JOBS[job_id]["progress"] = 88
        JOBS[job_id]["message"] = "Converting to .splat format (sorting + packing)..."
        await asyncio.sleep(0)

        splat_bytes = await asyncio.to_thread(gaussians_to_splat, merged_gaussians)
        del merged_gaussians

        JOBS[job_id]["progress"] = 97
        JOBS[job_id]["message"] = "Writing final output..."
        await asyncio.sleep(0)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".splat", mode="wb"
        ) as tmp_splat:
            tmp_splat.write(splat_bytes)
            tmp_splat_path = Path(tmp_splat.name)
        del splat_bytes

        JOBS[job_id]["status"] = "complete"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["message"] = "Ready!"
        JOBS[job_id]["result_path"] = str(tmp_splat_path)
        JOBS[job_id]["filename"] = f"{job_name}_multistation.splat"

        LOGGER.info(
            f"Multi-station job {job_id} completed successfully "
            f"({total_stations} stations, {total_count} total gaussians)."
        )

    except Exception as e:
        LOGGER.error(f"Multi-station job {job_id} failed: {e}", exc_info=True)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
    finally:
        # Cleanup all input images
        for path in image_paths.values():
            if path.exists():
                try:
                    os.remove(path)
                except OSError:
                    pass
        # Cleanup any temp files
        for path in temp_files_to_cleanup:
            if path.exists():
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.post("/predict-multistation")
async def submit_multistation_job(
    background_tasks: BackgroundTasks,
    stations_json: UploadFile = File(...),
    files: List[UploadFile] = File(...),
):
    """Submit multiple 360 images + stations.json for multi-station processing.

    Each station's 360 image is processed through the full 360 pipeline,
    then positioned in world space using the station's pose from the JSON.
    All stations are merged into a single .splat scene.
    """
    if not gaussian_predictor:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please check server logs."
        )

    job_id = str(uuid.uuid4())

    try:
        # 1. Read and parse stations.json
        json_content = await stations_json.read()
        try:
            json_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON in stations file: {e}",
            )

        try:
            stations = parse_stations_json(json_data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # 2. Validate we have files
        if not files or len(files) == 0:
            raise HTTPException(
                status_code=400,
                detail="No image files uploaded",
            )

        # 3. Save all uploaded images to temp files
        uploaded_filenames = []
        image_paths: Dict[str, Path] = {}  # basename -> temp path

        for upload_file in files:
            if not upload_file.filename:
                continue
            basename = Path(upload_file.filename).name
            suffix = Path(basename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp_input:
                shutil.copyfileobj(upload_file.file, tmp_input)
                image_paths[basename] = Path(tmp_input.name)
            uploaded_filenames.append(basename)

        # 4. Match images to stations
        try:
            matched = match_stations_to_files(stations, uploaded_filenames)
        except ValueError as e:
            # Cleanup saved files on match failure
            for path in image_paths.values():
                if path.exists():
                    os.remove(path)
            raise HTTPException(status_code=400, detail=str(e))

        # Build the ordered stations data with matched paths
        matched_stations = [station for station, _ in matched]
        # Remap image_paths to use the station basenames
        matched_image_paths: Dict[str, Path] = {}
        for station, uploaded_name in matched:
            station_basename = Path(station["path_to_image"]).name
            matched_image_paths[station_basename] = image_paths[uploaded_name]

        LOGGER.info(
            f"Multi-station job {job_id}: Matched {len(matched)} stations "
            f"to uploaded images"
        )

        # 5. Initialize job
        JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": f"Queued (multi-station: {len(matched)} stations)...",
            "result_path": None,
            "total_stations": len(matched),
            "current_station": 0,
        }

        # 6. Derive a job name from the first station
        job_name = "scene"

        # 7. Start background task
        background_tasks.add_task(
            process_job_multistation,
            job_id,
            matched_stations,
            matched_image_paths,
            job_name,
        )

        return {
            "job_id": job_id,
            "stations_matched": len(matched),
            "station_names": [s.get("name", s["id"]) for s, _ in matched],
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Error submitting multi-station job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit multi-station job: {str(e)}",
        )


@app.post("/predict")
async def submit_predict_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not gaussian_predictor:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    job_id = str(uuid.uuid4())
    
    try:
        # Save input file
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
            shutil.copyfileobj(file.file, tmp_input)
            tmp_input_path = Path(tmp_input.name)

        # Initialize Job
        JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Queued...",
            "result_path": None
        }
        
        # Start Background Task
        background_tasks.add_task(process_job, job_id, tmp_input_path, file.filename)
        
        return {"job_id": job_id}
    except Exception as e:
        LOGGER.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")

@app.post("/predict360")
async def submit_predict360_job(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    """Submit a 360 equirectangular image for Gaussian prediction.

    Extracts 6 cubemap faces, runs SHARP on each, transforms and merges
    the Gaussians into a single scene.
    """
    if not gaussian_predictor:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please check server logs."
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    job_id = str(uuid.uuid4())

    try:
        # Save input file
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
            shutil.copyfileobj(file.file, tmp_input)
            tmp_input_path = Path(tmp_input.name)

        # Initialize Job
        JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Queued (360 panorama)...",
            "result_path": None,
        }

        # Start Background Task
        background_tasks.add_task(
            process_job_360, job_id, tmp_input_path, file.filename
        )

        return {"job_id": job_id}
    except Exception as e:
        LOGGER.error(f"Error submitting 360 job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to submit 360 job: {str(e)}"
        )


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]

@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    if job["status"] != "complete" or not job["result_path"]:
        raise HTTPException(status_code=400, detail="Job not complete")
        
    return FileResponse(
        job["result_path"], 
        media_type="application/octet-stream", 
        filename=job["filename"]
    )
