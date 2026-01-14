"""WebUI for ml-sharp 3D Gaussian Splat prediction.

A simple Flask-based web interface for uploading images and generating 3DGS PLY files.
"""

from __future__ import annotations

import logging
import os
import sys
import subprocess
import tempfile
import uuid
import shutil
import threading
import time
import gc
from pathlib import Path
import urllib.parse
import traceback

# --- Environment Auto-Fix Start ---
# Check for Windows + CPU-only PyTorch and attempt to fix
try:
    import torch
    if os.name == 'nt' and not torch.cuda.is_available():
        if '+cpu' in torch.__version__:
            print("!" * 80)
            print(f"WARNING: Detected CPU-only PyTorch ({torch.__version__}) on Windows.")
            print("Attempting to automatically reinstall CUDA-enabled PyTorch...")
            print("!" * 80)
            
            try:
                # Uninstall existing torch packages
                subprocess.check_call([
                    sys.executable, "-m", "pip", "uninstall", "-y", 
                    "torch", "torchvision", "torchaudio"
                ])
                
                # Install GPU versions (using stable CUDA 12.1 index for best gsplat compatibility)
                # Note: Using --no-cache-dir to avoid picking up the cached CPU wheel again
                print("Installing CUDA-enabled PyTorch (cu121)... (This may take a while)")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                    "--no-cache-dir"
                ])
                
                print("\n" + "=" * 80)
                print("SUCCESS: PyTorch reinstalled with CUDA support.")
                print("Please RESTART this application now to use the GPU.")
                print("=" * 80 + "\n")
                sys.exit(0)
                
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Failed to auto-install PyTorch: {e}")
                print("Please manually run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                # Continue anyway, falling back to CPU logic below
except ImportError:
    pass
# --- Environment Auto-Fix End ---

import numpy as np
import imageio.v2 as iio
import imageio_ffmpeg
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_file

from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians
# Imports for SBS Rendering
from sharp.utils import gsplat, camera

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
LOGGER = logging.getLogger(__name__)

# SILENCE WERKZEUG (HTTP LOGS)
try:
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
except Exception:
    pass

# Flask app - use absolute paths for static and template folders
_base_dir = Path(__file__).parent.absolute()
app = Flask(
    __name__,
    static_folder=str(_base_dir / "webui_static"),
    static_url_path="/static",
    template_folder=str(_base_dir / "webui_templates")
)

# Global model cache
_model_cache = {"predictor": None, "device": None}

# Global job store for async video processing
_active_jobs = {}

# Output directory for generated PLY files
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model URL
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


def get_device() -> torch.device:
    """Get the best available device."""
    if _model_cache["device"] is not None:
        return _model_cache["device"]

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_predictor() -> tuple[RGBGaussianPredictor, torch.device]:
    """Get or create the Gaussian predictor model."""
    if _model_cache["predictor"] is None:
        target_device = torch.device("cpu")
        
        if torch.cuda.is_available():
            target_device = torch.device("cuda")
            try:
                gpu_name = torch.cuda.get_device_name(0)
                LOGGER.info(f"CUDA GPU detected: {gpu_name}")
            except Exception:
                LOGGER.info("CUDA GPU detected (name unknown)")
        elif torch.mps.is_available():
            target_device = torch.device("mps")
            LOGGER.info("Apple MPS acceleration detected.")
        else:
            LOGGER.info("No active GPU detected. Using CPU.")

        LOGGER.info(f"Targeting device for inference: {target_device}")

        LOGGER.info(f"Downloading model from {DEFAULT_MODEL_URL}")
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                DEFAULT_MODEL_URL, 
                progress=True, 
                map_location="cpu"
            )
        except Exception as e:
            LOGGER.error(f"Failed to download/load model checkpoint: {e}")
            raise

        LOGGER.info("Initializing predictor...")
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        
        final_device = torch.device("cpu")
        if target_device.type != "cpu":
            try:
                LOGGER.info(f"Moving model to {target_device}...")
                predictor.to(target_device)
                dummy = torch.zeros(1).to(target_device)
                del dummy
                final_device = target_device
            except RuntimeError as e:
                LOGGER.warning(f"Failed to initialize on {target_device}: {e}.")
                LOGGER.warning("Falling back to CPU mode.")
                predictor.to("cpu")
                final_device = torch.device("cpu")
        else:
            predictor.to("cpu")

        _model_cache["predictor"] = predictor
        _model_cache["device"] = final_device
        LOGGER.info(f"Model successfully loaded and running on: {final_device}")

    return _model_cache["predictor"], _model_cache["device"]


def get_next_job_prefix() -> str:
    """Scan output dir and return the next available 3-digit prefix (e.g. '005_')."""
    try:
        max_idx = 0
        for item in OUTPUT_DIR.iterdir():
            if item.name[:3].isdigit() and item.name[3] == '_':
                try:
                    idx = int(item.name[:3])
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    continue
        return f"{max_idx + 1:03d}"
    except Exception as e:
        LOGGER.error(f"Error calculating job prefix: {e}")
        return "000"


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    use_fp16: bool = False
) -> Gaussians3D:
    """Predict Gaussians from a single image."""
    # Wrap single image in list and use batch predictor
    return predict_batch(predictor, [image], f_px, device, use_fp16)[0]


@torch.no_grad()
def predict_batch(
    predictor: RGBGaussianPredictor,
    images: list[np.ndarray],
    f_px: float,
    device: torch.device,
    use_fp16: bool = False
) -> list[Gaussians3D]:
    """Predict Gaussians from a batch of images."""
    if not images:
        return []

    internal_shape = (1536, 1536)
    
    # Prepare batch tensors
    # Stack numpy arrays: (B, H, W, C) -> permute to (B, C, H, W)
    batch_np = np.stack(images)
    batch_pt = torch.from_numpy(batch_np).float().to(device).permute(0, 3, 1, 2) / 255.0
    
    batch_size, _, height, width = batch_pt.shape
    
    # Disparity factor: (B,)
    disparity_val = f_px / width
    disparity_factor = torch.full((batch_size,), disparity_val, device=device, dtype=torch.float32)

    # Resize batch
    batch_resized_pt = F.interpolate(
        batch_pt,
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Inference
    if use_fp16 and device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            gaussians_ndc = predictor(batch_resized_pt, disparity_factor)
    else:
        gaussians_ndc = predictor(batch_resized_pt, disparity_factor)

    # Post-processing intrinsics
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Unproject whole batch
    gaussians_batch = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    # Split batched Gaussians3D into list of individual Gaussians3D
    results = []
    for i in range(batch_size):
        results.append(Gaussians3D(
            mean_vectors=gaussians_batch.mean_vectors[i:i+1],
            singular_values=gaussians_batch.singular_values[i:i+1],
            quaternions=gaussians_batch.quaternions[i:i+1],
            colors=gaussians_batch.colors[i:i+1],
            opacities=gaussians_batch.opacities[i:i+1]
        ))
        
    return results


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/test")
def test_viewer():
    """Serve the test viewer page."""
    return render_template("test-viewer.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Generate a 3DGS PLY file from an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    quality = request.form.get('quality', 'balanced')
    use_fp16 = (quality == 'fast')

    allowed_extensions = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".tiff", ".tif", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        unique_id = str(uuid.uuid4())[:8]
        original_stem = Path(file.filename).stem

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        LOGGER.info(f"Processing uploaded file: {file.filename} | Quality: {quality} | FP16: {use_fp16}")

        image, _, f_px = io.load_rgb(tmp_path)
        height, width = image.shape[:2]

        predictor, device = get_predictor()
        gaussians = predict_image(predictor, image, f_px, device, use_fp16=use_fp16)

        output_filename = f"{original_stem}_{unique_id}.ply"
        output_path = OUTPUT_DIR / output_filename
        save_ply(gaussians, f_px, (height, width), output_path)

        LOGGER.info(f"Saved PLY to: {output_path}")
        tmp_path.unlink()

        return jsonify({
            "success": True,
            "filename": output_filename,
            "download_url": f"/download/{output_filename}",
            "view_url": f"/ply/{output_filename}",
        })

    except Exception as e:
        LOGGER.exception("Error during generation")
        return jsonify({"error": str(e)}), 500


def _process_video_job(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16, batch_size):
    """Background worker to process video frames into PLY sequence."""
    reader = None
    try:
        total_frames = 0
        try:
            tmp_reader = iio.get_reader(tmp_path)
            total_frames = tmp_reader.count_frames()
            tmp_reader.close()
            _active_jobs[job_id]['total_frames'] = total_frames
        except Exception:
            _active_jobs[job_id]['total_frames'] = 0

        reader = iio.get_reader(tmp_path)
        
        # Determine Folder Name
        job_prefix = get_next_job_prefix()
        # Folder for PLYs: output/001_video_name_plys/
        work_dir = OUTPUT_DIR / f"{job_prefix}_{original_stem}_plys"
        work_dir.mkdir(exist_ok=True)
        
        LOGGER.info(f"Job {job_id} [{job_prefix}] | Batch: {batch_size} | Folder: {work_dir}")

        # Batch accumulation
        current_batch_frames = []
        current_batch_indices = []
        
        frames_iterator = enumerate(reader)
        
        # Helper to process what's in buffer
        def process_batch(frames, indices):
            try:
                # Preprocess: just ensure 3 channels
                processed_frames = []
                for f in frames:
                    if f.shape[2] > 3:
                        processed_frames.append(f[:, :, :3])
                    else:
                        processed_frames.append(f)
                
                # Assume all frames in video are same size
                h, w = processed_frames[0].shape[:2]
                f_px = io.convert_focallength(w, h, 30.0)
                
                # Predict
                gaussians_list = predict_batch(predictor, processed_frames, f_px, device, use_fp16=use_fp16)
                
                # Save Individual PLYs
                for k, g in enumerate(gaussians_list):
                    frame_idx = indices[k]
                    frame_filename = f"{original_stem}_{unique_id}_f{frame_idx:04d}.ply"
                    
                    # Save to subfolder
                    output_path = work_dir / frame_filename
                    save_ply(g, f_px, (h, w), output_path)
                    
                    # Append file path relative to OUTPUT_DIR so webui works
                    # ex: "001_myvideo_plys/myvideo_uuid_f0001.ply"
                    relative_path = f"{work_dir.name}/{frame_filename}"
                    
                    _active_jobs[job_id]['files'].append(relative_path)
                    _active_jobs[job_id]['processed_frames'] = frame_idx + 1
                    
                    # Cleanup
                    del g

                # Clear cache periodically
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    try:
                        torch.mps.empty_cache() 
                    except Exception: 
                        pass
                gc.collect()
                
            except RuntimeError as e:
                # OOM Fallback logic
                if "out of memory" in str(e).lower() and len(frames) > 1:
                    LOGGER.warning(f"OOM detected with batch size {len(frames)}. Switching to batch size 1.")
                    torch.cuda.empty_cache()
                    # Recursive retry one by one
                    for idx, single_frame in enumerate(frames):
                        process_batch([single_frame], [indices[idx]])
                    return True # Signal that OOM happened
                else:
                    raise e
            return False # No OOM

        # Main Loop
        fallback_mode = False
        
        for i, frame in frames_iterator:
            if _active_jobs[job_id]['stop_signal']:
                LOGGER.info(f"Job {job_id} stopped by user.")
                _active_jobs[job_id]['status'] = 'stopped'
                break

            current_batch_frames.append(frame)
            current_batch_indices.append(i)
            
            # If batch full or force fallback
            effective_bs = 1 if fallback_mode else batch_size
            
            if len(current_batch_frames) >= effective_bs:
                oom_occurred = process_batch(current_batch_frames, current_batch_indices)
                if oom_occurred:
                    fallback_mode = True # Permamently switch to 1 for this job
                current_batch_frames = []
                current_batch_indices = []
                
            if i % 10 == 0:
                LOGGER.info(f"Job {job_id}: Processed frame {i+1} / {total_frames}")

        # Process remaining
        if current_batch_frames and not _active_jobs[job_id]['stop_signal']:
            process_batch(current_batch_frames, current_batch_indices)

        if not _active_jobs[job_id]['stop_signal']:
            _active_jobs[job_id]['status'] = 'done'

    except Exception as e:
        LOGGER.exception(f"Job {job_id} failed")
        _active_jobs[job_id]['status'] = 'error'
        _active_jobs[job_id]['error_msg'] = str(e)
    finally:
        if reader:
            try:
                reader.close()
            except:
                pass
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except:
                pass


def _process_sbs_video_job(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16, opacity_threshold, stereo_offset, brightness_factor, batch_size, frame_skip=1):
    """Background worker for SBS 3D Movie generation with Batching and unique folders.
    
    Args:
        frame_skip: Process every Nth frame (1=all frames, 2=every 2nd, etc.)
    """
    
    # Verify CUDA for rendering
    if device.type != 'cuda':
        _active_jobs[job_id]['status'] = 'error'
        _active_jobs[job_id]['error_msg'] = "SBS Rendering requires a CUDA GPU. CPU/MPS not supported for server-side rendering."
        return
    
    # DETERMINE FOLDER NAMES
    job_prefix = get_next_job_prefix()
    
    # Folder for frames: output/001_video_name_frames/
    work_dir = OUTPUT_DIR / f"{job_prefix}_{original_stem}_frames"
    work_dir.mkdir(exist_ok=True)
    
    audio_path = work_dir / "audio.aac"
    
    LOGGER.info(f"Job {job_id} [{job_prefix}] | Batch: {batch_size} | Folder: {work_dir}")
    
    reader = None
    
    try:
        # 1. Extract Audio
        LOGGER.info(f"Job {job_id}: Extracting audio...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        audio_cmd = [
            ffmpeg_exe, "-y", "-i", str(tmp_path), 
            "-vn", "-acodec", "copy", str(audio_path)
        ]
        has_audio = False
        try:
            subprocess.run(audio_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if audio_path.exists() and audio_path.stat().st_size > 0:
                has_audio = True
        except subprocess.CalledProcessError:
            LOGGER.warning(f"Job {job_id}: No audio track found or extraction failed.")

        # 2. Setup Processing
        try:
            reader = iio.get_reader(tmp_path)
            total_frames_raw = reader.count_frames()
            # Adjust total_frames to reflect actual frames to render after skip
            total_frames = (total_frames_raw + frame_skip - 1) // frame_skip  # Ceiling division
            _active_jobs[job_id]['total_frames'] = total_frames
            LOGGER.info(f"Job {job_id}: {total_frames_raw} total frames, {total_frames} to render (skip={frame_skip})")
        except Exception:
            _active_jobs[job_id]['total_frames'] = 0
            reader = iio.get_reader(tmp_path)

        # Initialize Renderer
        render_w, render_h = 1920, 1080
        try:
            renderer = gsplat.GSplatRenderer(color_space="linearRGB")
        except Exception as e:
            if "cl" in str(e) or "DLL" in str(e):
                raise RuntimeError("gsplat compilation failed. Install VS Build Tools.")
            raise e
            
        png_files = [] # Tuples of (index, path) to ensure sort order if needed, but append is sequential here
        
        # Batch Containers
        batch_frames = []
        batch_indices = []
        
        # Performance: Pre-compute stereo camera extrinsics (used for all frames)
        _cached_ext_left = torch.eye(4, device=device)
        _cached_ext_left[0, 3] = stereo_offset
        _cached_ext_right = torch.eye(4, device=device)
        _cached_ext_right[0, 3] = -stereo_offset
        # Cache intrinsics template (f_px will vary per frame size, but we cache the structure)
        _cached_intrinsics_template = None
        _cached_frame_dims = None

        def process_sbs_batch(frames, indices):
            nonlocal _cached_intrinsics_template, _cached_frame_dims
            try:
                # Preprocess
                clean_frames = []
                for f in frames:
                    if f.shape[2] > 3: clean_frames.append(f[:, :, :3])
                    else: clean_frames.append(f)
                
                h, w = clean_frames[0].shape[:2]
                f_px = io.convert_focallength(w, h, 30.0)
                
                # Predict Batch
                gaussians_list = predict_batch(predictor, clean_frames, f_px, device, use_fp16=use_fp16)
                
                # Performance: Cache intrinsics if frame dimensions match
                if _cached_frame_dims != (h, w):
                    f_px_render = f_px * (render_w / w)
                    _cached_intrinsics_template = torch.tensor([
                        [f_px_render, 0, render_w / 2, 0],
                        [0, f_px_render, render_h / 2, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ], device=device, dtype=torch.float32)
                    _cached_frame_dims = (h, w)
                
                intrinsics = _cached_intrinsics_template
                ext_left = _cached_ext_left
                ext_right = _cached_ext_right
                
                # Render Loop (Rendering is still sequential per gaussian, but prediction was batched)
                # Note: We could technically batch render if gsplat supports it, but gsplat renderer
                # usually takes one scene at a time. The speedup comes from the UNet prediction.
                
                for k, gaussians in enumerate(gaussians_list):
                    idx = indices[k]
                    
                    # Halo Removal
                    if opacity_threshold > 0.0:
                        mask = gaussians.opacities[0] > opacity_threshold
                        gaussians = Gaussians3D(
                            mean_vectors=gaussians.mean_vectors[:, mask],
                            singular_values=gaussians.singular_values[:, mask],
                            quaternions=gaussians.quaternions[:, mask],
                            colors=gaussians.colors[:, mask],
                            opacities=gaussians.opacities[:, mask]
                        )
                    
                    # Setup Cam
                    scale_x = render_w / w
                    # f_px_render = f_px * scale_x # Standard logic
                    f_px_render = f_px * (render_w / w)

                    intrinsics = torch.tensor([
                        [f_px_render, 0, render_w / 2, 0],
                        [0, f_px_render, render_h / 2, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ], device=device, dtype=torch.float32)

                    ext_left = torch.eye(4, device=device)
                    ext_left[0, 3] = stereo_offset
                    
                    ext_right = torch.eye(4, device=device)
                    ext_right[0, 3] = -stereo_offset 
                    
                    # Render Left
                    out_left = renderer(
                        gaussians, 
                        extrinsics=ext_left.unsqueeze(0), 
                        intrinsics=intrinsics.unsqueeze(0),
                        image_width=render_w, image_height=render_h
                    )
                    img_left_tensor = (out_left.color[0] * brightness_factor).clamp(0, 1)
                    img_left = (img_left_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()

                    # Render Right
                    out_right = renderer(
                        gaussians, 
                        extrinsics=ext_right.unsqueeze(0), 
                        intrinsics=intrinsics.unsqueeze(0),
                        image_width=render_w, image_height=render_h
                    )
                    img_right_tensor = (out_right.color[0] * brightness_factor).clamp(0, 1)
                    img_right = (img_right_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()

                    # Stitch
                    img_sbs = np.concatenate((img_left, img_right), axis=1)
                    
                    frame_name = f"frame_{idx:05d}.png"
                    frame_path = work_dir / frame_name
                    Image.fromarray(img_sbs).save(frame_path)
                    png_files.append(frame_path)

                    _active_jobs[job_id]['processed_frames'] = idx + 1
                    
                    del gaussians, out_left, out_right, img_left_tensor, img_right_tensor, img_left, img_right, img_sbs
                
                # Cleanup
                torch.cuda.empty_cache()
                gc.collect()

            except RuntimeError as e:
                # Fallback logic
                if "out of memory" in str(e).lower() and len(frames) > 1:
                    LOGGER.warning(f"OOM in SBS job with batch {len(frames)}. Switching to 1.")
                    torch.cuda.empty_cache()
                    for i in range(len(frames)):
                        process_sbs_batch([frames[i]], [indices[i]])
                    return True
                else:
                    raise e
            return False

        # 3. Frame Loop
        fallback_mode = False
        output_frame_idx = 0  # Separate counter for output frame numbering

        
        for i, frame in enumerate(reader):
            if _active_jobs[job_id]['stop_signal']:
                LOGGER.info(f"Job {job_id} stopped by user.")
                _active_jobs[job_id]['status'] = 'stopped'
                break
            
            # Frame skip logic: only process every Nth frame
            if i % frame_skip != 0:
                continue

            batch_frames.append(frame)
            batch_indices.append(output_frame_idx)  # Use output index for sequential frame naming
            output_frame_idx += 1
            
            effective_bs = 1 if fallback_mode else batch_size
            
            if len(batch_frames) >= effective_bs:
                oom = process_sbs_batch(batch_frames, batch_indices)
                if oom: fallback_mode = True
                batch_frames = []
                batch_indices = []
            
            if i % 10 == 0:
                 LOGGER.info(f"Job {job_id}: SBS Rendered frame {i+1}")
        
        # Remainder
        if batch_frames:
            # Process remainder even if stopped, to save progress
            process_sbs_batch(batch_frames, batch_indices)

        # 4. Final Assembly (Allowed even if stopped)
        if png_files:
            # Save Video to root OUTPUT_DIR with prefix
            output_filename = f"{job_prefix}_{original_stem}_SBS.mp4"
            output_path = OUTPUT_DIR / output_filename
            LOGGER.info(f"Job {job_id}: Encoding SBS video to {output_filename}...")
            
            # Adjust output FPS based on frame_skip to maintain original video duration
            output_fps = fps / frame_skip
            LOGGER.info(f"Job {job_id}: Output FPS adjusted from {fps} to {output_fps} (frame_skip={frame_skip})")
            
            input_pattern = str(work_dir / "frame_%05d.png")
            cmd = [ffmpeg_exe, "-y", "-framerate", str(output_fps), "-i", input_pattern]
            if has_audio: cmd.extend(["-i", str(audio_path)])
            cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast"])
            
            # If audio exists, shortest cuts it to video length (useful if we stopped early)
            if has_audio: cmd.append("-shortest")
            
            cmd.append(str(output_path))
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                _active_jobs[job_id]['files'].append(output_filename)
                
                # If we stopped, we still finished encoding the partial video.
                # Do we mark job as 'done' so the UI shows success? 
                # Yes, because the file is ready to view.
                _active_jobs[job_id]['status'] = 'done'
            except subprocess.CalledProcessError as e:
                LOGGER.error(f"FFmpeg encoding failed: {e.stderr.decode()}")
                _active_jobs[job_id]['status'] = 'error'
                _active_jobs[job_id]['error_msg'] = "Video encoding failed."
        elif not _active_jobs[job_id]['stop_signal']:
             _active_jobs[job_id]['status'] = 'error'
             _active_jobs[job_id]['error_msg'] = "No frames were processed."

    except Exception as e:
        LOGGER.exception(f"Job {job_id} failed")
        _active_jobs[job_id]['status'] = 'error'
        _active_jobs[job_id]['error_msg'] = str(e)
    finally:
        if reader:
            try:
                reader.close()
            except:
                pass
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except:
                pass
        # NOTE: rmtree removed per user request to keep frame files
        # if work_dir.exists(): try: shutil.rmtree(work_dir) except: pass


@app.route("/preview_sbs_frame", methods=["POST"])
def preview_sbs_frame():
    """Generate a single SBS preview frame from a video for testing settings."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Parse parameters
    try:
        frame_number = int(request.form.get('frame_number', 0))
    except ValueError:
        frame_number = 0
    
    quality = request.form.get('quality', 'balanced')
    use_fp16 = (quality == 'fast')
    opacity_threshold = float(request.form.get('opacity_threshold', 0.0))
    stereo_offset = float(request.form.get('stereo_offset', 0.015))
    brightness_factor = float(request.form.get('brightness_factor', 1.0))

    LOGGER.info(f"SBS Preview: frame={frame_number}, opacity={opacity_threshold}, offset={stereo_offset}, brightness={brightness_factor}")

    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    tmp_path = None
    try:
        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        # Get predictor and device
        predictor, device = get_predictor()
        
        # Verify CUDA for rendering
        if device.type != 'cuda':
            return jsonify({"error": "SBS Preview requires a CUDA GPU."}), 400

        # Open video and extract frame
        reader = iio.get_reader(tmp_path)
        try:
            frame = reader.get_data(frame_number)
        except IndexError:
            reader.close()
            return jsonify({"error": f"Frame {frame_number} out of range"}), 400
        reader.close()

        # Ensure 3 channels
        if frame.shape[2] > 3:
            frame = frame[:, :, :3]

        h, w = frame.shape[:2]
        f_px = io.convert_focallength(w, h, 30.0)

        # Predict Gaussians
        gaussians = predict_image(predictor, frame, f_px, device, use_fp16=use_fp16)

        # Apply halo removal
        if opacity_threshold > 0.0:
            mask = gaussians.opacities[0] > opacity_threshold
            gaussians = Gaussians3D(
                mean_vectors=gaussians.mean_vectors[:, mask],
                singular_values=gaussians.singular_values[:, mask],
                quaternions=gaussians.quaternions[:, mask],
                colors=gaussians.colors[:, mask],
                opacities=gaussians.opacities[:, mask]
            )

        # Setup rendering
        render_w, render_h = 1920, 1080
        renderer = gsplat.GSplatRenderer(color_space="linearRGB")

        f_px_render = f_px * (render_w / w)
        intrinsics = torch.tensor([
            [f_px_render, 0, render_w / 2, 0],
            [0, f_px_render, render_h / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], device=device, dtype=torch.float32)

        ext_left = torch.eye(4, device=device)
        ext_left[0, 3] = stereo_offset

        ext_right = torch.eye(4, device=device)
        ext_right[0, 3] = -stereo_offset

        # Render left eye
        out_left = renderer(
            gaussians,
            extrinsics=ext_left.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            image_width=render_w, image_height=render_h
        )
        img_left_tensor = (out_left.color[0] * brightness_factor).clamp(0, 1)
        img_left = (img_left_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()

        # Render right eye
        out_right = renderer(
            gaussians,
            extrinsics=ext_right.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            image_width=render_w, image_height=render_h
        )
        img_right_tensor = (out_right.color[0] * brightness_factor).clamp(0, 1)
        img_right = (img_right_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()

        # Stitch side-by-side (3840x1080)
        img_sbs = np.concatenate((img_left, img_right), axis=1)

        # Cleanup GPU memory
        del gaussians, out_left, out_right, img_left_tensor, img_right_tensor
        torch.cuda.empty_cache()
        gc.collect()

        # Save to temp file and return
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as out_tmp:
            Image.fromarray(img_sbs).save(out_tmp.name, "JPEG", quality=90)
            out_tmp_path = Path(out_tmp.name)

        response = send_file(
            out_tmp_path,
            mimetype='image/jpeg',
            as_attachment=False
        )
        
        # Schedule cleanup after response (Flask handles this)
        @response.call_on_close
        def cleanup():
            try:
                out_tmp_path.unlink()
            except:
                pass

        return response

    except Exception as e:
        LOGGER.exception("Error generating SBS preview")
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except:
                pass


@app.route("/generate_video", methods=["POST"])
def generate_video():
    """Start async video generation."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Settings
    quality = request.form.get('quality', 'balanced')
    use_fp16 = (quality == 'fast')
    output_mode = request.form.get('output_mode', 'ply_seq')
    opacity_threshold = float(request.form.get('opacity_threshold', 0.0))
    stereo_offset = float(request.form.get('stereo_offset', 0.015))
    brightness_factor = float(request.form.get('brightness_factor', 1.0))
    
    # BATCH SIZE parsing
    try:
        batch_size = int(request.form.get('batch_size', 1))
        if batch_size < 1: batch_size = 1
    except:
        batch_size = 1
    
    # FRAME SKIP parsing (1 = process all frames, 2 = every 2nd frame, etc.)
    try:
        frame_skip = int(request.form.get('frame_skip', 1))
        if frame_skip < 1: frame_skip = 1
    except:
        frame_skip = 1

    LOGGER.info(f"Starting video generation | Mode: {output_mode} | Batch Size: {batch_size} | Frame Skip: {frame_skip} | Quality: {quality}")

    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        unique_id = str(uuid.uuid4())[:8]
        original_stem = Path(file.filename).stem
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            reader = iio.get_reader(tmp_path)
            meta = reader.get_meta_data()
            fps = meta.get('fps', 30.0)
            reader.close() 
        except Exception:
            fps = 30.0
        
        job_id = str(uuid.uuid4())
        _active_jobs[job_id] = {
            "status": "running",
            "files": [],
            "total_frames": 0,
            "processed_frames": 0,
            "stop_signal": False,
            "error_msg": "",
            "fps": fps,
            "mode": output_mode
        }

        predictor, device = get_predictor()
        
        if output_mode == 'sbs_movie':
            thread = threading.Thread(
                target=_process_sbs_video_job,
                args=(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16, opacity_threshold, stereo_offset, brightness_factor, batch_size, frame_skip)
            )
        else:
            thread = threading.Thread(
                target=_process_video_job,
                args=(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16, batch_size)
            )

        thread.start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "running"
        })

    except Exception as e:
        LOGGER.exception("Error starting video generation")
        return jsonify({"error": str(e)}), 500


@app.route("/job_status/<job_id>")
def job_status(job_id):
    """Check status of a background job."""
    job = _active_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify({
        "status": job["status"],
        "processed": job["processed_frames"],
        "total": job["total_frames"],
        "files": job["files"], # Returns full list so client can see what's new
        "fps": job["fps"],
        "error": job["error_msg"],
        "mode": job.get("mode", "ply_seq"),
        "base_url": "/download/" if job.get("mode") == "sbs_movie" else "/ply/"
    })


@app.route("/stop_job/<job_id>", methods=["POST"])
def stop_job(job_id):
    """Signal a job to stop."""
    job = _active_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    job["stop_signal"] = True
    return jsonify({"success": True, "status": "stopping"})


@app.route("/scan_local", methods=["POST"])
def scan_local():
    """Scan a local path for PLY files."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        path_str = data.get("path")
        mode = data.get("mode") # 'single' or 'video'
        
        if not path_str:
            return jsonify({"error": "No path provided"}), 400

        # Clean path string (remove quotes users might copy-paste)
        path_str = path_str.strip().strip('"').strip("'")
        path = Path(path_str)
        
        if not path.exists():
            return jsonify({"error": f"Path does not exist: {path_str}"}), 404
            
        if mode == "single":
            if not path.is_file() or path.suffix.lower() != ".ply":
                 return jsonify({"error": "Path is not a valid PLY file"}), 400
            
            # Quote path to safely handle spaces in URL
            encoded_path = urllib.parse.quote(str(path))
            return jsonify({
                "success": True,
                "type": "single",
                "view_url": f"/view_local?path={encoded_path}",
                "filename": path.name
            })
            
        elif mode == "video":
            if not path.is_dir():
                 return jsonify({"error": "Path is not a directory"}), 400
            
            ply_files = sorted([f for f in path.glob("*.ply")])
            if not ply_files:
                 return jsonify({"error": "No PLY files found in directory"}), 400
                 
            # Create full URLs
            file_urls = []
            for p in ply_files:
                 encoded_path = urllib.parse.quote(str(p))
                 file_urls.append(f"/view_local?path={encoded_path}")
            
            return jsonify({
                "success": True,
                "type": "video",
                "fps": 30, # Defaulting to 30 as we can't infer from just files easily
                "frame_count": len(file_urls),
                "ply_files": file_urls,
                "base_url": "" # Empty because urls are fully formed
            })
        
        return jsonify({"error": "Invalid mode"}), 400
        
    except Exception as e:
        LOGGER.exception("Error scanning local path")
        # Return JSON error instead of 500 HTML
        return jsonify({"error": f"Server Error: {str(e)}"}), 500


@app.route("/view_local")
def view_local():
    """Serve a file from a local absolute path."""
    try:
        path_str = request.args.get("path")
        if not path_str:
            return "No path provided", 400
        
        # Check if path exists before sending
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            return "File not found", 404

        return send_file(path, mimetype="application/octet-stream")
    except Exception as e:
        LOGGER.exception("Error serving local file")
        return str(e), 500


@app.route("/download/<path:filename>")
def download(filename: str):
    """Download a generated file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    # Determine mimetype based on extension
    mime = "application/octet-stream"
    if filename.endswith(".mp4"):
        mime = "video/mp4"

    return send_file(
        file_path,
        as_attachment=True,
        download_name=Path(filename).name,
        mimetype=mime,
    )


@app.route("/ply/<path:filename>")
def serve_ply(filename: str):
    """Serve a PLY file for the viewer."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(
        file_path,
        mimetype="application/octet-stream",
    )


@app.route("/status")
def status():
    """Get server status."""
    device = get_device()
    model_loaded = _model_cache["predictor"] is not None
    return jsonify({
        "status": "ok",
        "device": str(device),
        "model_loaded": model_loaded,
        "cuda_available": torch.cuda.is_available(),
    })

@app.route("/open_output_folder", methods=["POST"])
def open_output_folder():
    """Opens the 'output' directory in the system file explorer."""
    try:
        # Get the absolute path to the 'output' folder in the app root
        path = str(OUTPUT_DIR.absolute())
        
        if os.name == 'nt':  # Windows
            os.startfile(path)
        elif sys.platform == 'darwin':  # macOS
            subprocess.Popen(['open', path])
        else:  # Linux
            subprocess.Popen(['xdg-open', path])
            
        return jsonify({"success": True})
    except Exception as e:
        LOGGER.error(f"Failed to open folder: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ml-sharp WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--preload", action="store_true", help="Preload model on startup")

    args = parser.parse_args()

    if args.preload:
        LOGGER.info("Preloading model...")
        get_predictor()

    LOGGER.info(f"Starting WebUI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)