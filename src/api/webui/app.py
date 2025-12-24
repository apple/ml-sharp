"""Gradio Web UI for ml-sharp API.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
import time
import requests
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import gradio as gr
from PIL import Image

LOGGER = logging.getLogger(__name__)

# API endpoint URL - update this if the API is running on a different host/port
API_URL = "http://localhost:8000"


def predict_batch(
    images: List[str],
    device: str,
    render: bool,
    checkpoint_path: Optional[str] = None
) -> Tuple[str, str, str]:
    """Predict 3D Gaussians from multiple images using the API.

    Args:
        images: List of input image file paths.
        device: The device to run inference on.
        render: Whether to render the results.
        checkpoint_path: Path to a custom model checkpoint.

    Returns:
        Tuple containing:
        - Task ID for tracking progress
        - Status message
        - Empty string (placeholder for compatibility)
    """
    try:
        if not images:
            return "", "Error: No images provided", ""

        # Convert images to bytes
        import io
        files = []
        for i, image_path in enumerate(images):
            # Open image file
            img_pil = Image.open(image_path)
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            # Get original filename
            import os
            filename = os.path.basename(image_path)
            files.append(("files", (filename, img_byte_arr, "image/png")))

        # Prepare API request
        data = {
            "device": device,
            "render": render,
            "checkpoint_path": checkpoint_path
        }

        # Send request to API
        response = requests.post(f"{API_URL}/api/predict/batch", files=files, data=data)
        result = response.json()

        if "error" in result:
            return "", f"Error: {result['error']}", ""

        # Return task ID and status
        task_id = result["task_id"]
        return task_id, f"Batch prediction started. Task ID: {task_id}", ""

    except Exception as e:
        LOGGER.exception(f"Batch prediction failed: {str(e)}")
        return "", f"Error: {str(e)}", ""


def check_progress(task_id: str) -> Tuple[float, str, str]:
    """Check the progress of a batch prediction task.

    Args:
        task_id: The ID of the batch prediction task.

    Returns:
        Tuple containing:
        - Progress percentage (0-100)
        - Current status message
        - Estimated time remaining
    """
    try:
        # Send request to API
        response = requests.get(f"{API_URL}/api/predict/batch/{task_id}/status")
        result = response.json()

        # Extract progress data
        progress = result["progress"]
        status = result["status"]
        current_image = result.get("current_image", "")
        processed = result["processed_images"]
        total = result["total_images"]
        estimated_time = result.get("estimated_time_remaining", 0)

        # Format status message
        status_msg = f"Status: {status} | Processing: {current_image} ({processed}/{total})"
        time_remaining = f"Estimated time remaining: {estimated_time:.1f} seconds" if estimated_time else ""

        return progress, status_msg, time_remaining

    except Exception as e:
        LOGGER.exception(f"Failed to check progress: {str(e)}")
        return 0, f"Error checking progress: {str(e)}", ""


def get_results(task_id: str) -> List[Dict]:  # pyright: ignore[reportDeprecated, reportMissingTypeArgument, reportUnknownParameterType]
    """Get the results of a batch prediction task.

    Args:·
        task_id: The ID of the batch prediction task.

    Returns:
        List of result dictionaries containing:
        - filename: Original filename
        - ply_url: URL to download the PLY file
        - video_url: URL to the rendered video (if enabled)
        - inference_time: Inference time for the image
        - num_gaussians: Number of 3D Gaussians generated
    """
    try:
        # Send request to API
        response = requests.get(f"{API_URL}/api/predict/batch/{task_id}/results")
        result = response.json()

        if result["status"] != "completed":
            return [{"error": f"Task not completed. Current status: {result['status']}"}]  # pyright: ignore[reportUnknownVariableType]

        # Extract results
        batch_results = result["result"]["results"]
        formatted_results = []

        for res in batch_results:
            formatted_results.append({
                "filename": res["filename"],
                "ply_url": f"{API_URL}{res['gaussians_ply_path']}",
                "video_url": f"{API_URL}{res['render_video_path']}" if res['render_video_path'] else None,
                "inference_time": res["inference_time"],
                "num_gaussians": res["num_gaussians"]
            })

        return formatted_results

    except Exception as e:
        LOGGER.exception(f"Failed to get results: {str(e)}")
        return [{"error": f"Error getting results: {str(e)}"}]


def create_gradio_app() -> gr.Blocks:
    """Create and configure the Gradio application.

    Returns:
        The configured Gradio Blocks instance.
    """
    with gr.Blocks(title="ml-sharp 3D Gaussian Predictor") as app:
        # Header
        gr.Markdown("""# ml-sharp 3D Gaussian Point Cloud Predictor

Generate 3D Gaussian point clouds from 2D images using the ml-sharp model.
        """)

        # State variables
        task_id_state = gr.State("")
        results_state = gr.State([])

        # Input and Output sections - Now below Results section
        with gr.Row():
            with gr.Column(scale=1):
                # Image list input
                image_list_input = gr.Files(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=["image"],
                    height=300
                )

                # Device selection
                device_choices = ["default", "cpu", "cuda", "mps"]
                device_dropdown = gr.Dropdown(
                    choices=device_choices,
                    value="default",
                    label="Device",
                    info="Select the device to run inference on"
                )

                # Render option
                render_checkbox = gr.Checkbox(
                    value=False,
                    label="Render to Video",
                    info="Generate a video rendering of the 3D Gaussians (CUDA only)"
                )

                # Checkpoint path (optional)
                checkpoint_path = gr.Textbox(
                    value="",
                    label="Custom Checkpoint Path (Optional)",
                    placeholder="Path to a custom .pt checkpoint file",
                    info="Leave empty to use the default model"
                )

                # Predict button
                predict_button = gr.Button(
                    "Generate 3D Gaussians",
                    variant="primary",
                    size="lg"
                )

            # Output section
            with gr.Column(scale=2):
                    # Status message
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Ready to process images...",
                    lines=2
                )

                # Progress details
                progress_details = gr.Textbox(
                    label="Progress Details",
                    interactive=False,
                    placeholder="Progress details will be shown here...",
                    lines=3
                )

                # Add empty space to match left column height
                # empty_space = gr.Markdown("", label="")
                
                with gr.Column(scale=1):
                    # Results display with direct download buttons
                    results_display = gr.Markdown(
                        label="结果列表",
                        value="点击“获取结果”按钮查看结果"
                    )

                    # Get results button
                    get_results_button = gr.Button(
                        "获取结果",
                        # variant="secondary",
                        interactive=False
                    )


        # Function to continuously check progress
        def check_progress_continuous(task_id: str) -> Tuple[str, bool]:
            """Check progress continuously until task is completed.
            
            Args:
                task_id: The task ID to check.
                
            Returns:
                Tuple containing status details and whether task is completed.
            """
            try:
                # Send request to API
                response = requests.get(f"{API_URL}/api/predict/batch/{task_id}/status")
                result = response.json()

                # Extract progress data
                status = result["status"]
                current_image = result.get("current_image", "")
                processed = result["processed_images"]
                total = result["total_images"]

                # Format status message
                status_msg = f"Status: {status} | Processing: {current_image} ({processed}/{total})"
                
                # Check if task is completed
                is_completed = status in ["completed", "failed"]
                
                return status_msg, is_completed
            except Exception as e:
                LOGGER.exception(f"Failed to check progress: {str(e)}")
                return f"Error checking progress: {str(e)}", False

        # Set up event handlers
        predict_button.click(  # pyright: ignore[reportUnusedCallResult]
            fn=predict_batch,
            inputs=[image_list_input, device_dropdown, render_checkbox, checkpoint_path],
            outputs=[task_id_state, status_output, progress_details]
        ).then(
            fn=check_progress_continuous,
            inputs=[task_id_state],
            outputs=[progress_details, get_results_button]
        ).then(
            fn=lambda is_completed: gr.Button(interactive=is_completed),
            inputs=[get_results_button],
            outputs=[get_results_button]
        )

        # Get results button click
        get_results_button.click(  # pyright: ignore[reportUnusedCallResult]
            fn=get_results,
            inputs=[task_id_state],
            outputs=[results_state]
        ).then(
            fn=lambda results: 
                # Format results as Markdown table with direct download buttons
                "# Results Summary\n\n" +
                "| Filename | Gaussians | Time (s) | Actions |\n" +
                "|----------|-----------|----------|---------|\n" +
                "\n".join([
                    "| {} | {:,} | {:.2f} | {}{} |".format(
                        res['filename'],
                        res['num_gaussians'],
                        res['inference_time'],
                        "[Download PLY]({})".format(res['ply_url']),
                        " | [Download Video]({})".format(res['video_url']) if res['video_url'] else ""
                    )
                    for res in results
                ]),
            inputs=[results_state],
            outputs=[results_display]
        )

        # Footer
        gr.Markdown("""## 关于  
此应用程序使用 ml-sharp 模型从 2D 图像生成 3D 高斯点云。
该模型基于最先进的计算机视觉技术。
        """) # pyright: ignore[reportUnusedCallResult]

    return app


# Create Gradio app instance
app = create_gradio_app()
