"""Gradio Web UI for ml-sharp API.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import requests
from PIL import Image

LOGGER = logging.getLogger(__name__)

# API endpoint URL - update this if the API is running on a different host/port
API_URL = "http://localhost:8000"


def predict_image(
    image: Image.Image,
    device: str,
    render: bool,
    checkpoint_path: Optional[str] = None
) -> Tuple[str, Optional[str], str, str]:
    """Predict 3D Gaussians from an image using the API.

    Args:
        image: The input image.
        device: The device to run inference on.
        render: Whether to render the result.
        checkpoint_path: Path to a custom model checkpoint.

    Returns:
        Tuple containing:
        - Path to the generated PLY file
        - Path to the rendered video (if enabled)
        - Inference time
        - Status message
    """
    try:
        # Convert image to bytes
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Prepare API request
        files = {"file": ("image.png", img_byte_arr, "image/png")}
        data = {
            "device": device,
            "render": render,
            "checkpoint_path": checkpoint_path
        }

        # Send request to API
        start_time = time.time()
        response = requests.post(f"{API_URL}/api/predict", files=files, data=data)
        response_time = time.time() - start_time

        # Parse response
        result = response.json()

        if not result["success"]:
            return "", None, "", f"Error: {result.get('error', 'Unknown error')}"

        # Get result data
        result_data = result["result"]
        ply_path = result_data["gaussians_ply_path"]
        video_path = result_data.get("render_video_path")
        inference_time = result_data["inference_time"]

        # Format results
        ply_url = f"{API_URL}{ply_path}"
        video_url = f"{API_URL}{video_path}" if video_path else None
        time_str = f"Inference time: {inference_time:.2f} seconds"
        status = f"Success! Generated {result_data['num_gaussians']} 3D Gaussians"

        return ply_url, video_url, time_str, status

    except Exception as e:
        LOGGER.exception(f"Prediction failed: {str(e)}")
        return "", None, "", f"Error: {str(e)}"


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

        # Input section
        with gr.Row():
            with gr.Column(scale=1):
                # Image input
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=300,
                    width=300
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
                    placeholder="Ready to process images..."
                )

                # Inference time
                time_output = gr.Textbox(
                    label="Inference Time",
                    interactive=False
                )

                # PLY file download
                ply_output = gr.Textbox(
                    label="Generated PLY File",
                    interactive=False,
                    placeholder="No PLY file generated yet"
                )

                # PLY download button
                ply_download_button = gr.Button(
                    "Download PLY File",
                    variant="secondary"
                )

                # Video output (if enabled)
                video_output = gr.Video(
                    label="Rendered Video",
                    interactive=False,
                    visible=False
                )

        # Set up event handlers
        predict_button.click(
            fn=predict_image,
            inputs=[image_input, device_dropdown, render_checkbox, checkpoint_path],
            outputs=[ply_output, video_output, time_output, status_output]
        )

        # Update video visibility based on render checkbox
        render_checkbox.change(
            fn=lambda render: gr.Video(visible=render),
            inputs=[render_checkbox],
            outputs=[video_output]
        )

        # Add example images
        gr.Markdown("""## Example Images
        """)
        with gr.Row():
            gr.Examples(
                examples=[
                    "examples/example1.jpg",
                    "examples/example2.jpg",
                    "examples/example3.jpg"
                ],
                inputs=image_input,
                outputs=[ply_output, video_output, time_output, status_output],
                fn=predict_image,
                cache_examples=False
            )

        # Footer
        gr.Markdown("""## About

This application uses the ml-sharp model to generate 3D Gaussian point clouds from 2D images.
The model is based on state-of-the-art computer vision techniques.
        """)

    return app


# Create Gradio app instance
app = create_gradio_app()
