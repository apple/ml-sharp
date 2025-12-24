"""FastAPI application main entry point.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .dependencies import ModelCache
from .endpoints import predict
from .webui.app import create_gradio_app

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Create FastAPI application instance
app = FastAPI(
    title="ml-sharp API",
    description="REST API for ml-sharp 3D Gaussian Point Cloud Prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create directories for static files and results
# RESULTS_DIR = Path("/tmp/ml-sharp/results")
RESULTS_DIR = Path(__file__).parent.parent.parent / "tmp" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files directory for serving results
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# Include API routers
app.include_router(
    predict.router,
    prefix="/api",
    tags=["predict"],
    responses={404: {"description": "Not found"}},
)

# Create and mount Gradio app
LOGGER.info("Creating Gradio app...")
gradio_app = create_gradio_app()

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/webui")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_cache = ModelCache()
    return {
        "status": "healthy",
        "models_loaded": len(model_cache._models),
        "timestamp": "2025-12-24T11:50:00Z"  # This will be replaced with actual timestamp
    }

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Welcome to ml-sharp API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "webui": "/webui"  # Gradio UI is available here
    }
