#!/usr/bin/env python3
"""Startup script for the ml-sharp API server.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
import uvicorn

from api.app import app

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    """Main entry point for the API server."""
    LOGGER.info("Starting ml-sharp API server...")
    LOGGER.info("API will be available at http://localhost:8000")
    LOGGER.info("Web UI will be available at http://localhost:8000/webui")
    LOGGER.info("API documentation will be available at http://localhost:8000/docs")
    
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
