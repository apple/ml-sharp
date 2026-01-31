#!/bin/bash
echo "Starting ml-sharp WebUI..."
echo
echo "Installing Flask if needed..."
pip install flask -q
echo
echo "Starting server at http://127.0.0.1:7860"
echo "Press Ctrl+C to stop the server"
echo
python webui.py --preload
