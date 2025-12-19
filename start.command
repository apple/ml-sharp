#!/bin/bash
# Sharp Web Interface - One-Click Launcher
# Double-click this file to start the Sharp web interface

# Change to the script's directory
cd "$(dirname "$0")"

ENV_NAME="sharp"
PYTHON_VERSION="3.13"

echo "======================================"
echo "  Sharp 3D Prediction - Web Interface"
echo "======================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH."
    echo ""
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Check if the environment exists
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "📦 Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create conda environment."
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "✅ Environment created."
    echo ""
fi

# Activate the environment
echo "🔄 Activating conda environment '${ENV_NAME}'..."
conda activate "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment."
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if sharp is installed by trying to import it
echo "🔍 Checking if dependencies are installed..."
if ! python -c "import sharp" 2>/dev/null; then
    echo "📦 Installing project dependencies (this may take a few minutes)..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install requirements."
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "✅ Dependencies installed."
    echo ""
fi

# Check if web dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing web interface dependencies..."
    pip install -r src/sharp/web/requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install web requirements."
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "✅ Web dependencies installed."
    echo ""
fi

echo "======================================"
echo "🚀 Starting Sharp Web Interface..."
echo "======================================"
echo ""
echo "Open your browser and go to:"
echo ""
echo "  👉  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Start the web server
python src/sharp/web/app.py

# Keep terminal open if server stops unexpectedly
read -p "Press Enter to exit..."
