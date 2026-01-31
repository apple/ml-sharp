@echo off
setlocal

:: --- CONFIGURATION ---
set "VENV_DIR=%~dp0venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PIP=%VENV_DIR%\Scripts\pip.exe"

:: --- ENVIRONMENT ISOLATION ---
:: This allows the script to ignore conflicting global packages in AppData
set "PYTHONNOUSERSITE=1"

echo ======================================================================
echo                 ML-SHARP WEBUI LAUNCHER (Windows)
echo ======================================================================
echo.

:: 1. Check for Virtual Environment
if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment not found at:
    echo %VENV_DIR%
    echo.
    echo Please run the installation steps in README.md first!
    pause
    exit /b 1
)

:: 2. Check for Critical Dependencies (Quick Verification)
"%PYTHON%" -c "import torch; import gsplat; print(f'Torch: {torch.__version__} | CUDA: {torch.version.cuda} | GSplat loaded')" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Critical dependencies appear missing or broken.
    echo Attempting to auto-fix...
    
    :: Install PyTorch 2.4.0 (CUDA 12.1)
    echo Installing PyTorch...
    "%PIP%" install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    
    :: Install compatible gsplat
    echo Installing GSplat...
    "%PIP%" install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu121
    
    :: Downgrade numpy to prevent crashes
    "%PIP%" install "numpy<2"
    
    :: Install Flask & App
    "%PIP%" install flask
    "%PIP%" install -e .
)

:: 3. Launch WebUI
echo.
echo Starting server...
echo Access the UI at: http://127.0.0.1:7860
echo.

"%PYTHON%" webui.py --preload

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The server crashed. Sometimes the server can crash if you just started the server and attemp to make a SBS movie. Try again run_webui.bat. See error message above if any.
    pause
)