@echo off
setlocal

echo Starting ml-sharp API server...
echo Activating virtual environment...

REM Activate the virtual environment
call .venv\Scripts\activate

REM Run the server using python
python src/start_server.py

pause