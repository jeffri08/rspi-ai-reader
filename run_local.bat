@echo off
title Local Webcam Stream Server
echo ==================================================
echo   Starting Local Webcam Stream Server (Windows)
echo ==================================================
set PYTHON_PATH="C:\Users\Asus\AppData\Local\Programs\Python\Python313\python.exe"

if exist %PYTHON_PATH% (
    echo Using Python at: %PYTHON_PATH%
    %PYTHON_PATH% stream_server.py --port 8085 --device 0
) else (
    echo Python 3.13 not found at default location. Trying system PATH...
    python stream_server.py --port 8085 --device 0
)
pause
