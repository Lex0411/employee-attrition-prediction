@echo off
title Employee Attrition Prediction
cd /d "%~dp0"

echo Installing dependencies (first run may take a minute)...
py -3 -m pip install -r requirements.txt -q
if errorlevel 1 (
    python -m pip install -r requirements.txt -q
)

echo.
echo Starting application...
echo Open your browser at: http://127.0.0.1:5000
echo Press Ctrl+C to stop.
echo.

py -3 app.py
if errorlevel 1 (
    python app.py
)

pause
