@echo off
:: DisasterVision — One-command startup (Windows)
:: Run from the project root: Disaster AI\
:: Requires: Python venv activated (or use full path to python)

echo.
echo ======================================
echo  DisasterVision — Starting Services
echo ======================================
echo.

:: ── Install backend deps if needed ────────────────────────────────────
echo [1/3] Checking backend dependencies...
pip install -q -r disaster_vision\backend\requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ERROR: pip install failed. Make sure your venv is active.
    pause & exit /b 1
)

:: ── Install frontend deps if needed ───────────────────────────────────
echo [2/3] Checking frontend dependencies...
cd disaster_vision\frontend
call npm install --silent
if %ERRORLEVEL% neq 0 (
    echo ERROR: npm install failed. Make sure Node.js is installed.
    pause & exit /b 1
)
cd ..\..

:: ── Start backend in a new window ─────────────────────────────────────
echo [3/3] Starting services...
start "DisasterVision Backend" cmd /k "uvicorn disaster_vision.backend.main:app --host 0.0.0.0 --port 8000 --reload"

:: Wait briefly for backend to start
timeout /t 3 /nobreak >nul

:: ── Start frontend in a new window ────────────────────────────────────
start "DisasterVision Frontend" cmd /k "cd disaster_vision\frontend && npm run dev"

echo.
echo Both services are starting in separate windows.
echo.
echo  Backend  → http://localhost:8000
echo  Frontend → http://localhost:5175
echo  API Docs → http://localhost:8000/docs
echo.
echo Opening browser in 4 seconds...
timeout /t 4 /nobreak >nul
start http://localhost:5175
