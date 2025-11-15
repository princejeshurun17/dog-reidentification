@echo off
REM Start Dog Re-ID System (Backend + Frontend)
echo ========================================
echo Starting Dog Re-Identification System
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: .\venv\Scripts\Activate.ps1
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Start Backend (FastAPI) in new window
echo [1/2] Starting Backend Inference Service...
start "Dog Re-ID Backend" cmd /k "venv\Scripts\activate.bat && python backend\inference_service.py"
timeout /t 3 /nobreak >nul

REM Start Frontend (Flask) in new window
echo [2/2] Starting Frontend UI Server...
start "Dog Re-ID Frontend" cmd /k "venv\Scripts\activate.bat && python frontend\app.py"
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo System Started Successfully!
echo ========================================
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5000
echo.
echo Open http://localhost:5000 in your browser
echo.
echo Press any key to stop all services...
pause >nul

REM Kill the processes when user presses a key
echo.
echo Stopping services...
taskkill /FI "WINDOWTITLE eq Dog Re-ID Backend*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Dog Re-ID Frontend*" /T /F >nul 2>&1
echo Services stopped.
