# Start Dog Re-ID System (Backend + Frontend)
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Dog Re-Identification System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv" -ForegroundColor Yellow
    Write-Host "Then: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "Then: pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Start Backend (FastAPI)
Write-Host "[1/2] Starting Backend Inference Service..." -ForegroundColor Green
$backend = Start-Process powershell -ArgumentList "-NoExit", "-Command", "& { `$host.ui.RawUI.WindowTitle='Dog Re-ID Backend'; .\venv\Scripts\Activate.ps1; python backend\inference_service.py }" -PassThru
Start-Sleep -Seconds 3

# Start Frontend (Flask)
Write-Host "[2/2] Starting Frontend UI Server..." -ForegroundColor Green
$frontend = Start-Process powershell -ArgumentList "-NoExit", "-Command", "& { `$host.ui.RawUI.WindowTitle='Dog Re-ID Frontend'; .\venv\Scripts\Activate.ps1; python frontend\app.py }" -PassThru
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "System Started Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Backend:  http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "Frontend: http://127.0.0.1:5000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Open http://localhost:5000 in your browser" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to stop all services..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Stop services
Write-Host ""
Write-Host "Stopping services..." -ForegroundColor Yellow
Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
Write-Host "Services stopped." -ForegroundColor Green
