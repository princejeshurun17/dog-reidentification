# Stop Dog Re-ID System
Write-Host "Stopping Dog Re-ID System..." -ForegroundColor Yellow

# Kill processes by window title
Get-Process | Where-Object { $_.MainWindowTitle -like "*Dog Re-ID*" } | Stop-Process -Force -ErrorAction SilentlyContinue

# Also kill by process name as fallback
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
foreach ($proc in $pythonProcesses) {
    $cmdline = (Get-WmiObject Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
    if ($cmdline -like "*inference_service.py*" -or $cmdline -like "*frontend\app.py*") {
        Write-Host "Stopping: $cmdline" -ForegroundColor Gray
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "All services stopped." -ForegroundColor Green
