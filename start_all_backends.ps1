# Start All Backends (RAG + ChatKit)
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting All Backend Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting RAG Backend (port 8000)..." -ForegroundColor Yellow
Write-Host "Starting ChatKit Backend (port 8001)..." -ForegroundColor Yellow
Write-Host ""

$ragBackendDir = Join-Path $PSScriptRoot "backend"
$chatkitBackendDir = Join-Path $PSScriptRoot "backend-chatkit"

# Start RAG Backend in new window
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", "cd '$ragBackendDir'; & '.\venv\Scripts\python.exe' main.py"

# Wait a moment
Start-Sleep -Seconds 2

# Start ChatKit Backend in new window
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", "cd '$chatkitBackendDir'; & '.\venv\Scripts\python.exe' main.py"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Both backends are starting..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "RAG Backend:     http://localhost:8000" -ForegroundColor Cyan
Write-Host "ChatKit Backend: http://localhost:8001" -ForegroundColor Cyan
Write-Host "ChatKit Health:  http://localhost:8001/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit (backends will continue running in separate windows)..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
