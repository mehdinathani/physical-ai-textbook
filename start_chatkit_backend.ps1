# Start ChatKit Backend (Port 8001)
Write-Host "Starting ChatKit Backend on port 8001..." -ForegroundColor Cyan

$backendDir = Join-Path $PSScriptRoot "backend-chatkit"
Set-Location $backendDir

# Check if venv exists
if (!(Test-Path ".\venv")) {
    Write-Host "Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    & ".\venv\Scripts\pip.exe" install -r requirements.txt
}

# Check if .env exists
if (!(Test-Path ".\.env")) {
    Write-Host "WARNING: .env file not found. Copy .env.example to .env and add your GEMINI_API_KEY" -ForegroundColor Red
    Write-Host "Press any key to continue anyway..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# Activate virtual environment and run
Write-Host "Launching ChatKit backend..." -ForegroundColor Green
& ".\venv\Scripts\python.exe" main.py
