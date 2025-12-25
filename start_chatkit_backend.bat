@echo off
echo Starting ChatKit Backend on port 8001...
powershell -ExecutionPolicy Bypass -File "%~dp0start_chatkit_backend.ps1"
pause
