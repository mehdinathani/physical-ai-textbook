@echo off
REM Deployment script for PhysAI Foundations Backend (Windows)

setlocal enabledelayedexpansion

REM Default values
set "ENVIRONMENT=local"
set "PORT=8000"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto args_parsed
if "%~1"=="-e" set "ENVIRONMENT=%~2"
if "%~1"=="--environment" set "ENVIRONMENT=%~2"
if "%~1"=="-p" set "PORT=%~2"
if "%~1"=="--port" set "PORT=%~2"
if "%~1"=="-h" goto show_help
if "%~1"=="--help" goto show_help
shift
goto parse_args

:show_help
echo Usage: %0 [OPTIONS]
echo Options:
echo   -e, --environment ENV    Set environment (local^|production) [default: local]
echo   -p, --port PORT          Set port number [default: 8000]
echo   -h, --help               Show this help message
exit /b 0

:args_parsed

echo Deploying PhysAI Foundations Backend in !ENVIRONMENT! environment...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing/updating dependencies...
pip install -r requirements.txt

REM Check if environment variables are set
if "%QDRANT_URL%"=="" (
    if not exist ".env" (
        echo Warning: .env file not found. Please create one with QDRANT_URL, QDRANT_API_KEY, and GOOGLE_API_KEY
        echo Using .env.example as template...
        copy .env.example .env
        echo Please update .env with your actual API keys before running the server.
    ) else (
        echo Loading environment variables from .env file...
        for /f "tokens=1,* delims==" %%a in (.env) do (
            if not "%%b"=="" (
                set "%%a=%%b"
            )
        )
    )
)

REM Run setup tasks
echo Running setup tasks...

REM Start the application
echo Starting server on port !PORT!...
if "!ENVIRONMENT!"=="production" (
    REM Production deployment
    python -m uvicorn main:app --host 0.0.0.0 --port !PORT! --workers 4
) else (
    REM Local development
    python -m uvicorn main:app --host 0.0.0.0 --port !PORT! --reload
)