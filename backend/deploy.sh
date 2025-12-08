#!/bin/bash

# Deployment script for PhysAI Foundations Backend

set -e  # Exit on any error

# Default values
ENVIRONMENT="local"
PORT=8000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --environment ENV    Set environment (local|production) [default: local]"
            echo "  -p, --port PORT          Set port number [default: 8000]"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Deploying PhysAI Foundations Backend in $ENVIRONMENT environment..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix/Linux/macOS
    source venv/bin/activate
fi

# Install dependencies
echo "Installing/updating dependencies..."
pip install -r requirements.txt

# Check if environment variables are set
if [ -z "$QDRANT_URL" ] || [ -z "$QDRANT_API_KEY" ] || [ -z "$GOOGLE_API_KEY" ]; then
    if [ ! -f ".env" ]; then
        echo "Warning: .env file not found. Please create one with QDRANT_URL, QDRANT_API_KEY, and GOOGLE_API_KEY"
        echo "Using .env.example as template..."
        cp .env.example .env
        echo "Please update .env with your actual API keys before running the server."
    else
        echo "Loading environment variables from .env file..."
        export $(grep -v '^#' .env | xargs)
    fi
fi

# Run migrations or setup if needed
echo "Running setup tasks..."
# Add any setup tasks here if needed

# Start the application
echo "Starting server on port $PORT..."
if [ "$ENVIRONMENT" == "production" ]; then
    # Production deployment with uvicorn and gunicorn
    if ! command -v gunicorn &> /dev/null; then
        pip install "uvicorn[standard]"
    fi
    exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 4
else
    # Local development
    uvicorn main:app --host 0.0.0.0 --port $PORT --reload
fi