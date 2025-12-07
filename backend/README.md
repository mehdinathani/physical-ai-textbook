# PhysAI Foundations Backend

This is the backend service for the Physical AI & Humanoid Robotics Textbook project.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the development server:
```bash
uvicorn main:app --reload
```

The server will be available at `http://localhost:8000`.

## API Endpoints

- `GET /` - Root endpoint
- `GET /api/health` - Health check
- `GET /api/hello` - Hello World endpoint