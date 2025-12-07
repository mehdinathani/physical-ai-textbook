from fastapi import FastAPI
from typing import Optional

app = FastAPI(
    title="PhysAI Foundations Backend",
    description="Backend service for Physical AI & Humanoid Robotics Textbook",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to PhysAI Foundations Backend", "status": "running"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "PhysAI Foundations Backend"}

@app.get("/api/hello")
def hello_world(name: Optional[str] = None):
    if name:
        return {"message": f"Hello, {name}! Welcome to Physical AI & Humanoid Robotics."}
    return {"message": "Hello World! Welcome to Physical AI & Humanoid Robotics."}

# Additional endpoints will be added as the system develops
# - /api/chat for RAG-powered conversations
# - /api/auth for authentication
# - /api/content for content-related APIs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)