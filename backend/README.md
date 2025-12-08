# PhysAI Foundations Backend

This is the backend service for the Physical AI & Humanoid Robotics Textbook project. Implements a RAG (Retrieval-Augmented Generation) system for textbook content using Google Gemini.

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

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys:
     - `QDRANT_URL` - Your Qdrant Cloud URL
     - `QDRANT_API_KEY` - Your Qdrant API key
     - `GOOGLE_API_KEY` - Your Google API key (for Gemini and embeddings)

5. Run the development server:
```bash
uvicorn main:app --reload
```

The server will be available at `http://localhost:8000`.

## Ingestion

Before using the chat API, you need to ingest the textbook content:

```bash
python ingest.py
```

This will:
- Find all markdown files in `../frontend/docs`
- Strip Docusaurus frontmatter
- Chunk the content
- Generate embeddings using Google's text-embedding-004 model (768 dimensions)
- Upload to Qdrant vector database

**Note**: If you're using Python 3.14, you may encounter compatibility issues with protobuf-based libraries. Use Python 3.11 or 3.12 for best compatibility.

## API Endpoints

- `GET /` - Root endpoint
- `GET /api/health` - Health check
- `POST /api/chat` - Chat endpoint that implements RAG logic (embed query using Google embeddings -> search Qdrant -> construct system prompt -> call Google Gemini for response)