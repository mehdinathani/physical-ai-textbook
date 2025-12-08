# Phase 2 Architecture: RAG Backend & Ingestion System

## Overview
This document outlines the architecture for the RAG (Retrieval-Augmented Generation) backend system for the Physical AI & Humanoid Robotics Textbook. The system will index Docusaurus content and provide an API for question-answering.

## 1. Environment Requirements

### Runtime Environment
- Python 3.10+ in `backend/` directory
- Virtual environment recommended for dependency isolation

### Dependencies
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
qdrant-client==1.9.1
openai==1.3.7
python-dotenv==1.0.0
langchain-text-splitters==0.1.0
python-frontmatter==1.1.0
pydantic==2.5.0
```

### System Requirements
- Minimum 4GB RAM for vector operations
- Stable internet connection for API calls
- Access to OpenAI API and Qdrant Cloud

## 2. Data Structure (Qdrant Point)

### Vector Configuration
- Dimensions: 768 (Google text-embedding-004 standard)
- Distance metric: Cosine similarity
- Collection name: `textbook_docs`

### Payload Structure
```json
{
  "text": "chunk_content",
  "source": "module-1/chapter-1",
  "page_url": "/docs/module-1/...",
  "chunk_index": 0,
  "metadata": {
    "sidebar_position": 1
  }
}
```

### Indexing Strategy
- Text chunks: 1000 tokens with 200-token overlap
- Markdown-aware splitting to preserve document structure
- Frontmatter metadata extraction and storage

## 3. Security Requirements

### Environment Variables
- Store `GOOGLE_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` in `.env` file inside `backend/`
- **Never commit `.env` file** - add to `.gitignore`
- Use `python-dotenv` for secure loading

### API Security
- Rate limiting to prevent abuse
- Input validation for all endpoints
- Authentication layer for production deployment
- Secure connection to Qdrant (TLS)

### Data Security
- No sensitive user data stored in Qdrant
- OpenAI API calls comply with data privacy requirements
- Logs do not contain sensitive information

## 4. System Components

### Ingestion Pipeline (`backend/ingest.py`)
- Scans `frontend/docs` directory recursively
- Parses markdown files and extracts frontmatter
- Splits content using LangChain text splitters
- Generates embeddings with Google text-embedding-004 API
- Uploads to Qdrant with proper metadata

### API Service (`backend/main.py`)
- FastAPI application with `/api/chat` endpoint
- Query embedding generation with Google text-embedding-004
- Qdrant similarity search
- Context-aware response generation with Google Gemini 1.5 Flash
- Source attribution in responses

## 5. Deployment Guidelines

### Local Development
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python -m ingest
uvicorn main:app --reload
```

### Production Deployment
- Containerize with Docker
- Use environment variables for configuration
- Implement health checks
- Set up monitoring and logging
- Use reverse proxy (nginx) for HTTPS

## 6. Error Handling & Monitoring

### Error Handling
- Graceful degradation when APIs are unavailable
- Comprehensive logging for debugging
- Input validation and sanitization
- Proper HTTP status codes

### Monitoring
- API response times
- Vector search performance
- API token usage
- System resource utilization

## 7. Performance Considerations

### Caching
- Query result caching for common questions
- Embedding caching to avoid duplicate API calls

### Optimization
- Batch processing for ingestion
- Asynchronous operations where possible
- Efficient vector search parameters

## 8. Scalability

### Horizontal Scaling
- Stateless API service for easy scaling
- Qdrant cluster for handling increased load
- CDN for static assets

### Future Enhancements
- Support for additional document formats
- Multi-language support
- Advanced search capabilities

## 9. Frontend Integration

### Chat Widget Implementation
- Floating Action Button (FAB) for chat interface
- React component with TypeScript
- Integration with Docusaurus via Root component
- API communication with backend service
- Markdown rendering for AI responses

### Tech Stack
- React (with TypeScript) for component development
- Tailwind CSS for styling
- react-markdown for content rendering
- Docusaurus theme swizzling for global integration

### Architecture
- Simple state management with React hooks
- Component-based design with clear separation of concerns
- Responsive design for all device sizes
- Dark mode compatibility with Docusaurus theme