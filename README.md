# PhysAI Foundations - Physical AI & Humanoid Robotics Textbook

An interactive educational platform for learning Physical AI and Humanoid Robotics with integrated RAG-powered chat assistance.

## 🚀 Production Deployment

**Status**: ✅ **LIVE AND OPERATIONAL**

### Live URLs

- **Frontend (Docusaurus)**: https://physai-foundations.vercel.app
- **RAG Backend**: https://physai-backend.onrender.com
- **ChatKit Backend**: https://physai-backend-chatkit.onrender.com

### Quick Health Checks

```bash
# RAG Backend
curl https://physai-backend.onrender.com/api/health

# ChatKit Backend
curl https://physai-backend-chatkit.onrender.com/health
```

---

## 🏗️ Architecture

### Dual Backend Architecture

**1. RAG Backend** (`backend/`)
- **Framework**: FastAPI
- **Purpose**: Traditional RAG chatbot with Qdrant vector database
- **Features**:
  - Textbook content ingestion
  - Vector embeddings (Google text-embedding-004)
  - Semantic search
  - Context-aware responses
- **Port**: 8001 (local), 10000 (production)

**2. ChatKit Backend** (`backend-chatkit/`)
- **Framework**: FastAPI + OpenAI ChatKit Server
- **Purpose**: Modern chat interface with streaming support
- **Features**:
  - OpenAI ChatKit integration
  - Gemini 2.5 Flash Lite LLM
  - RAG integration with Qdrant
  - Thread persistence
  - Streaming responses
  - 📚 indicator for RAG-enhanced answers
  - 💭 indicator for general knowledge
- **Port**: 8000 (local), 10000 (production)

**3. Frontend** (`frontend/`)
- **Framework**: Docusaurus 3.1.0
- **Features**:
  - Static documentation site
  - Integrated ChatKit widget
  - Responsive design
  - Dark mode support
- **Port**: 3000 (local)

---

## 🛠️ Technology Stack

### Backend
- **Runtime**: Python 3.11
- **Web Framework**: FastAPI
- **LLM Provider**: Google Gemini 2.5 Flash Lite
- **Vector Database**: Qdrant Cloud (Free Tier)
- **Embeddings**: Google text-embedding-004
- **Deployment**: Render.com (Free Tier)

### Frontend
- **Framework**: Docusaurus 3.1.0
- **UI Library**: React 18
- **Chat Component**: OpenAI ChatKit React
- **Styling**: Custom CSS + Tailwind utilities
- **Deployment**: Vercel

---

## 💻 Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Setup Backend (RAG)

```bash
# Navigate to RAG backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
cp .env.example .env
# Edit .env and add your API keys

# Run ingestion (one-time)
python ingest.py

# Start server
uvicorn main:app --reload --port 8001
```

### Setup Backend (ChatKit)

```bash
# Navigate to ChatKit backend
cd backend-chatkit

# Create virtual environment (if not already created)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your API keys

# Start server
uvicorn main:app --reload --port 8000
```

### Setup Frontend

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The site will be available at http://localhost:3000

---

## 🔑 Required API Keys

Create `.env` files in both backend directories with:

### `backend/.env`
```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GOOGLE_API_KEY=your_google_api_key
```

### `backend-chatkit/.env`
```env
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GOOGLE_API_KEY=your_google_api_key
```

---

## 📦 Deployment

### Backend Deployment (Render)

Both backends are deployed on Render.com free tier:

**Configuration**:
- **Runtime**: Python 3.11
- **Build Command (RAG)**: `cd backend && pip install -r requirements.txt`
- **Start Command (RAG)**: `cd backend && uvicorn main:app --host 0.0.0.0 --port 10000`
- **Build Command (ChatKit)**: `cd backend-chatkit && pip install -r requirements.txt`
- **Start Command (ChatKit)**: `cd backend-chatkit && uvicorn main:app --host 0.0.0.0 --port 10000`

**Environment Variables**: Configured in Render dashboard (not in code)

**See**: `DEPLOYMENT_GUIDE.md` for detailed instructions

### Frontend Deployment (Vercel)

Frontend automatically deploys from GitHub `master` branch.

**Production Environment**: `frontend/.env.production`

---

## 📚 Documentation

- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **Production URLs**: `PRODUCTION_URLS.md`
- **Deployment Runbook**: `docs/DEPLOYMENT_RUNBOOK.md`
- **ChatKit Setup**: `CHATKIT_SETUP.md`
- **Project Constitution**: `.specify/memory/constitution.md`
- **Deployment Plan**: `specs/master/plan.md`
- **Implementation Tasks**: `specs/master/tasks.md`

---

## 🎯 Features

### Interactive Chat Widget
- **Streaming Responses**: Real-time message streaming
- **RAG Integration**: Answers enhanced with textbook content
- **Thread Persistence**: Conversation history saved locally
- **Context Indicators**:
  - 📚 = Answer from textbook (RAG)
  - 💭 = General knowledge
- **Responsive Design**: Works on desktop and mobile

### Educational Content
- Structured textbook chapters
- Interactive examples
- Code snippets
- Visual diagrams
- Progressive learning path

---

## ⚠️ Important Notes

### Free Tier Limitations

**Render.com (Backends)**:
- Services sleep after 15 minutes of inactivity
- Cold start: 30-60 seconds for first request
- 750 hours/month per service (sufficient for 24/7)
- 512 MB RAM per service

**Qdrant Cloud**:
- Free tier: 1GB cluster
- Sufficient for educational use case

**Google Gemini**:
- Free tier: 60 requests/minute
- Adequate for development and testing

---

## 🐛 Troubleshooting

### Backend Not Responding
- **Issue**: 503 Service Unavailable
- **Cause**: Cold start (service was sleeping)
- **Solution**: Wait 30-60 seconds, service will wake up

### Chat Widget Not Connecting
- **Issue**: CORS errors in browser console
- **Solution**: Verify `allow_origins` in backend CORS configuration

### "ModuleNotFoundError" on Render
- **Issue**: Python can't find `src` module
- **Solution**: Ensure start command includes `cd backend` or `cd backend-chatkit`

---

## 🤝 Contributing

This is an educational project. Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

Educational project - See repository for license details

---

## 📞 Support

For issues or questions:
1. Check documentation in `/docs`
2. Review `DEPLOYMENT_GUIDE.md`
3. Open an issue on GitHub

---

**Last Updated**: 2025-12-27
**Version**: 1.0.0
**Status**: Production Ready ✅
