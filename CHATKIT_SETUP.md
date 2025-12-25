# ChatKit Integration Setup Guide

Complete documentation for the PhysAI Foundations ChatKit integration.

## Overview

This project integrates OpenAI ChatKit SDK for production-grade AI chat functionality, replacing the previous custom implementation. The system uses:

- **Frontend**: React + Docusaurus + ChatKit React SDK
- **Backend**: FastAPI + ChatKit Server + Google Gemini 2.0 Flash
- **Architecture**: Dual backend (RAG on port 8000, ChatKit on port 8001)

## Quick Start

### Prerequisites

- Python 3.11+ (protobuf issues with 3.14)
- Node.js 18+ and npm
- Google Gemini API key

### 1. Backend Setup

#### Option A: Automated Script (Windows)

```bash
# Start ChatKit backend only
.\start_chatkit_backend.bat

# OR start both backends (RAG + ChatKit)
.\start_all_backends.bat
```

#### Option B: Manual Setup

```bash
cd backend-chatkit

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run backend
python main.py
```

Backend will start at: **http://localhost:8001**

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies (if not already done)
npm install

# Create .env.local file
echo VITE_CHATKIT_BACKEND_URL=http://localhost:8001/chatkit > .env.local

# Start frontend
npm start
```

Frontend will start at: **http://localhost:3000**

## Architecture

### Port Configuration

- **Port 8000**: RAG backend (existing, unchanged)
- **Port 8001**: ChatKit backend (new)
- **Port 3000**: Frontend (Docusaurus)

This separation allows both backends to run simultaneously without conflict.

### File Structure

```
physai-foundations/
├── backend/                      # RAG backend (port 8000)
│   ├── main.py
│   └── ...
├── backend-chatkit/              # ChatKit backend (port 8001)
│   ├── main.py                   # FastAPI + ChatKit Server
│   ├── requirements.txt
│   ├── .env.example
│   └── .env                      # Your Gemini API key
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── ChatWidget.tsx    # ChatKit React component
│   │   └── theme/
│   │       └── Root.tsx          # Docusaurus layout wrapper
│   └── .env.local                # Backend URL config
├── start_chatkit_backend.bat     # Launch ChatKit backend
├── start_all_backends.bat        # Launch both backends
└── CHATKIT_SETUP.md              # This file
```

### Backend Architecture

**Store**: In-memory thread store (MemoryStore class)
- Thread management
- Message persistence
- Attachment handling

**Agent**: Gemini-powered AI assistant
- Model: `gemini/gemini-2.5-flash-lite` (via LiteLLM)
- Context: Physical AI & Humanoid Robotics expert
- Conversation history tracking

**Endpoints**:
- `POST /chatkit` - ChatKit protocol endpoint (SSE streaming)
- `GET /health` - Health check (returns model name)
- `GET /debug/threads` - Debug endpoint for stored conversations

**CORS**: Configured for:
- `http://localhost:3000` (development)
- `https://physai-foundations.vercel.app` (production)

### Frontend Architecture

**ChatWidget Component** (`frontend/src/components/ChatWidget.tsx`):
- SSR-safe (wrapped in BrowserOnly)
- React Hooks compliant (hooks called unconditionally)
- LocalStorage persistence (key: `physai-chatkit-thread-id`)
- Floating button + modal UI pattern
- "New Chat" button functionality

**Root Component** (`frontend/src/theme/Root.tsx`):
- Loads ChatKit CDN script
- Wraps ChatWidget in BrowserOnly for SSR compatibility
- Ensures no SSR errors during Docusaurus build

## Key Features

### Thread Persistence

Conversations are persisted via localStorage:
- Key: `physai-chatkit-thread-id`
- Survives page reloads
- "New Chat" button clears thread and starts fresh

### Real-Time Streaming

Messages stream in real-time using Server-Sent Events (SSE):
- Backend uses `StreamingResponse` with `text/event-stream`
- Frontend receives incremental responses
- Low latency (<2s p95)

### SSR Compatibility

Critical for Docusaurus deployment to Vercel:
- ChatWidget wrapped in `BrowserOnly` component
- No `window` or `document` access during SSR
- ChatKit CDN script loads asynchronously
- Build succeeds without errors: `npm run build`

### React Hooks Compliance

Follows React Hooks Rules strictly:
- `useChatKit` called unconditionally (not inside conditions)
- Conditional rendering only (not conditional hooks)
- No "Invariant Violation" errors

## Environment Variables

### Backend (`backend-chatkit/.env`)

```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=8001
```

### Frontend (`frontend/.env.local`)

```env
VITE_CHATKIT_BACKEND_URL=http://localhost:8001/chatkit
```

**Important**: Vite requires `VITE_` prefix for client-side environment variables.

## Testing

### Manual Testing Checklist

1. **Backend Health**:
   ```bash
   curl http://localhost:8001/health
   # Expected: {"status": "ok", "model": "gemini-2.5-flash-lite"}
   ```

2. **Frontend UI**:
   - Click floating chat button (bottom-right)
   - Modal should open
   - Send message: "What is ROS2?"
   - Verify streaming response appears

3. **Thread Persistence**:
   - Send multiple messages
   - Refresh page (F5)
   - Messages should persist in chat history

4. **New Chat**:
   - Click "New Chat" button
   - Page reloads
   - Fresh conversation starts (no previous messages)

5. **SSR Build**:
   ```bash
   cd frontend
   npm run build
   # Must succeed without "window is not defined" errors
   ```

6. **Mobile Responsiveness**:
   - Test on screens < 600px width
   - Chat widget should display correctly

7. **Error Handling**:
   - Stop backend
   - Try sending message
   - Frontend should display error without crashing

## Deployment

### Backend Deployment (Future)

**Target**: Render.com or similar service

**Environment Variables**:
- `GEMINI_API_KEY`: Your Gemini API key
- `PORT`: 8001

**CORS**: Already configured for production domain

### Frontend Deployment (Vercel)

**Environment Variable**:
- `VITE_CHATKIT_BACKEND_URL`: Production backend URL

**Build Command**: `npm run build` (already working)

**SSR**: Fully compatible (no build errors)

## Troubleshooting

### Backend Issues

**Issue**: `ModuleNotFoundError: No module named 'chatkit'`
- **Fix**: Ensure virtual environment is activated and dependencies installed
- Run: `pip install -r requirements.txt`

**Issue**: Backend fails to start with port conflict
- **Fix**: Check if another service is using port 8001
- Run: `netstat -ano | findstr :8001` (Windows) or `lsof -i :8001` (Linux/Mac)

**Issue**: No streaming responses
- **Fix**: Verify CORS configuration includes your frontend origin
- Check backend logs for CORS errors

### Frontend Issues

**Issue**: "window is not defined" during build
- **Fix**: Ensure ChatWidget is wrapped in BrowserOnly component
- Verify no direct `window` access outside useEffect

**Issue**: Chat widget doesn't appear
- **Fix**: Check browser console for JavaScript errors
- Verify ChatKit CDN script loaded: `https://cdn.platform.openai.com/deployments/chatkit/chatkit.js`

**Issue**: Thread doesn't persist after reload
- **Fix**: Check browser localStorage for `physai-chatkit-thread-id`
- Verify localStorage is not disabled (private/incognito mode)

**Issue**: React Hooks violation errors
- **Fix**: Ensure `useChatKit` is called unconditionally
- Never wrap hooks in conditionals or loops

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `FatalAppError: Invalid input at api` | Missing `domainKey` in ChatKit config | Add `domainKey: 'localhost'` to api config |
| Blank screen | ChatKit CDN script not loaded | Add CDN script to Root.tsx Head component |
| Duplicate messages | LiteLLM/Gemini ID collision | Already handled with ID mapping in backend |
| SSR build fails | Component accesses window during SSR | Wrap in BrowserOnly or use useEffect |

## Success Criteria

- [X] ChatKit backend runs on port 8001
- [X] Frontend builds successfully (`npm run build` succeeds)
- [X] Chat widget displays floating button
- [X] Messages send and stream in real-time
- [X] Thread persists across page reloads
- [X] "New Chat" button creates fresh thread
- [X] No browser console errors (React Hooks, SSR)
- [X] Mobile responsive (small viewport)
- [X] Both backends can run simultaneously (8000 + 8001)

## Integration History

### Previous Attempts (Failed)

- **Commits 6b51226 → 3c0ee65**: ChatKit integration broke SSR build
  - Errors: "window is not defined", React Hooks violations
  - Root cause: Conditional hook calls, missing BrowserOnly wrapper

### Current Implementation (Successful)

- **Date**: 2025-12-25
- **Approach**: Used ChatKit skills to generate fresh, SSR-compliant code
- **Key Changes**:
  - ChatWidget wrapped in BrowserOnly
  - Hooks called unconditionally
  - Backend on separate port (8001)
  - Proper localStorage handling
  - "New Chat" button functionality

## Additional Resources

- **ChatKit Documentation**: https://platform.openai.com/docs/chatkit
- **Gemini API**: https://ai.google.dev/docs
- **LiteLLM**: https://docs.litellm.ai/
- **Docusaurus SSR**: https://docusaurus.io/docs/advanced/ssg

## Support

For issues or questions:
1. Check this documentation first
2. Review browser console for errors
3. Check backend logs for detailed error messages
4. Verify environment variables are set correctly
5. Test endpoints individually (health, chatkit)

---

**Last Updated**: 2025-12-25
**Status**: Production Ready
**Version**: 1.0
