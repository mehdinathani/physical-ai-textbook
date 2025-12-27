# Production Deployment URLs

**Deployment Date**: 2025-12-27
**Platform**: Render.com (Free Tier)
**Status**: ✅ Live and Operational

---

## Backend Services

### RAG Backend
- **Service Name**: `physai-backend-rag`
- **Production URL**: https://physai-backend.onrender.com
- **Health Endpoint**: https://physai-backend.onrender.com/api/health
- **Expected Response**: `{"status":"healthy","service":"Textbook Chat API"}`
- **Status**: ✅ **LIVE**

### ChatKit Backend
- **Service Name**: `physai-backend-chatkit`
- **Production URL**: https://physai-backend-chatkit.onrender.com
- **Health Endpoint**: https://physai-backend-chatkit.onrender.com/health
- **ChatKit Endpoint**: https://physai-backend-chatkit.onrender.com/chatkit
- **Expected Response**: `{"status":"ok","model":"gemini-2.5-flash-lite"}`
- **Status**: ✅ **LIVE**

---

## Frontend

- **Platform**: Vercel
- **Production URL**: https://physai-foundations.vercel.app
- **Status**: ✅ Deployed (needs backend URL updates)

---

## Quick Health Checks

```bash
# RAG Backend
curl https://physai-backend.onrender.com/api/health

# ChatKit Backend
curl https://physai-backend-chatkit.onrender.com/health

# Test RAG functionality
curl -X POST https://physai-backend.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is ROS2?"}'
```

---

## Important Notes

### Free Tier Behavior
- **Cold Start**: Services sleep after 15 minutes of inactivity
- **First Request**: May take 30-60 seconds to wake up
- **Subsequent Requests**: Fast (~2-5 seconds)

### API Keys Configuration
All API keys are configured securely in Render dashboard (not in code):
- ✅ QDRANT_URL
- ✅ QDRANT_API_KEY
- ✅ GOOGLE_API_KEY
- ✅ GEMINI_API_KEY (ChatKit backend)

---

**Last Updated**: 2025-12-27
