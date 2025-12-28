# PhysAI Foundations Deployment Runbook

## Quick Redeployment

1. Make code changes locally
2. Commit and push to master:
   ```bash
   git add .
   git commit -m "feat: Description of changes"
   git push origin master
   ```
3. Render auto-detects push and redeploys (5-10 min)
4. Vercel auto-deploys frontend (~2-3 min)
5. Monitor build logs in respective dashboards

## Rollback Procedures

### Backend Rollback (Render)

1. Go to Render Dashboard: https://dashboard.render.com/
2. Navigate to the service (`physai-backend-rag` or `physai-backend-chatkit`)
3. Go to **Deploys** tab
4. Find the last working deployment
5. Click **"Redeploy"** button
6. Verify health checks pass after deployment

### Frontend Rollback (Vercel)

1. Go to Vercel Dashboard: https://vercel.com/dashboard
2. Find project: `physical-ai-textbook`
3. Go to **Deployments** tab
4. Find the working deployment
5. Click **"..."** → **"Redeploy"**
6. Or revert git and push:
   ```bash
   git revert <commit-hash>
   git push origin master
   ```

## Health Check Commands

```bash
# RAG Backend
curl https://physai-backend.onrender.com/api/health
# Expected: {"status":"healthy","service":"Textbook Chat API"}

# ChatKit Backend
curl https://physai-backend-chatkit.onrender.com/health
# Expected: {"status":"ok","model":"gemini-2.5-flash-lite"}

# Frontend
curl -I https://physical-ai-textbook-git-master-mehdinathanis-projects.vercel.app
# Expected: HTTP 200
```

## Common Issues

### Issue: 503 Service Unavailable
- **Cause**: Service sleeping (free tier after 15 min inactivity)
- **Fix**: Wait 30-60 seconds for cold start

### Issue: Health check fails
- **Cause**: Build error or environment variable misconfiguration
- **Fix**: Check Render logs, verify environment variables in Settings → Environment

### Issue: CORS errors on frontend
- **Cause**: Frontend domain not in CORS whitelist
- **Fix**: Update `allow_origins` in:
  - `backend/main.py:80`
  - `backend-chatkit/main.py:410`
- Then redeploy the backend

### Issue: ModuleNotFoundError: No module named 'src'
- **Cause**: Working directory context issue on Render
- **Fix**: Ensure build/start commands use `cd backend &&` prefix (already configured in render.yaml)

### Issue: Frontend shows wrong backend URL
- **Cause**: Using local `.env.local` instead of production
- **Fix**: Use `frontend/.env.production` for Vercel deployments

### Issue: ChatWidget not connecting
1. Check browser DevTools Console for errors
2. Verify CORS is configured for frontend domain
3. Verify health endpoint works: `curl https://physai-backend-chatkit.onrender.com/health`
4. Check Render logs for backend errors

## Monitoring

### Dashboards
- **Render**: https://dashboard.render.com/
- **Vercel**: https://vercel.com/dashboard

### Logs
- Backend: Render Dashboard → Service → Logs
- Frontend: Vercel Dashboard → Deployments → Function Logs

### Metrics to Watch
- Backend response times
- Error rates (>5% needs attention)
- Free tier hours usage (750 hrs/month per service)

## Environment Variables Reference

### RAG Backend (`physai-backend-rag`)
| Variable | Value Source |
|----------|--------------|
| PYTHON_VERSION | 3.11.0 |
| PORT | 10000 |
| QDRANT_URL | backend/.env |
| QDRANT_API_KEY | backend/.env |
| GOOGLE_API_KEY | backend/.env |

### ChatKit Backend (`physai-backend-chatkit`)
| Variable | Value Source |
|----------|--------------|
| PYTHON_VERSION | 3.11.0 |
| PORT | 10000 |
| GEMINI_API_KEY | backend-chatkit/.env |
| QDRANT_URL | backend-chatkit/.env |
| QDRANT_API_KEY | backend-chatkit/.env |
| GOOGLE_API_KEY | backend-chatkit/.env |

### Frontend (Vercel)
| Variable | Value |
|----------|-------|
| VITE_CHATKIT_BACKEND_URL | https://physai-backend-chatkit.onrender.com/chatkit |
| VITE_RAG_BACKEND_URL | https://physai-backend.onrender.com/api/chat |
| VITE_CHATKIT_DOMAIN_KEY | domain_pk_... (from OpenAI platform) |

## Free Tier Limitations

- Services sleep after 15 minutes inactivity
- Cold start: 30-60 seconds
- 750 hours/month per service (sufficient for 2 services 24/7)
- 512 MB RAM per service

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PhysAI Foundations                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────┐     ┌─────────────────┐               │
│   │  Vercel         │     │  Render         │               │
│   │  Frontend       │────▶│  ChatKit API    │               │
│   │  (Docusaurus)   │     │  (FastAPI)      │               │
│   └─────────────────┘     └────────┬────────┘               │
│                                    │                        │
│                                    ▼                        │
│                           ┌─────────────────┐               │
│                           │  Qdrant Cloud   │               │
│                           │  (Vector DB)    │               │
│                           └─────────────────┘               │
│                                                              │
│   URLs:                                                     │
│   - Frontend: https://physical-ai-textbook-git-master-mehdinathanis-projects.vercel.app │
│   - ChatKit:  https://physai-backend-chatkit.onrender.com/chatkit │
│   - RAG:      https://physai-backend.onrender.com/api/chat  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Version Info

- **Last Updated**: 2025-12-28
- **Backend**: Python 3.11, FastAPI
- **Frontend**: TypeScript, Docusaurus 3.4+
- **LLM**: Google Gemini 2.5 Flash Lite
- **Vector DB**: Qdrant Cloud (Free Tier)
