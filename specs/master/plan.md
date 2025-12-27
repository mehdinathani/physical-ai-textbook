# Implementation Plan: Render Backend Deployment

**Feature**: Deploy PhysAI Foundations dual backend architecture to Render.com
**Branch**: master
**Status**: Planning
**Created**: 2025-12-27

---

## Executive Summary

Deploy both PhysAI backend services (RAG Backend + ChatKit Backend) to Render.com free tier with production-ready configuration, environment security, health monitoring, and frontend integration.

**Success Criteria**:
- ✅ Both backends deployed and accessible via HTTPS
- ✅ Health checks passing on both services
- ✅ Environment variables secured in Render dashboard
- ✅ Frontend updated with production backend URLs
- ✅ End-to-end chat functionality working in production
- ✅ Zero deployment errors or downtime

---

## Technical Context

### Current State
- **Local Development**: Both backends running successfully
  - RAG Backend: `http://localhost:8001` (port 8000 in production)
  - ChatKit Backend: `http://localhost:8000` (port 8000 in production)
- **Configuration**: `render.yaml` exists with dual service definitions
- **Dependencies**: All requirements.txt files present and tested
- **Frontend**: Deployed on Vercel, needs backend URL updates

### Target State
- **Production Deployment**: Both backends on Render.com free tier
  - Service 1: `physai-backend-rag` → `https://physai-backend-rag.onrender.com`
  - Service 2: `physai-backend-chatkit` → `https://physai-backend-chatkit.onrender.com`
- **Port Configuration**: Both use port 10000 on Render (free tier standard)
- **Health Monitoring**: Active health checks on `/api/health` and `/health`
- **CORS**: Configured for production Vercel domain

### Technology Stack
- **Platform**: Render.com (Free Tier)
- **Runtime**: Python 3.11
- **Backend Frameworks**: FastAPI (both services)
- **Vector Database**: Qdrant Cloud (Free Tier)
- **LLM Provider**: Google Gemini 2.5 Flash Lite
- **Deployment**: Blueprint via `render.yaml`

### Known Constraints
- **Free Tier Limits**:
  - Services sleep after 15 minutes inactivity
  - Cold start: 30-60 seconds
  - 750 hours/month per service (sufficient for 2 services 24/7)
  - 512 MB RAM per service
- **Environment Variables**: Must be configured manually in Render dashboard
- **Build Time**: ~5-10 minutes per service on first deploy
- **Dependencies**: No breaking changes allowed in deployment

---

## Constitution Check

### Alignment Review

✅ **Spec-Driven Execution**: Deployment follows documented specifications in `render.yaml` and `DEPLOYMENT_GUIDE.md`

✅ **Bilingual Excellence**: Python backends maintain separation from TypeScript frontend

✅ **Hackathon Velocity**: Render free tier chosen for rapid deployment without infrastructure complexity

✅ **AI-Native First**: Both backends leverage Gemini LLM with RAG integration

✅ **Academic Rigor**: N/A (infrastructure deployment, no content changes)

✅ **Modular Architecture**: Dual backend architecture maintains clear separation of concerns

✅ **Context Awareness**: Leveraging existing `render.yaml` and deployment documentation

✅ **Agent & Skill Discovery**: N/A (deployment task, not development)

✅ **Clean UX**: Backend URLs will be HTTPS with proper CORS for frontend

✅ **Code Quality**: No code changes required, existing code meets standards

✅ **Performance Efficiency**: Free tier sufficient for educational use case with acceptable cold start latency

### Governance Compliance

✅ **Technology Stack**: Using approved Free Tier services (Render, Gemini, Qdrant Cloud)

✅ **Zero-Tolerance Policy**: Deployment includes health checks and validation before frontend integration

⚠️ **Risk**: Cold starts may impact UX - documented and acceptable for free tier

---

## Phase 0: Pre-Deployment Validation

### Research Questions (All Resolved)

**Q1: Are current requirements.txt files complete for production?**
- ✅ Resolved: Both `backend/requirements.txt` and `backend-chatkit/requirements.txt` include all dependencies
- Evidence: Local testing successful with same requirements

**Q2: Are API keys and secrets properly managed?**
- ✅ Resolved: Keys stored in `.env` files (gitignored), will be manually configured in Render dashboard
- Security: No secrets in codebase or render.yaml

**Q3: Is render.yaml syntax valid for Blueprint deployment?**
- ✅ Resolved: Syntax follows Render Blueprint v2 specification
- Validated against: https://render.com/docs/blueprint-spec

**Q4: Are health check endpoints implemented and tested?**
- ✅ Resolved:
  - RAG Backend: `/api/health` returns `{"status":"healthy","service":"Textbook Chat API"}`
  - ChatKit Backend: `/health` returns `{"status":"ok","model":"gemini-2.5-flash-lite"}`
  - Both tested locally and return 200 OK

**Q5: Is CORS configured for production frontend domain?**
- ✅ Resolved: Both backends include `https://physai-foundations.vercel.app` in CORS allow_origins
- Location: `backend/main.py:80` and `backend-chatkit/main.py:411`

---

## Phase 1: Render Service Setup

### 1.1 Repository Preparation

**Objective**: Ensure all deployment files are committed and pushed to GitHub

**Actions**:
1. Verify `render.yaml` is in repository root
2. Commit any pending changes to backend code
3. Push to `master` branch (Render will auto-deploy from this branch)

**Validation**:
- [ ] `git status` shows clean working directory or only untracked documentation
- [ ] Latest commit includes all backend changes
- [ ] Remote repository is up to date

### 1.2 Render Account Setup

**Objective**: Connect GitHub repository to Render for Blueprint deployment

**Actions**:
1. Log into [Render Dashboard](https://dashboard.render.com/)
2. Navigate to "New +" → "Blueprint"
3. Select GitHub repository: `physai-foundations`
4. Choose branch: `master`
5. Confirm service detection (should show 2 services)

**Validation**:
- [ ] Render detects both services from `render.yaml`
- [ ] Service names match: `physai-backend-rag` and `physai-backend-chatkit`

### 1.3 Environment Variables Configuration

**Objective**: Securely configure API keys for both services

**RAG Backend Environment Variables**:
```
PYTHON_VERSION=3.11.0
PORT=10000
QDRANT_URL=<from backend/.env>
QDRANT_API_KEY=<from backend/.env>
GOOGLE_API_KEY=<from backend/.env>
```

**ChatKit Backend Environment Variables**:
```
PYTHON_VERSION=3.11.0
PORT=10000
GEMINI_API_KEY=<from backend-chatkit/.env>
QDRANT_URL=<from backend-chatkit/.env>
QDRANT_API_KEY=<from backend-chatkit/.env>
GOOGLE_API_KEY=<from backend-chatkit/.env>
```

**Actions**:
1. For each service, navigate to Settings → Environment
2. Add variables one by one (copy from local `.env` files)
3. Double-check no typos in keys or values
4. Save configuration

**Validation**:
- [ ] All required variables configured for RAG Backend (5 variables)
- [ ] All required variables configured for ChatKit Backend (5 variables)
- [ ] No placeholder values like `your_key_here` remain

---

## Phase 2: Deployment & Verification

### 2.1 Initial Deployment

**Objective**: Trigger first build and deploy for both services

**Actions**:
1. After environment variables configured, Render auto-triggers deployment
2. Monitor build logs for both services in parallel
3. Watch for common errors:
   - Missing dependencies
   - Python version mismatches
   - Import errors
   - Port binding issues

**Expected Build Time**: 5-10 minutes per service

**Validation**:
- [ ] RAG Backend build completes successfully
- [ ] ChatKit Backend build completes successfully
- [ ] Both services show "Live" status in dashboard
- [ ] No error logs in deployment output

### 2.2 Health Check Verification

**Objective**: Confirm both services are responding correctly

**Actions**:
1. Get production URLs from Render dashboard:
   - RAG Backend: `https://physai-backend-rag.onrender.com`
   - ChatKit Backend: `https://physai-backend-chatkit.onrender.com`

2. Test health endpoints:
```bash
# RAG Backend
curl https://physai-backend-rag.onrender.com/api/health

# Expected: {"status":"healthy","service":"Textbook Chat API"}

# ChatKit Backend
curl https://physai-backend-chatkit.onrender.com/health

# Expected: {"status":"ok","model":"gemini-2.5-flash-lite"}
```

**Validation**:
- [ ] RAG Backend health check returns 200 OK
- [ ] ChatKit Backend health check returns 200 OK
- [ ] Response bodies match expected JSON format
- [ ] Response time < 5 seconds (first request may be slower due to cold start)

### 2.3 Functional Testing

**Objective**: Verify RAG and ChatKit functionality in production

**RAG Backend Test**:
```bash
curl -X POST https://physai-backend-rag.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is ROS2?"}'
```

**Expected**: JSON response with AI-generated answer and RAG context

**ChatKit Backend Test**:
- Use ChatKit client or Postman to send message to `/chatkit` endpoint
- Verify streaming response works
- Check conversation persistence

**Validation**:
- [ ] RAG Backend returns relevant answers with context
- [ ] ChatKit Backend handles streaming correctly
- [ ] No 500 errors or timeout issues
- [ ] Response quality matches local testing

---

## Phase 3: Frontend Integration

### 3.1 Update Frontend Environment Variables

**Objective**: Point production frontend to production backend URLs

**File**: `frontend/.env.production` (create if doesn't exist)

**Content**:
```env
VITE_CHATKIT_BACKEND_URL=https://physai-backend-chatkit.onrender.com/chatkit
VITE_RAG_BACKEND_URL=https://physai-backend-rag.onrender.com/api/chat
```

**Actions**:
1. Create or update `frontend/.env.production`
2. Commit and push to trigger Vercel redeployment
3. Wait for Vercel deployment to complete (~2-3 minutes)

**Validation**:
- [ ] File committed to repository
- [ ] Vercel deployment successful
- [ ] Production build uses new environment variables

### 3.2 CORS Verification

**Objective**: Ensure frontend can communicate with backend

**Actions**:
1. Open production frontend: `https://physai-foundations.vercel.app`
2. Open browser DevTools Console
3. Interact with ChatWidget
4. Check for CORS errors

**Common Issues**:
- `Access-Control-Allow-Origin` errors → Update backend CORS configuration
- Preflight request failures → Check OPTIONS method handling

**Validation**:
- [ ] No CORS errors in browser console
- [ ] ChatWidget successfully connects to backend
- [ ] Messages send and receive correctly

### 3.3 End-to-End Testing

**Objective**: Validate complete user flow in production

**Test Scenarios**:
1. **New Conversation**: Open ChatWidget, send first message, verify response
2. **Conversation Persistence**: Refresh page, verify conversation loads
3. **RAG Integration**: Ask textbook-related question, verify 📚 indicator appears
4. **General Knowledge**: Ask non-textbook question, verify 💭 indicator appears
5. **Multiple Messages**: Send 3-5 messages in sequence, verify threading

**Validation**:
- [ ] All test scenarios pass
- [ ] Response times acceptable (< 5 seconds typical, < 60 seconds cold start)
- [ ] No JavaScript errors in console
- [ ] UI remains responsive during loading

---

## Phase 4: Monitoring & Documentation

### 4.1 Setup Monitoring

**Objective**: Enable alerting for service health issues

**Actions**:
1. In Render dashboard, navigate to each service → Settings → Notifications
2. Enable email alerts for:
   - Deploy failures
   - Health check failures
   - High error rates (>5%)
3. Add notification email

**Validation**:
- [ ] Email alerts configured for both services
- [ ] Test notification received (optional)

### 4.2 Document Production URLs

**Objective**: Update project documentation with production endpoints

**Files to Update**:
1. `README.md` - Add "Production" section with URLs
2. `DEPLOYMENT_GUIDE.md` - Mark deployment as complete
3. `frontend/README.md` - Update API endpoint documentation

**Content Template**:
```markdown
## Production Deployment

**Backend Services**:
- RAG API: https://physai-backend-rag.onrender.com
- ChatKit API: https://physai-backend-chatkit.onrender.com

**Frontend**:
- Live Site: https://physai-foundations.vercel.app

**Status**: ✅ Deployed and operational
```

**Validation**:
- [ ] Documentation updated with actual production URLs
- [ ] Links tested and accessible
- [ ] Status badges updated (if applicable)

### 4.3 Create Deployment Runbook

**Objective**: Document procedures for future deployments and troubleshooting

**Create File**: `docs/DEPLOYMENT_RUNBOOK.md`

**Sections**:
1. **Quick Deployment** - Step-by-step for redeployment
2. **Rollback Procedure** - How to revert to previous deploy
3. **Common Issues** - Troubleshooting guide
4. **Health Check Commands** - Quick verification scripts
5. **Monitoring Dashboards** - Links to Render logs and metrics

**Validation**:
- [ ] Runbook created and comprehensive
- [ ] Procedures tested for accuracy
- [ ] Team members can follow independently

---

## Risk Analysis & Mitigation

### Risk 1: Cold Start Latency (HIGH PROBABILITY, MEDIUM IMPACT)

**Description**: Render free tier sleeps services after 15 minutes inactivity. First request after sleep takes 30-60 seconds.

**Impact**: Poor UX for first user after idle period

**Mitigation**:
- ✅ Document expected behavior in UI ("Loading..." indicator)
- ✅ Consider implementing wake-up ping service (optional, not required for Phase 1)
- ✅ Accept as acceptable trade-off for free tier

**Owner**: Acknowledged and documented

### Risk 2: Environment Variable Misconfiguration (MEDIUM PROBABILITY, HIGH IMPACT)

**Description**: Typos in API keys or URLs cause service failures

**Impact**: Complete service failure, potential security exposure

**Mitigation**:
- ✅ Double-check all variables before deployment
- ✅ Use copy-paste from local `.env` files (no manual typing)
- ✅ Test health endpoints immediately after deployment
- ✅ Keep backup of all keys in secure location

**Owner**: Deployment checklist includes validation steps

### Risk 3: CORS Misconfiguration (LOW PROBABILITY, HIGH IMPACT)

**Description**: Frontend domain not in backend CORS allow list

**Impact**: Frontend cannot communicate with backend

**Mitigation**:
- ✅ CORS already configured in code: `backend/main.py:80`, `backend-chatkit/main.py:411`
- ✅ Includes production domain: `https://physai-foundations.vercel.app`
- ✅ Testing plan includes CORS verification step

**Owner**: Verified in Phase 3.2

### Risk 4: Build Failure Due to Missing Dependencies (LOW PROBABILITY, HIGH IMPACT)

**Description**: `requirements.txt` missing packages that work locally

**Impact**: Deployment fails, service unavailable

**Mitigation**:
- ✅ Requirements files tested locally
- ✅ No environment-specific packages used
- ✅ Render build logs monitored during deployment
- ✅ Rollback procedure documented

**Owner**: Phase 2.1 includes build log monitoring

### Risk 5: Free Tier Quota Exhaustion (VERY LOW PROBABILITY, MEDIUM IMPACT)

**Description**: Services exceed 750 hours/month per service limit

**Impact**: Service interruption until next billing cycle

**Mitigation**:
- ✅ 750 hours × 2 services = 1500 hours/month (sufficient for 24/7 operation)
- ✅ Render dashboard shows usage metrics
- ✅ Alert configured for approaching limits

**Owner**: Monitored via Render dashboard

---

## Dependencies & Prerequisites

### External Dependencies
- ✅ Render.com account (free tier) - **READY**
- ✅ GitHub repository connected to Render - **READY**
- ✅ Qdrant Cloud instance operational - **VERIFIED**
- ✅ Google Gemini API key valid - **VERIFIED**
- ✅ Vercel frontend deployed - **VERIFIED**

### Internal Dependencies
- ✅ `render.yaml` configuration - **EXISTS**
- ✅ Both `requirements.txt` files - **EXISTS**
- ✅ Health check endpoints implemented - **TESTED LOCALLY**
- ✅ CORS configuration - **VERIFIED IN CODE**
- ✅ Environment variable templates - **DOCUMENTED**

### Team Readiness
- ✅ Access to API keys and secrets - **CONFIRMED**
- ✅ Render.com dashboard access - **REQUIRED**
- ✅ GitHub repository write access - **REQUIRED**
- ✅ Deployment guide available - **EXISTS: DEPLOYMENT_GUIDE.md**

---

## Success Metrics

### Deployment Success
- ✅ Both services show "Live" status in Render dashboard
- ✅ Build time < 15 minutes per service
- ✅ Zero build errors or warnings
- ✅ Health checks return 200 OK within 5 seconds

### Functional Success
- ✅ RAG Backend responds to queries with context
- ✅ ChatKit Backend handles streaming correctly
- ✅ Frontend integrates without CORS errors
- ✅ End-to-end chat flow works in production

### Operational Success
- ✅ Services remain stable for 24 hours post-deployment
- ✅ Cold start time < 60 seconds
- ✅ No critical errors in logs
- ✅ Response time acceptable for educational use case (< 5 seconds typical)

### Documentation Success
- ✅ Production URLs documented in README
- ✅ Deployment runbook created
- ✅ Team can independently redeploy using guide

---

## Out of Scope

The following are explicitly **NOT** included in this deployment phase:

❌ **Custom Domain Configuration** - Using default `.onrender.com` domains
❌ **Paid Tier Upgrade** - Staying on free tier per constitution
❌ **Auto-Scaling Configuration** - Not available on free tier
❌ **Database Persistence** - Using in-memory store (acceptable for hackathon)
❌ **Authentication/Authorization** - Planned for Phase 2
❌ **Rate Limiting** - Not critical for educational free tier usage
❌ **Load Testing** - Beyond scope for free tier deployment
❌ **CI/CD Pipeline** - Using Render's automatic GitHub deployment
❌ **Backup/Disaster Recovery** - Code in Git is sufficient
❌ **Performance Optimization** - Current performance acceptable

---

## Next Steps After Deployment

1. **Monitor for 48 hours** - Watch for unexpected issues or errors
2. **Gather User Feedback** - Test with real students/users
3. **Optimize Cold Start** - If UX issues arise, consider wake-up service
4. **Plan Phase 2 Features**:
   - Authentication system
   - Conversation persistence (database)
   - Urdu translation
   - Personalization features

---

## Rollback Plan

If deployment fails or critical issues arise:

### Immediate Rollback (< 5 minutes)
1. In Render dashboard, navigate to service → Deploys
2. Find last working deployment (should be "None" for first deploy)
3. Click "Redeploy" on previous version
4. Notify team of rollback

### Frontend Rollback
1. Revert `frontend/.env.production` changes in Git
2. Push to trigger Vercel redeployment
3. Verify frontend returns to working state

### Communication
- Update status page (if exists)
- Notify stakeholders of issue and ETA
- Document issue for post-mortem

---

## Appendix A: Deployment Checklist

Copy this checklist when executing deployment:

**Pre-Deployment**:
- [ ] All code committed and pushed to master
- [ ] Local testing passed for both backends
- [ ] API keys accessible and valid
- [ ] Render account logged in
- [ ] GitHub repository connected to Render

**Deployment**:
- [ ] Blueprint deployment initiated
- [ ] Both services detected by Render
- [ ] Environment variables configured for RAG Backend (5 vars)
- [ ] Environment variables configured for ChatKit Backend (5 vars)
- [ ] Build logs monitored for errors
- [ ] Both services show "Live" status

**Verification**:
- [ ] RAG Backend health check passes
- [ ] ChatKit Backend health check passes
- [ ] RAG Backend functional test passes
- [ ] ChatKit Backend functional test passes
- [ ] Frontend environment variables updated
- [ ] Vercel redeployment completed
- [ ] No CORS errors in browser
- [ ] End-to-end chat flow works

**Post-Deployment**:
- [ ] Monitoring/alerts configured
- [ ] Production URLs documented
- [ ] Deployment runbook created
- [ ] Team notified of successful deployment

---

## Appendix B: Environment Variable Reference

### RAG Backend Required Variables
| Variable | Source | Example |
|----------|--------|---------|
| PYTHON_VERSION | Fixed | 3.11.0 |
| PORT | Fixed | 10000 |
| QDRANT_URL | backend/.env | https://xxx.gcp.cloud.qdrant.io |
| QDRANT_API_KEY | backend/.env | [secret] |
| GOOGLE_API_KEY | backend/.env | [secret] |

### ChatKit Backend Required Variables
| Variable | Source | Example |
|----------|--------|---------|
| PYTHON_VERSION | Fixed | 3.11.0 |
| PORT | Fixed | 10000 |
| GEMINI_API_KEY | backend-chatkit/.env | [secret] |
| QDRANT_URL | backend-chatkit/.env | https://xxx.gcp.cloud.qdrant.io |
| QDRANT_API_KEY | backend-chatkit/.env | [secret] |
| GOOGLE_API_KEY | backend-chatkit/.env | [secret] |

---

**Plan Version**: 1.0
**Last Updated**: 2025-12-27
**Status**: Ready for Implementation
