# Implementation Tasks: Render Backend Deployment

**Feature**: Deploy PhysAI Foundations dual backend architecture to Render.com
**Branch**: master
**Plan**: specs/master/plan.md
**Status**: Ready for Implementation
**Created**: 2025-12-27

---

## Task Overview

Deploy both backend services (RAG + ChatKit) to Render.com free tier with complete configuration, testing, and frontend integration.

**Estimated Effort**: 2-3 hours (includes monitoring first deployment)
**Dependencies**: Render.com account, API keys, GitHub repository access

---

## Phase 1: Pre-Deployment Preparation

### Task 1.1: Verify Repository State
**Priority**: CRITICAL
**Estimated Time**: 5 minutes

**Acceptance Criteria**:
- [ ] `git status` shows clean working directory or only documentation files
- [ ] `render.yaml` exists at repository root
- [ ] Both `backend/requirements.txt` and `backend-chatkit/requirements.txt` exist
- [ ] Latest changes committed to master branch
- [ ] Remote repository up to date with local

**Implementation**:
```bash
# Check repository status
git status

# If there are uncommitted backend changes, commit them
git add backend/ backend-chatkit/ render.yaml
git commit -m "chore: Prepare backends for Render deployment"

# Push to remote
git push origin master

# Verify push successful
git log origin/master -1
```

**Test Cases**:
1. Run `git status` → Should show "nothing to commit, working tree clean" or only untracked docs
2. Run `git log origin/master..master` → Should show no commits (everything pushed)
3. Verify `render.yaml` in root: `ls render.yaml` → File exists

**Definition of Done**:
- All backend code pushed to GitHub master branch
- `render.yaml` present and accessible
- No uncommitted changes to deployment-critical files

---

### Task 1.2: Collect Environment Variables
**Priority**: CRITICAL
**Estimated Time**: 10 minutes

**Acceptance Criteria**:
- [ ] All API keys accessible from local `.env` files
- [ ] Keys documented in temporary secure note for deployment
- [ ] Qdrant URL and API key verified working
- [ ] Gemini/Google API keys verified working
- [ ] Backup of keys stored securely

**Implementation**:
```bash
# Display RAG Backend environment variables (DO NOT commit output)
cat backend/.env

# Display ChatKit Backend environment variables (DO NOT commit output)
cat backend-chatkit/.env

# Verify Qdrant connection (optional verification)
# curl -H "api-key: YOUR_QDRANT_KEY" https://YOUR_QDRANT_URL/collections
```

**Environment Variables Checklist**:

**RAG Backend** (`backend/.env`):
- [ ] QDRANT_URL
- [ ] QDRANT_API_KEY
- [ ] GOOGLE_API_KEY

**ChatKit Backend** (`backend-chatkit/.env`):
- [ ] GEMINI_API_KEY
- [ ] QDRANT_URL
- [ ] QDRANT_API_KEY
- [ ] GOOGLE_API_KEY

**Test Cases**:
1. Verify each `.env` file exists and is readable
2. Confirm no environment variable contains placeholder text like "your_key_here"
3. Test Qdrant connection locally to verify credentials work

**Definition of Done**:
- All required API keys documented and accessible
- Keys verified to be non-placeholder values
- Secure temporary storage for deployment process

---

### Task 1.3: Verify Local Health Checks
**Priority**: HIGH
**Estimated Time**: 5 minutes

**Acceptance Criteria**:
- [ ] RAG Backend `/api/health` returns 200 OK
- [ ] ChatKit Backend `/health` returns 200 OK
- [ ] Response bodies match expected JSON format
- [ ] Both backends respond within 2 seconds locally

**Implementation**:
```bash
# Ensure both backends are running (from previous session)
# If not, start them:
# Terminal 1: cd backend && python -m uvicorn main:app --port 8001
# Terminal 2: cd backend-chatkit && python -m uvicorn main:app --port 8000

# Test RAG Backend health
curl http://localhost:8001/api/health
# Expected: {"status":"healthy","service":"Textbook Chat API"}

# Test ChatKit Backend health
curl http://localhost:8000/health
# Expected: {"status":"ok","model":"gemini-2.5-flash-lite"}
```

**Test Cases**:
1. RAG health check returns HTTP 200 with correct JSON
2. ChatKit health check returns HTTP 200 with correct JSON
3. Both endpoints respond without errors
4. Response time acceptable (<2 seconds)

**Definition of Done**:
- Both health endpoints tested and working locally
- Response formats validated
- Confidence that code is deployment-ready

---

## Phase 2: Render Service Setup

### Task 2.1: Create Render Blueprint Deployment
**Priority**: CRITICAL
**Estimated Time**: 15 minutes

**Acceptance Criteria**:
- [ ] Logged into Render dashboard
- [ ] GitHub repository connected to Render
- [ ] Blueprint deployment created from `render.yaml`
- [ ] Both services detected: `physai-backend-rag` and `physai-backend-chatkit`
- [ ] Services in "Waiting for environment variables" or "Ready to deploy" state

**Implementation**:
1. Navigate to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** button in top-right
3. Select **"Blueprint"** from dropdown
4. Connect GitHub account (if not already connected)
5. Select repository: `physai-foundations` (or your repo name)
6. Choose branch: `master`
7. Click **"Apply"**
8. Render will parse `render.yaml` and show 2 services
9. Review service configuration, click **"Create Blueprint"**

**Test Cases**:
1. Render detects 2 services from `render.yaml`
2. Service names match: `physai-backend-rag` and `physai-backend-chatkit`
3. Build commands correctly set for each service
4. Health check paths configured: `/api/health` and `/health`

**Definition of Done**:
- Blueprint created successfully
- Both services visible in Render dashboard
- Services awaiting environment variables configuration

---

### Task 2.2: Configure RAG Backend Environment Variables
**Priority**: CRITICAL
**Estimated Time**: 10 minutes

**Acceptance Criteria**:
- [ ] All 5 environment variables configured correctly
- [ ] No typos in variable names or values
- [ ] Values copied (not manually typed) from local `.env`
- [ ] Configuration saved successfully

**Implementation**:
1. In Render dashboard, click on **`physai-backend-rag`** service
2. Navigate to **Settings** → **Environment** tab
3. Add each variable one by one:

**Variables to Add**:
```
PYTHON_VERSION = 3.11.0
PORT = 10000
QDRANT_URL = <paste from backend/.env>
QDRANT_API_KEY = <paste from backend/.env>
GOOGLE_API_KEY = <paste from backend/.env>
```

4. For each variable:
   - Click **"Add Environment Variable"**
   - Enter **Key** exactly as shown (case-sensitive)
   - Paste **Value** from your local `backend/.env` file
   - Click **"Save"**

5. After all variables added, click **"Save Changes"** at bottom

**Test Cases**:
1. Variable count = 5 (PYTHON_VERSION, PORT, QDRANT_URL, QDRANT_API_KEY, GOOGLE_API_KEY)
2. No variables with empty values
3. QDRANT_URL starts with `https://`
4. API keys are long alphanumeric strings (not placeholder text)

**Definition of Done**:
- All 5 variables configured in Render dashboard
- Values match local `.env` file exactly
- Configuration saved without errors

---

### Task 2.3: Configure ChatKit Backend Environment Variables
**Priority**: CRITICAL
**Estimated Time**: 10 minutes

**Acceptance Criteria**:
- [ ] All 5 environment variables configured correctly
- [ ] No typos in variable names or values
- [ ] Values copied (not manually typed) from local `.env`
- [ ] Configuration saved successfully

**Implementation**:
1. In Render dashboard, click on **`physai-backend-chatkit`** service
2. Navigate to **Settings** → **Environment** tab
3. Add each variable one by one:

**Variables to Add**:
```
PYTHON_VERSION = 3.11.0
PORT = 10000
GEMINI_API_KEY = <paste from backend-chatkit/.env>
QDRANT_URL = <paste from backend-chatkit/.env>
QDRANT_API_KEY = <paste from backend-chatkit/.env>
GOOGLE_API_KEY = <paste from backend-chatkit/.env>
```

4. For each variable:
   - Click **"Add Environment Variable"**
   - Enter **Key** exactly as shown (case-sensitive)
   - Paste **Value** from your local `backend-chatkit/.env` file
   - Click **"Save"**

5. After all variables added, click **"Save Changes"** at bottom

**Test Cases**:
1. Variable count = 5 (PYTHON_VERSION, PORT, GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY, GOOGLE_API_KEY)
2. No variables with empty values
3. QDRANT_URL starts with `https://`
4. API keys are long alphanumeric strings (not placeholder text)
5. GEMINI_API_KEY and GOOGLE_API_KEY may be the same value (both work with Gemini)

**Definition of Done**:
- All 5 variables configured in Render dashboard
- Values match local `.env` file exactly
- Configuration saved without errors

---

## Phase 3: Deployment & Verification

### Task 3.1: Monitor Initial Build - RAG Backend
**Priority**: CRITICAL
**Estimated Time**: 10-15 minutes

**Acceptance Criteria**:
- [ ] Build triggered automatically after environment variables configured
- [ ] Build logs show successful dependency installation
- [ ] Python 3.11 runtime detected
- [ ] Uvicorn starts successfully on port 10000
- [ ] Service reaches "Live" status
- [ ] No errors in build logs

**Implementation**:
1. After configuring environment variables, build auto-triggers
2. In Render dashboard, click on `physai-backend-rag` service
3. Navigate to **"Logs"** tab
4. Watch build progress in real-time
5. Look for key success indicators:
   - `Installing dependencies from backend/requirements.txt`
   - `Successfully installed <packages>`
   - `Starting server...`
   - `Uvicorn running on http://0.0.0.0:10000`
   - Service status changes to **"Live"**

**Common Issues to Watch For**:
- `ModuleNotFoundError` → Missing dependency in requirements.txt
- `Port already in use` → Should not happen on Render
- `Import error` → Code issue, verify locally first
- `Authentication failed` → Check API keys configured correctly

**Test Cases**:
1. Build completes without errors (exit code 0)
2. Service status shows "Live" (green indicator)
3. Logs show "Uvicorn running on..."
4. No Python exceptions in logs

**Definition of Done**:
- RAG Backend build completed successfully
- Service shows "Live" status in dashboard
- No error messages in logs
- Service URL accessible (will test in next task)

---

### Task 3.2: Monitor Initial Build - ChatKit Backend
**Priority**: CRITICAL
**Estimated Time**: 10-15 minutes

**Acceptance Criteria**:
- [ ] Build triggered automatically after environment variables configured
- [ ] Build logs show successful dependency installation
- [ ] Python 3.11 runtime detected
- [ ] Uvicorn starts successfully on port 10000
- [ ] Service reaches "Live" status
- [ ] No errors in build logs

**Implementation**:
1. After configuring environment variables, build auto-triggers
2. In Render dashboard, click on `physai-backend-chatkit` service
3. Navigate to **"Logs"** tab
4. Watch build progress in real-time
5. Look for key success indicators:
   - `Installing dependencies from backend-chatkit/requirements.txt`
   - `Successfully installed <packages>`
   - `Starting server...`
   - `Uvicorn running on http://0.0.0.0:10000`
   - Service status changes to **"Live"**

**Common Issues to Watch For**:
- `ModuleNotFoundError` → Missing dependency in requirements.txt
- `google.generativeai deprecated warning` → Expected, non-blocking
- `Import error` → Code issue, verify locally first
- `Authentication failed` → Check API keys configured correctly

**Test Cases**:
1. Build completes without errors (exit code 0)
2. Service status shows "Live" (green indicator)
3. Logs show "Uvicorn running on..."
4. Deprecation warning about google.generativeai is acceptable

**Definition of Done**:
- ChatKit Backend build completed successfully
- Service shows "Live" status in dashboard
- No critical error messages in logs
- Service URL accessible (will test in next task)

---

### Task 3.3: Test Production Health Endpoints
**Priority**: CRITICAL
**Estimated Time**: 5 minutes

**Acceptance Criteria**:
- [ ] RAG Backend health endpoint returns 200 OK
- [ ] ChatKit Backend health endpoint returns 200 OK
- [ ] Response bodies match expected JSON format
- [ ] Response time < 60 seconds (cold start acceptable)

**Implementation**:
1. Get production URLs from Render dashboard:
   - Click on each service to see URL (e.g., `https://physai-backend-rag.onrender.com`)

2. Test RAG Backend:
```bash
curl https://physai-backend-rag.onrender.com/api/health
```
**Expected Response**:
```json
{"status":"healthy","service":"Textbook Chat API"}
```

3. Test ChatKit Backend:
```bash
curl https://physai-backend-chatkit.onrender.com/health
```
**Expected Response**:
```json
{"status":"ok","model":"gemini-2.5-flash-lite"}
```

**Test Cases**:
1. Both endpoints return HTTP 200 status code
2. Response bodies are valid JSON
3. Response content matches expected format
4. No CORS errors (curl doesn't check CORS, browser testing in later task)

**Troubleshooting**:
- **503 Service Unavailable** → Service still starting, wait 30 seconds and retry
- **404 Not Found** → Check URL and health check path configuration
- **500 Internal Server Error** → Check Render logs for Python errors
- **Timeout** → Service may be sleeping, first request can take 30-60 seconds

**Definition of Done**:
- Both production health endpoints tested successfully
- Responses match expected format
- Services confirmed operational

---

### Task 3.4: Test Production Functionality - RAG Backend
**Priority**: HIGH
**Estimated Time**: 5 minutes

**Acceptance Criteria**:
- [ ] RAG Backend `/api/chat` endpoint accepts POST requests
- [ ] Returns AI-generated response
- [ ] Response includes textbook context when relevant
- [ ] No 500 errors or exceptions

**Implementation**:
```bash
# Test RAG Backend with sample question
curl -X POST https://physai-backend-rag.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is ROS2?"}'
```

**Expected Response** (example):
```json
{
  "response": "ROS2 (Robot Operating System 2) is...",
  "context": [...],
  "success": true
}
```

**Test Cases**:
1. Endpoint returns HTTP 200
2. Response is valid JSON
3. Response contains answer to question
4. No timeout errors (should respond within 10 seconds)

**Definition of Done**:
- RAG endpoint functionally tested
- AI response generation working
- No critical errors

---

### Task 3.5: Test Production Functionality - ChatKit Backend
**Priority**: HIGH
**Estimated Time**: 10 minutes

**Acceptance Criteria**:
- [ ] ChatKit Backend `/chatkit` endpoint accepts POST requests
- [ ] Streaming response works correctly
- [ ] Thread persistence functions
- [ ] No 500 errors or exceptions

**Implementation**:
**Option A: Using Frontend (Recommended)**:
1. Update frontend `.env.local` temporarily to point to production:
```env
VITE_CHATKIT_BACKEND_URL=https://physai-backend-chatkit.onrender.com/chatkit
```
2. Restart frontend: `cd frontend && npm start`
3. Open http://localhost:3000
4. Open ChatWidget
5. Send test message: "Hello"
6. Verify response received

**Option B: Using Curl (Advanced)**:
```bash
# ChatKit uses Server-Sent Events (SSE), curl testing is complex
# Recommend using frontend or Postman for full test
```

**Test Cases**:
1. ChatWidget connects successfully
2. Message sends without errors
3. Streaming response displays properly
4. Thread ID persists in localStorage
5. Second message in same thread works

**Definition of Done**:
- ChatKit endpoint functionally tested
- Streaming and thread persistence working
- No critical errors

---

## Phase 4: Frontend Integration

### Task 4.1: Create Frontend Production Environment File
**Priority**: HIGH
**Estimated Time**: 5 minutes

**Acceptance Criteria**:
- [ ] File `frontend/.env.production` created
- [ ] Contains production backend URLs
- [ ] URLs match actual Render service URLs
- [ ] File committed to repository

**Implementation**:
1. Create file `frontend/.env.production`:
```env
VITE_CHATKIT_BACKEND_URL=https://physai-backend-chatkit.onrender.com/chatkit
VITE_RAG_BACKEND_URL=https://physai-backend-rag.onrender.com/api/chat
```

2. Replace the placeholder URLs with actual URLs from Render dashboard

3. Commit the file:
```bash
git add frontend/.env.production
git commit -m "feat: Add production backend URLs for Vercel deployment"
git push origin master
```

**Test Cases**:
1. File exists at `frontend/.env.production`
2. Contains both VITE_CHATKIT_BACKEND_URL and VITE_RAG_BACKEND_URL
3. URLs are HTTPS and end with `.onrender.com`
4. File committed to git

**Definition of Done**:
- Production environment file created
- URLs configured correctly
- File pushed to GitHub (triggers Vercel redeploy)

---

### Task 4.2: Verify Vercel Redeployment
**Priority**: HIGH
**Estimated Time**: 5 minutes (wait for deployment)

**Acceptance Criteria**:
- [ ] Vercel detected GitHub push and triggered build
- [ ] Build completed successfully
- [ ] Production site updated with new environment variables
- [ ] No build errors

**Implementation**:
1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Find your project: `physai-foundations`
3. Check **"Deployments"** tab
4. Wait for latest deployment to complete (2-3 minutes)
5. Verify status shows "Ready"

**Test Cases**:
1. Latest commit matches your environment file commit
2. Build logs show no errors
3. Deployment status is "Ready" (not "Failed")
4. Preview URL works

**Definition of Done**:
- Vercel redeployment completed successfully
- Production site updated
- New environment variables active

---

### Task 4.3: Test Production End-to-End Flow
**Priority**: CRITICAL
**Estimated Time**: 10 minutes

**Acceptance Criteria**:
- [ ] Production frontend connects to production ChatKit backend
- [ ] No CORS errors in browser console
- [ ] ChatWidget sends and receives messages
- [ ] RAG context indicator (📚) appears for textbook questions
- [ ] Conversation persists across page refresh
- [ ] No JavaScript errors

**Implementation**:
1. Open production site: `https://physai-foundations.vercel.app`
2. Open browser DevTools (F12) → Console tab
3. Open ChatWidget (click chat button)
4. Send test message: "Hello"
5. Verify response received without errors
6. Send textbook question: "What is ROS2?"
7. Verify 📚 indicator appears (RAG context used)
8. Refresh page
9. Verify conversation persists (thread loaded from localStorage)

**Test Scenarios**:

**Scenario 1: First Message**
- Action: Send "Hello"
- Expected: Response with 💭 indicator (general knowledge)
- Verify: No CORS errors, response within 10 seconds

**Scenario 2: Textbook Question**
- Action: Send "What is ROS2?"
- Expected: Response with 📚 indicator (RAG context)
- Verify: Response includes relevant textbook content

**Scenario 3: Conversation Persistence**
- Action: Refresh browser
- Expected: Previous messages still visible
- Verify: Thread ID persists in localStorage

**Scenario 4: Multiple Messages**
- Action: Send 3 messages in sequence
- Expected: All responses received correctly
- Verify: Threading maintains order

**CORS Verification**:
- Check DevTools Console for any red errors
- Should NOT see: "Access-Control-Allow-Origin"
- Should NOT see: "CORS policy blocked"

**Test Cases**:
1. ChatWidget opens without errors
2. First message sends and receives response
3. RAG integration works (📚 indicator)
4. Conversation persists after refresh
5. No CORS errors in console
6. No JavaScript exceptions in console

**Definition of Done**:
- Complete end-to-end flow tested in production
- All test scenarios pass
- No errors in browser console
- Chat functionality confirmed working

---

## Phase 5: Post-Deployment

### Task 5.1: Configure Monitoring Alerts
**Priority**: MEDIUM
**Estimated Time**: 10 minutes

**Acceptance Criteria**:
- [ ] Email alerts configured for RAG Backend
- [ ] Email alerts configured for ChatKit Backend
- [ ] Alerts enabled for deploy failures
- [ ] Alerts enabled for health check failures

**Implementation**:
1. In Render dashboard, navigate to `physai-backend-rag`
2. Go to **Settings** → **Notifications**
3. Enable email notifications:
   - ✅ Deploy failures
   - ✅ Health check failures
4. Enter notification email
5. Click **"Save"**
6. Repeat for `physai-backend-chatkit`

**Test Cases**:
1. Both services have notifications configured
2. Email address is correct and accessible
3. Deploy failure alerts enabled
4. Health check failure alerts enabled

**Definition of Done**:
- Monitoring alerts configured for both services
- Email notifications enabled
- Alert configuration saved

---

### Task 5.2: Document Production URLs
**Priority**: MEDIUM
**Estimated Time**: 10 minutes

**Acceptance Criteria**:
- [ ] README.md updated with production section
- [ ] Production URLs documented
- [ ] Status badge added (optional)
- [ ] Documentation committed to repository

**Implementation**:
1. Update `README.md` with production deployment section
2. Add production URLs from Render dashboard
3. Mark deployment as complete

**Content to Add** (example):
```markdown
## 🚀 Production Deployment

**Live Site**: [https://physai-foundations.vercel.app](https://physai-foundations.vercel.app)

**Backend Services**:
- RAG API: [https://physai-backend-rag.onrender.com](https://physai-backend-rag.onrender.com)
- ChatKit API: [https://physai-backend-chatkit.onrender.com](https://physai-backend-chatkit.onrender.com)

**Health Checks**:
- RAG: `curl https://physai-backend-rag.onrender.com/api/health`
- ChatKit: `curl https://physai-backend-chatkit.onrender.com/health`

**Status**: ✅ Deployed and operational (Free Tier)

**Note**: Services may experience cold starts (~30-60s) after 15 minutes of inactivity due to free tier limits.
```

3. Commit changes:
```bash
git add README.md
git commit -m "docs: Add production deployment information"
git push origin master
```

**Test Cases**:
1. README contains production section
2. All URLs are correct and clickable
3. Health check commands provided
4. Cold start behavior documented

**Definition of Done**:
- README.md updated with production information
- Documentation committed and pushed
- Team can access production URLs from README

---

### Task 5.3: Create Deployment Runbook
**Priority**: MEDIUM
**Estimated Time**: 15 minutes

**Acceptance Criteria**:
- [ ] Runbook file created at `docs/DEPLOYMENT_RUNBOOK.md`
- [ ] Contains quick deployment steps
- [ ] Contains rollback procedure
- [ ] Contains troubleshooting guide
- [ ] Committed to repository

**Implementation**:
Create `docs/DEPLOYMENT_RUNBOOK.md` with the following sections:

```markdown
# Deployment Runbook

## Quick Redeployment
1. Make code changes locally
2. Commit and push to master: `git push origin master`
3. Render auto-detects push and redeploys (5-10 min)
4. Monitor build logs in Render dashboard
5. Test health endpoints after deploy

## Rollback Procedure
If deployment fails:
1. Go to Render dashboard → Service → Deploys
2. Find last working deployment
3. Click "Redeploy" button
4. Verify health checks pass

## Health Check Commands
```bash
# RAG Backend
curl https://physai-backend-rag.onrender.com/api/health

# ChatKit Backend
curl https://physai-backend-chatkit.onrender.com/health
```

## Common Issues
### Issue: 503 Service Unavailable
- **Cause**: Service sleeping (free tier)
- **Fix**: Wait 30-60 seconds for cold start

### Issue: Health check fails
- **Cause**: Build error or env var misconfiguration
- **Fix**: Check Render logs, verify environment variables

### Issue: CORS errors on frontend
- **Cause**: Frontend domain not in CORS whitelist
- **Fix**: Update `backend/main.py:80` and `backend-chatkit/main.py:411`

## Monitoring
- **Dashboard**: https://dashboard.render.com/
- **Logs**: Service → Logs tab
- **Metrics**: Service → Metrics tab
```

**Test Cases**:
1. File created at `docs/DEPLOYMENT_RUNBOOK.md`
2. Contains all required sections
3. Commands are copy-paste ready
4. File committed to git

**Definition of Done**:
- Runbook created and comprehensive
- Team can follow procedures independently
- File committed to repository

---

## Phase 6: Validation & Completion

### Task 6.1: Run Full Deployment Checklist
**Priority**: CRITICAL
**Estimated Time**: 10 minutes

**Acceptance Criteria**:
- [ ] All checklist items verified
- [ ] No outstanding issues
- [ ] Deployment considered successful
- [ ] Team notified of completion

**Implementation**:
Run through complete checklist from `specs/master/plan.md` Appendix A:

**Pre-Deployment**:
- [x] All code committed and pushed to master
- [x] Local testing passed for both backends
- [x] API keys accessible and valid
- [x] Render account logged in
- [x] GitHub repository connected to Render

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

**Test Cases**:
1. All checklist items marked complete
2. No pending action items
3. All health checks passing
4. Production site fully functional

**Definition of Done**:
- Complete checklist verified
- All critical items passed
- Deployment declared successful

---

### Task 6.2: 24-Hour Stability Monitoring
**Priority**: MEDIUM
**Estimated Time**: 5 minutes/day for 2 days

**Acceptance Criteria**:
- [ ] Services remain stable for 24 hours
- [ ] No unexpected errors in logs
- [ ] Health checks consistently pass
- [ ] User testing feedback collected (if applicable)

**Implementation**:
**Day 1 (0-24 hours post-deployment)**:
- Check Render dashboard once every 6-8 hours
- Verify both services show "Live" status
- Scan logs for any error messages
- Test health endpoints: RAG and ChatKit
- Monitor Vercel deployment status

**Day 2 (24-48 hours post-deployment)**:
- Final check of Render dashboard
- Review any email alerts received
- Test production site functionality
- Collect any user feedback
- Document any issues or optimizations needed

**Monitoring Checklist**:
- [ ] Hour 0: Initial deployment verified
- [ ] Hour 8: Check service status and logs
- [ ] Hour 16: Check service status and logs
- [ ] Hour 24: Full health check test
- [ ] Hour 48: Final stability confirmation

**Test Cases**:
1. No service restarts or crashes
2. No critical errors in logs
3. Health checks pass consistently
4. User interactions work smoothly

**Definition of Done**:
- Services stable for 24+ hours
- No critical issues identified
- Deployment considered production-ready

---

## Success Criteria Summary

### Critical Success Criteria (Must Pass)
- ✅ Both backends deployed to Render and showing "Live" status
- ✅ Health endpoints return 200 OK for both services
- ✅ Production frontend connects to production backends
- ✅ End-to-end chat flow works without errors
- ✅ No CORS errors in browser console
- ✅ Conversation persistence works correctly

### Important Success Criteria (Should Pass)
- ✅ Build time < 15 minutes per service
- ✅ Response time < 10 seconds (excluding cold starts)
- ✅ No Python exceptions in Render logs
- ✅ Monitoring alerts configured
- ✅ Documentation updated with production URLs

### Optional Success Criteria (Nice to Have)
- ✅ Cold start time < 60 seconds
- ✅ Deployment runbook created
- ✅ 24-hour stability monitoring completed
- ✅ User feedback collected

---

## Risk Mitigation Checklist

- [ ] **Cold Start Risk**: Documented in UI, acceptable for free tier
- [ ] **Environment Variable Risk**: Double-checked all keys before deployment
- [ ] **CORS Risk**: Verified configuration in code before deployment
- [ ] **Build Failure Risk**: Local testing passed, requirements.txt complete
- [ ] **Free Tier Quota Risk**: Usage monitoring enabled in Render dashboard

---

## Rollback Triggers

Immediately rollback if:
- ❌ Both services fail to build
- ❌ Health checks fail for more than 10 minutes
- ❌ Critical errors in production logs
- ❌ Complete frontend-backend communication failure
- ❌ Data loss or security breach detected

Monitor and consider rollback if:
- ⚠️ Intermittent 500 errors (>10% of requests)
- ⚠️ Response time >30 seconds consistently
- ⚠️ Services restart more than 3 times in 1 hour
- ⚠️ User-reported critical bugs

---

## Post-Deployment Next Steps

After successful deployment:
1. Gather user feedback from early testers
2. Monitor performance metrics for 1 week
3. Identify optimization opportunities
4. Plan Phase 2 features (auth, persistence, Urdu translation)
5. Consider implementing wake-up service if cold starts become UX issue

---

**Tasks Version**: 1.0
**Last Updated**: 2025-12-27
**Status**: Ready for Execution
**Total Estimated Time**: 2-3 hours (excluding 24-hour monitoring)
