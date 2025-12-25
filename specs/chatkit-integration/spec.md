# Feature Specification: ChatKit Integration

**Feature Branch**: `feature/chatkit-integration`
**Created**: 2025-12-20
**Status**: Draft
**Input**: Replace custom ChatWidget with OpenAI ChatKit + Gemini backend integration from chatkit-gemini-bot repository

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Chat Widget Replacement (Priority: P1) 🎯 MVP

Students interact with an improved AI chat assistant that replaces the current custom implementation with OpenAI's ChatKit SDK, providing a production-grade chat experience with thread persistence and real-time streaming.

**Why this priority**: Core feature - chat is the primary student interaction point. Moving to ChatKit SDK improves reliability, UI/UX quality, and reduces maintenance burden. MVP cannot ship without this.

**Independent Test**: Can be fully tested by opening physai-foundations website, clicking floating chat button, sending messages, and verifying responses appear in real-time. Delivers working chat interface that persists conversations across page reloads.

**Acceptance Scenarios**:

1. **Given** student visits the website, **When** they click the floating chat button in bottom-right corner, **Then** the ChatKit modal opens displaying greeting and suggested prompts (ROS2, Isaac Sim, Humanoid Robotics)

2. **Given** chat modal is open, **When** student sends message "What is ROS2?", **Then** message appears in chat history and Gemini provides streaming response in real-time

3. **Given** student sends multiple messages in a chat session, **When** they refresh the page (F5), **Then** previous messages persist in the chat history (thread restoration via localStorage)

4. **Given** student clicks "New Chat" button, **When** chat widget reloads, **Then** a fresh conversation thread starts with no previous messages

5. **Given** backend service is unreachable, **When** student attempts to send a message, **Then** ChatKit UI gracefully displays error message without crashing the page

---

### User Story 2 - Backend Infrastructure (Priority: P1) 🎯 MVP

System provides a production-ready ChatKit Server backend that processes student queries through Google Gemini 2.0 Flash with proper thread management and error handling.

**Why this priority**: Core infrastructure - all chat interactions depend on this. Must be stable, fast, and properly configured for educational use. MVP blocks on this.

**Independent Test**: Can be fully tested by running backend locally, calling `/health` endpoint (verifies Gemini API connection), and sending test ChatKit protocol requests to `/chatkit` endpoint (verifies message processing and streaming).

**Acceptance Scenarios**:

1. **Given** backend service is running locally with GEMINI_API_KEY set, **When** GET request sent to `/health`, **Then** response is `{"status": "ok", "model": "gemini-2.0-flash"}`

2. **Given** ChatKit client connects to `/chatkit` endpoint on port 8001, **When** user message is sent via ChatKit protocol, **Then** backend streams response events with text/event-stream media type

3. **Given** student sends message to Gemini, **When** response is generated, **Then** message IDs are correctly mapped (LiteLLM/Gemini ID collision handling works - no duplicate messages)

4. **Given** thread with multiple messages exists, **When** GET request sent to `/debug/threads`, **Then** all stored items appear with correct thread ID and message count

5. **Given** CORS request from `https://physai-foundations.vercel.app`, **When** OPTIONS preflight sent, **Then** backend responds with proper CORS headers allowing the request

---

### User Story 3 - SSR Compatibility (Priority: P1) 🎯 MVP

Docusaurus frontend builds successfully with ChatKit integration without SSR (Server-Side Rendering) errors, enabling deployment to Vercel.

**Why this priority**: Previous ChatKit attempt failed (commits 6b51226 → 3c0ee65) due to SSR incompatibility. Must resolve this or integration fails. Critical blocker for deployment.

**Independent Test**: Can be fully tested by running `npm run build` in frontend directory - must complete without "window is not defined" or "document is not defined" errors. Successful build artifact ready for Vercel deployment.

**Acceptance Scenarios**:

1. **Given** ChatWidget is wrapped in BrowserOnly component in Root.tsx, **When** `npm run build` runs, **Then** build succeeds with no SSR-related errors

2. **Given** ChatWidget uses useState/useEffect for localStorage access, **When** SSR phase renders, **Then** component returns null (SSR-safe) preventing build errors

3. **Given** ChatKit CDN script is loaded in HTML head, **When** frontend builds, **Then** script tag is properly included without breaking SSR

4. **Given** frontend builds successfully, **When** artifact deployed to Vercel, **Then** site loads without JavaScript errors in browser console

5. **Given** student accesses site from multiple browsers, **When** chat initializes, **Then** ChatKit functionality is available and working (no missing CDN script issues)

---

### Edge Cases

- What happens when student has localStorage disabled (private/incognito mode)? System should still allow chat but won't persist thread across reloads.
- What happens when ChatKit CDN (cdn.platform.openai.com) is unavailable? Chat widget should degrade gracefully without crashing page.
- What happens when student sends 100+ messages rapidly? System should handle ID collision mapping correctly without duplicate messages.
- What happens when Gemini API returns error or timeout? Backend should return proper error response; frontend should display user-friendly message.
- What happens if backend service is down when student tries to chat? Frontend should show error in ChatKit UI, not crash the page.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide floating chat button in bottom-right corner of page (fixed positioning, visible on all pages)
- **FR-002**: System MUST allow students to send text messages and receive streaming responses from Gemini
- **FR-003**: System MUST persist conversation threads in localStorage with key `physai-chatkit-thread-id`
- **FR-004**: System MUST provide "New Chat" button to start fresh conversation thread
- **FR-005**: System MUST display suggested prompts on chat initialization (ROS2, Isaac Sim, Humanoid Robotics topics)
- **FR-006**: System MUST implement ChatKit Server protocol on `/chatkit` POST endpoint at port 8001 (separate from RAG backend on port 8000)
- **FR-007**: System MUST use Google Gemini 2.0 Flash model for all responses
- **FR-008**: System MUST handle ID collision mapping for LiteLLM/Gemini message IDs
- **FR-009**: System MUST provide `/health` endpoint that verifies Gemini API connectivity
- **FR-010**: System MUST provide `/debug/threads` endpoint for monitoring stored conversations
- **FR-011**: System MUST configure CORS to allow requests from `localhost:3000` (dev) and `https://physai-foundations.vercel.app` (production)
- **FR-012**: Frontend MUST use fresh ChatKit implementation generated via chatkit.quickstart skill (replaces broken ChatWidget.tsx with React Hooks violations)
- **FR-013**: Backend MUST be fresh ChatKit server generated via chatkit.backend skill on port 8001 (ensures frontend/backend version alignment)
- **FR-014**: System MUST build successfully with Docusaurus SSR without "window is not defined" errors
- **FR-015**: System MUST follow React Hooks Rules (hooks called unconditionally, conditional rendering only)

### Key Entities

- **ChatThread**: Represents a conversation thread with unique ID, creation timestamp, and list of messages
- **ThreadMessage**: Individual message (user or assistant) with ID, content, role (user/assistant), and timestamp
- **ChatKitConfig**: Configuration for ChatKit React component including API URL (http://localhost:8001/chatkit), domain key, theme, initial thread ID
- **BackendPortConfiguration**: ChatKit backend on port 8001, RAG backend on port 8000 (separate services)

## Clarifications

### Session 2025-12-25

- **Q1**: Which backend port configuration should be used for ChatKit backend? → A: Use different ports - ChatKit backend on 8001, RAG backend on 8000 (allows both services to run simultaneously)
- **Q2**: How should the broken ChatWidget.tsx with React Hooks violations be fixed? → C: Use chatkit.quickstart skill - generate a fresh ChatKit implementation from scratch
- **Q3**: How should the ChatKit backend be implemented? → A: Use chatkit.backend skill - generate fresh ChatKit backend on port 8001 (ensures frontend/backend version alignment)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Chat widget loads without JavaScript console errors on production website
- **SC-002**: Students can send message and receive response in under 2 seconds (p95 latency)
- **SC-003**: 95%+ of chat threads persist across page reloads (localStorage persistence success rate)
- **SC-004**: Backend handles 20+ concurrent conversations without degradation
- **SC-005**: No duplicate messages appear due to ID collision issues (mapping success rate = 100%)
- **SC-006**: Frontend build succeeds 100% of the time with no SSR errors (0 build failures)
- **SC-007**: 90%+ of chat interactions complete successfully without errors
- **SC-008**: Mobile responsiveness: chat widget works on screens < 600px width
- **SC-009**: Documentation complete and deployment process documented
- **SC-010**: Team can deploy new backend version and update frontend env var within 5 minutes (deployment simplicity)
