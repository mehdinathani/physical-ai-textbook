---
description: "Task list for ChatKit integration implementation"
---

# Tasks: ChatKit Integration

**Input**: Design documents from `/specs/chatkit-integration/`
**Prerequisites**: plan.md (required), spec.md (required for user stories)
**Tests**: Manual testing only - no automated tests requested in spec.md

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Frontend**: `frontend/src/`
- **Backend**: `backend-chatkit/`
- **Project root**: Repository root

## Phase 1: Setup (Skill Execution)

**Purpose**: Generate fresh frontend and backend using ChatKit skills

- [X] T001 Execute `/chatkit.quickstart` skill to generate React frontend with ChatKit integration
- [X] T002 [P] Execute `/chatkit.backend` skill to generate FastAPI backend with ChatKit Server protocol
- [X] T003 [P] Create `frontend/.env.local` with VITE_CHATKIT_BACKEND_URL=http://localhost:8001/chatkit
- [X] T004 [P] Create `backend-chatkit/.env.example` with GEMINI_API_KEY and PORT=8001
- [X] T005 Update `.gitignore` to include backend-chatkit/.env, backend-chatkit/venv/, frontend/.env.local

**Checkpoint**: At this point, skill-generated code is in place and ready for configuration

---

## Phase 2: Foundational (Skill Output Configuration)

**Purpose**: Configure generated code for ChatKit integration

- [X] T006 Update generated ChatWidget.tsx with ChatKit configuration:
  - API URL: http://localhost:8001/chatkit
  - Domain key: physai
  - Theme: Dark with Docusaurus blue accent (#3578e5)
- [X] T007 Verify ChatWidget follows React Hooks Rules (hooks called unconditionally, no conditional hook calls)
- [X] T008 Update generated backend-chatkit/main.py with server port 8001
- [X] T009 Update backend-chatkit assistant instructions for Physical AI & Humanoid Robotics context
- [X] T010 Verify backend CORS configuration includes localhost:3000 and https://physai-foundations.vercel.app

**Checkpoint**: Configuration complete - both frontend and backend are ready for integration and testing

---

## Phase 3: User Story 1 - Chat Widget Replacement (Priority: P1) 🎯 MVP

**Goal**: Students interact with improved AI chat assistant via production-grade ChatKit SDK with thread persistence and real-time streaming

**Independent Test**: Can be tested by opening physai-foundations website, clicking floating chat button, sending messages, and verifying responses appear in real-time

### Implementation for User Story 1

- [X] T011 [US1] Copy generated ChatWidget.tsx to `frontend/src/components/ChatWidget.tsx`
- [X] T012 [US1] Delete old `frontend/src/components/ChatWidget.tsx` (broken with React Hooks violations)
- [X] T013 [US1] Delete `frontend/src/components/SimpleChatWidget.tsx` (not needed with ChatKit)
- [X] T014 [US1] Verify `frontend/src/theme/Root.tsx` has correct ChatWidget import and BrowserOnly wrapper
- [X] T015 [US1] Verify ChatKit CDN script is loaded in Root.tsx Head component
- [X] T016 [US1] Verify ChatWidget uses VITE_CHATKIT_BACKEND_URL environment variable or default to localhost:8001
- [X] T017 [US1] Verify ChatWidget persists thread ID in localStorage with key `physai-chatkit-thread-id`
- [X] T018 [US1] Verify ChatWidget implements "New Chat" button functionality

**Checkpoint**: At this point, User Story 1 should be fully functional and independently testable

---

## Phase 4: User Story 2 - Backend Infrastructure (Priority: P1) 🎯 MVP

**Goal**: System provides production-ready ChatKit Server backend on port 8001 with proper thread management and error handling

**Independent Test**: Can be tested by running backend locally, calling /health endpoint, and sending test ChatKit protocol requests

### Implementation for User Story 2

- [X] T019 [US2] Copy generated backend to `backend-chatkit/` directory
- [X] T020 [US2] Create `backend-chatkit/requirements.txt` from skill output
- [X] T021 [US2] Verify backend-chatkit/main.py implements ChatKit Server protocol at /chatkit endpoint
- [X] T022 [US2] Verify backend uses Google Gemini 2.0 Flash model (or gemini-2.5-flash-lite)
- [X] T023 [US2] Verify backend provides /health endpoint that returns model name
- [X] T024 [US2] Verify backend provides /debug/threads endpoint for monitoring stored conversations
- [X] T025 [US2] Verify backend handles LiteLLM/Gemini message ID collision mapping

**Checkpoint**: At this point, User Story 2 should be fully functional and independently testable

---

## Phase 5: User Story 3 - SSR Compatibility (Priority: P1) 🎯 MVP

**Goal**: Docusaurus frontend builds successfully with ChatKit integration without SSR errors, enabling deployment to Vercel

**Independent Test**: Can be tested by running `npm run build` in frontend directory - must complete without "window is not defined" or "document is not defined" errors

### Implementation for User Story 3

- [X] T026 [US3] Run `npm run build` in frontend directory to verify SSR compatibility
- [X] T027 [US3] Fix any "window is not defined" errors if they occur
- [X] T028 [US3] Fix any "document is not defined" errors if they occur
- [X] T029 [US3] Fix any React Hooks violation errors if they occur
- [X] T030 [US3] Verify ChatKit CDN script loads correctly without breaking SSR

**Checkpoint**: At this point, User Story 3 should pass - build succeeds without SSR errors

---

## Phase 6: Testing & Validation

**Purpose**: Validate all three user stories work correctly together

**NOTE**: These tests require manual execution after starting services. Ready for user testing.

- [ ] T031 [P] Start RAG backend on port 8000: `cd backend && python main.py`
- [ ] T032 [P] Start ChatKit backend on port 8001: `cd backend-chatkit && python main.py`
- [ ] T033 [P] Start frontend: `cd frontend && npm start`
- [ ] T034 Verify ChatKit backend /health endpoint responds with {"status": "ok", "model": "gemini-*"}
- [ ] T035 Verify both backends can run simultaneously (RAG on 8000, ChatKit on 8001)
- [ ] T036 Test clicking floating chat button opens ChatKit modal
- [ ] T037 Test sending message "What is ROS2?" receives streaming response
- [ ] T038 Test thread persistence: refresh page, messages should persist in chat history
- [ ] T039 Test "New Chat" button creates fresh conversation thread
- [ ] T040 Verify no React Hooks violation errors in browser console
- [ ] T041 Test chat widget on screens < 600px width (mobile responsiveness)
- [ ] T042 Test backend unreachable scenario displays error message without crashing page

**Checkpoint**: At this point, all user stories should be validated and ready for deployment

---

## Phase 7: Polish & Documentation

**Purpose**: Final improvements and documentation before deployment

- [X] T043 [P] Create `start_chatkit_backend.bat` script for easy local startup
- [X] T044 [P] Create `start_all_backends.bat` script to start both RAG and ChatKit backends simultaneously
- [X] T045 Create CHATKIT_SETUP.md documenting integration and setup process
- [X] T046 Verify frontend package.json has @openai/chatkit-react dependency
- [X] T047 Verify backend-chatkit/requirements.txt has correct dependencies

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - can start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 completion
- **Phase 3 (US1)**: Depends on Phase 2 completion
- **Phase 4 (US2)**: Depends on Phase 2 completion (can run in parallel with Phase 3)
- **Phase 5 (US3)**: Depends on US1 completion (SSR test requires working frontend)
- **Phase 6 (Testing)**: Depends on Phases 1, 2, 3, 4, 5 completion
- **Phase 7 (Polish)**: Depends on Phase 6 completion

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Can run in parallel with US1
- **User Story 3 (P3)**: Depends on US1 completion (SSR test requires working frontend)

### Parallel Opportunities

- All tasks marked [P] can run in parallel:
  - T002, T003, T004, T005 (skill execution, environment setup)
  - T006, T007, T008, T009, T010 (configuration tasks)
  - T012, T013 (frontend setup can parallel backend)
  - T031, T032 (starting both backends simultaneously)
- Tasks within different user stories (US1, US2) can run in parallel by different team members

---

## Parallel Example: Setup Phase

```bash
# Launch skill execution in parallel:
Task: "Execute /chatkit.quickstart skill"
Task: "Execute /chatkit.backend skill"
Task: "Create frontend/.env.local"
Task: "Create backend-chatkit/.env.example"
Task: "Update .gitignore"
```

---

## Parallel Example: Testing Phase

```bash
# Launch both backends simultaneously:
Task: "Start RAG backend on port 8000"
Task: "Start ChatKit backend on port 8001"
Task: "Start frontend"
```

---

## Implementation Strategy

### MVP First (User Stories 1, 2, 3 Together)

1. Complete Phase 1: Setup (skill execution)
2. Complete Phase 2: Foundational (configuration)
3. Complete Phase 3: User Story 1 - Chat Widget Replacement
4. Complete Phase 4: User Story 2 - Backend Infrastructure
5. Complete Phase 5: User Story 3 - SSR Compatibility
6. Complete Phase 6: Testing & Validation
7. Complete Phase 7: Polish & Documentation
8. **STOP AND VALIDATE**: Test all three user stories together

### Incremental Delivery

Each user story adds value without breaking previous stories:
- US1 (P1): Delivers production-grade ChatKit UI with thread persistence
- US2 (P1): Delivers working ChatKit backend on separate port 8001
- US3 (P1): Ensures SSR compatibility for Vercel deployment

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Skills (chatkit.quickstart, chatkit.backend) generate production-ready code
- Port separation (8000 for RAG, 8001 for ChatKit) prevents conflicts
- React Hooks compliance is critical - no conditional hook calls allowed
- SSR compatibility is validated via npm run build
- localStorage key: physai-chatkit-thread-id (matches spec requirement)
- Environment variable: VITE_CHATKIT_BACKEND_URL (Vite requires VITE_ prefix)
