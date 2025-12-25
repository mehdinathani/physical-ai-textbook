---
id: 0018
title: ChatKit Implementation Plan
stage: plan
date: 2025-12-25
surface: agent
model: claude-opus-4-5-20251101
feature: chatkit-integration
branch: master
user: unknown
command: /sp.plan
labels: [chatkit, plan, architecture, skills]
links:
  spec: specs/chatkit-integration/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/chatkit-integration/plan.md
tests:
 - N/A
---

## Prompt

(no user arguments provided - proceeding with clarified spec)

## Response snapshot

Updated implementation plan for ChatKit integration based on clarifications from previous session. Key changes from original plan:

**Clarifications Applied:**
1. ChatKit backend on port 8001 (separate from RAG backend on 8000)
2. Use chatkit.quickstart skill for fresh frontend (instead of manual rewrite)
3. Use chatkit.backend skill for fresh backend (instead of copying existing broken code)

**Plan Structure:**
- Phase 0: Skill Execution (chatkit.quickstart + chatkit.backend)
- Phase 1: Frontend Integration (Root.tsx, environment vars)
- Phase 2: Backend Configuration (port 8001, environment setup)
- Phase 3: Testing Strategy (SSR build, dual-backend verification)
- Phase 4: Deployment (future Vercel/Render setup)

**Architecture Decisions:**
- Dual backend architecture (RAG on 8000, ChatKit on 8001)
- Skill-generated code for consistency (chatkit.quickstart/chatkit.backend)
- React Hooks compliance enforced (no conditional hook calls)

**Next Steps:**
1. Execute `/chatkit.quickstart` to generate frontend
2. Execute `/chatkit.backend` to generate backend
3. Run `/sp.tasks` to create task list
4. Run `/sp.implement` to execute implementation

## Outcome

- Impact: Plan updated with skill-based approach, port separation, and clear implementation phases
- Tests: N/A
- Files: specs/chatkit-integration/plan.md (complete rewrite based on clarifications)
- Next prompts: /sp.tasks
- Reflection: Using skills reduces risk of repeating previous React Hooks violations and SSR incompatibility issues

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Run chatkit.quickstart skill to generate frontend
