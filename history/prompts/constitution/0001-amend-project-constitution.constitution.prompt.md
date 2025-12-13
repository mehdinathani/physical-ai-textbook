---
id: 0001
title: Amend Project Constitution
stage: constitution
date: 2025-12-13
surface: agent
model: claude-haiku-4-5-20251001
feature: none
branch: master
user:
command: /sp.constitution
labels: [constitution, governance, hackathon]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
/sp.constitution

Project: Physical AI & Humanoid Robotics Textbook (Hackathon I)

Core principles:
- **Spec-Driven Execution:** All code changes must originate from updated `specs/` files (Plan -> Task -> Implement).
- **Bilingual Excellence:** Strict separation of concerns: TypeScript for Frontend (Docusaurus) and Python for Backend (FastAPI).
- **Hackathon Velocity:** Prioritize "Working Deployment" over "Perfect Abstraction."
- **AI-Native First:** Leverage LLMs (Gemini/OpenAI) for both content generation and the RAG intelligence layer.

Key standards:
- **Code Quality:** Modular, commented, and lint-free.
- **Content Rigor:** Textbook content must accurately reflect official ROS 2, NVIDIA Isaac, and VLA documentation.
- **Deployment:** Zero-tolerance for build failures on Vercel (Frontend) or Render (Backend).
- **User Experience:** Chat widgets must be unobtrusive, responsive, and Z-index optimized.

Constraints:
- **Cost:** Must utilize Free Tier services only (Google Gemini, Qdrant Cloud Free, Render Free).
- **Tech Stack:** Docusaurus (UI), FastAPI (API), Qdrant (Vector DB).
- **Performance:** Chatbot response latency under 3 seconds.

Success criteria:
- **Base Score:** 100/100 (Live Book + Functional RAG Chatbot).
- **Bonus Score:** +100 Points secured via Urdu Translation and Content Personalization features.
- **Live Demo:** Fully functional URLs for both Book and Backend API.
```

## Response snapshot

Updated the project constitution to incorporate the Physical AI & Humanoid Robotics Textbook hackathon principles while maintaining the structure and key elements. The updated constitution now includes:

- Spec-Driven Execution: All code changes must originate from updated `specs/` files
- Bilingual Excellence: Strict separation of TypeScript for Frontend and Python for Backend
- Hackathon Velocity: Prioritize "Working Deployment" over "Perfect Abstraction"
- AI-Native First: Leverage LLMs for content generation and RAG intelligence layer
- Updated content requirements to reflect ROS 2, NVIDIA Isaac, and VLA documentation accuracy
- Technology stack updated to include Docusaurus, FastAPI, Qdrant, and Free Tier service constraints
- Performance requirement for chatbot response latency under 3 seconds
- Zero-tolerance for build failures on Vercel and Render

Version incremented from 1.0.0 to 1.1.0 with the changes.

## Outcome

- ✅ Impact: Updated constitution to align with hackathon requirements and project goals
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with new principles and requirements
- 🔁 Next prompts: Continue with project implementation following updated constitution
- 🧠 Reflection: The constitution now properly captures the hackathon's core principles and technical requirements

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin implementing features according to the updated constitution