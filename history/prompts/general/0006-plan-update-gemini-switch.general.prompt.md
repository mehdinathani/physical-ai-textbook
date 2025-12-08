---
id: 6
title: Plan Update Gemini Switch
stage: plan
date: 2025-12-07
surface: agent
model: claude-haiku-4-5-20251001
feature: physai-foundations
branch: master
user: mehdinathani
command: manual
labels: ["plan", "migration", "gemini", "embeddings", "api"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - backend/requirements.txt
 - backend/main.py
 - backend/ingest.py
 - backend/.env.example
 - .env
tests:
 - none
---

## Prompt

UPDATE: Switching AI Provider to Google Gemini

We are removing OpenAI and switching to **Google Gemini** for both Embeddings and Chat Generation to utilize the free tier.

**Technical Changes:**
1. **Library:** Replace `openai` with `google-generativeai` (and `langchain-google-genai` if applicable).
2. **Embeddings:** Use model `models/text-embedding-004`.
   - **CRITICAL:** This model outputs **768 dimensions**.
   - We MUST recreate the Qdrant collection because the previous one was set to 1536 dimensions.
3. **Chat Model:** Use `gemini-1.5-flash` (Fast and efficient for RAG).
4. **Environment:** Require `GOOGLE_API_KEY` in `.env`.

## Response snapshot

Created plan to migrate from OpenAI to Google Gemini for both embeddings and chat generation. This includes updating dependencies, modifying backend code to use Google's API, updating environment variables, and recreating the Qdrant collection with the correct 768 dimensions for the new embedding model.

## Outcome

- ✅ Impact: Planned migration from OpenAI to Google Gemini API to utilize free tier
- 🧪 Tests: N/A
- 📁 Files: Planned updates to requirements.txt, main.py, ingest.py, and environment files
- 🔁 Next prompts: Execute the migration plan by updating code and dependencies
- 🧠 Reflection: Critical to recreate Qdrant collection with 768 dimensions to match new embedding model

## Evaluation notes (flywheel)

- Failure modes observed: Qdrant collection dimension mismatch could cause errors
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Update dependencies and environment variables first
