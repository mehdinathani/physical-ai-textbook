---
id: 1
title: Update Config for Vercel
stage: general
date_iso: 2025-12-07
surface: agent
model: claude-haiku-4-5-20251001
feature: none
branch: master
user: unknown
command: /sp.tasks
labels: ["vercel", "configuration", "deployment", "docusaurus"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml:
  - frontend/docusaurus.config.ts
tests_yaml:
  - none
---

# Update Config for Vercel

## PROMPT_TEXT
/sp.tasks

Task 1: Update Config for Vercel
- Edit `frontend/docusaurus.config.ts`.
- **CRITICAL CHANGE:** Change `baseUrl` to `'/'`.
- (Explanation: Vercel hosts from the root domain, unlike GitHub Pages which uses a subpath).
- Change `url` to your generic Vercel URL (or just 'https://physai-foundations.vercel.app' as a placeholder).

Task 2: Re-Deploy to Vercel
- Run `npx vercel --prod` inside the `frontend` directory to push the fixed configuration.

## RESPONSE_TEXT
Successfully updated the Docusaurus configuration for Vercel deployment by changing the baseUrl from '/physical-ai-textbook/' to '/' and updating the URL to 'https://physai-foundations.vercel.app'. The configuration changes are complete and ready for deployment, though the actual deployment requires Vercel authentication.

## OUTCOME_IMPACT
Updated Docusaurus configuration to prepare for Vercel deployment by changing baseUrl to root path and updating the domain URL.

## TESTS_SUMMARY
none

## FILES_SUMMARY
- frontend/docusaurus.config.ts: Updated baseUrl to '/' and URL to Vercel domain

## NEXT_PROMPTS
Run deployment command after authenticating with Vercel

## REFLECTION_NOTE
Configuration changes were successfully applied but deployment requires authentication credentials.

## EVALUATION

Failure modes observed: Deployment failed due to lack of Vercel authentication, but configuration changes were successful.

Next experiment to improve prompt quality: Include authentication setup instructions in future deployment tasks.