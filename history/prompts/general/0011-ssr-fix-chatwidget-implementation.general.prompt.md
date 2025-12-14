---
id: 11
title: SSR Fix ChatWidget Implementation
stage: general
date: 2025-12-14
surface: agent
model: claude-haiku-4-5-20251001
feature: none
branch: master
user: Claude
command: general
labels: ["ssr", "chat-widget", "build-error", "vercel"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - C:\mehdibackup\GIAIC\Q4\HACKATHON1\physai-foundations\frontend\src\components\ChatWidget.tsx
tests:
 - npm run build
---

## Prompt

Fix Vercel build error related to server-side rendering (SSR) in ChatWidget component. The error was:
- (0 , ai_dist_namespaceObject.useChat) is not a function
- export 'cn' (imported as 'cn') was not found in 'clsx'

## Response snapshot

Fixed the ChatWidget component to handle server-side rendering properly by:

1. Updating imports to use correct packages for cn utility:
   - Changed import from 'clsx' to import { clsx } from 'clsx'
   - Created a cn utility function that combines clsx and twMerge

2. Handling the useChat hook properly for SSR:
   - Removed direct import of useChat from 'ai/react' which isn't available during SSR
   - Implemented dynamic import of useChat hook that only runs on the client side
   - Added loading states to handle the case when the chat hook is not yet loaded

3. Updated the component to handle the asynchronous loading of the useChat hook:
   - Added state to store chat data
   - Added useEffect to dynamically load the useChat hook
   - Added conditional rendering for when chat data is not yet available

The changes ensure that the useChat hook is only accessed on the client side, preventing SSR errors while maintaining functionality.

## Outcome

- ✅ Impact: Fixed Vercel build error and enabled successful deployment
- 🧪 Tests: npm run build completed successfully
- 📁 Files: Updated ChatWidget.tsx to handle SSR properly
- 🔁 Next prompts: None needed - issue resolved
- 🧠 Reflection: Dynamic imports are essential for client-side only functionality in SSR environments

## Evaluation notes (flywheel)

- Failure modes observed: Server-side rendering errors during build process
- Graders run and results (PASS/FAIL): Build process - PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A - issue resolved
