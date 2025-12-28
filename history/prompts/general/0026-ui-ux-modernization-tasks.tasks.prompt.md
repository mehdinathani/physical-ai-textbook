---
id: 0026
title: UI/UX Modernization Task Generation
stage: tasks
date: 2025-12-28
surface: agent
model: claude-opus-4-5-20251101
feature: master
branch: master
user: mehdinathani
command: /sp.tasks
labels: ["ui-ux", "tasks", "docusaurus", "css", "frontend"]
links:
  spec: specs/master/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/master/tasks.md
tests:
  - none
---

## Prompt

# Tasks: UI/UX Modernization

**Input**: Design requirements from plan.md
**Context**: Converting standard Docusaurus theme to a custom "Physical AI" brand.

## Phase 1: Foundation (Typography & Colors)

- [ ] T001 Update `docusaurus.config.ts` to load Google Fonts: `['Inter', 'Plus Jakarta Sans', 'JetBrains Mono']`.
- [ ] T002 Overhaul `src/css/custom.css`:
    - Define new CSS variables for `--ifm-color-primary` (Use a modern Teal/Emerald shade like `#10b981`).
    - Update `--ifm-font-family-base` and `--ifm-font-family-monospace`.
    - Adjust `--ifm-background-color` for a softer dark mode (e.g., `#0f172a` instead of black).

## Phase 2: Component Styling (The Details)

- [ ] T003 **Navbar Styling**: Add a glassmorphism effect (backdrop-blur) and update the logo/title font weight.
- [ ] T004 **Sidebar Styling**: Remove the right border. Add rounded background highlights for the active menu item.
- [ ] T005 **Admonitions/Alerts**: Customize the CSS for "Note", "Tip", and "Warning" boxes to use modern border-left styling instead of full blocks.
- [ ] T006 **Doc Cards**: Style the "Next/Previous" pagination cards to look like clickable interactive tiles (hover effects, shadows).

## Phase 3: Landing Page Redesign (The "Wow" Factor)

- [ ] T007 Rewrite `src/pages/index.tsx`.
- [ ] T008 Implement a **Hero Section**:
    - Large Gradient Text Title ("From Digital Brains to Embodied Bodies").
    - "Get Started" and "Chat with Book" Call-to-Action buttons.
- [ ] T009 Implement a **Feature Grid**:
    - 3 Columns displaying "ROS 2", "Isaac Sim", and "RealSense" with icons.

## Phase 4: Polish

- [ ] T010 Add smooth scrolling behavior to `html`.
- [ ] T011 Update the Footer to a "Big Footer" style with columns (Community, Docs, Socials).
- [ ] T012 Push changes to GitHub to trigger Vercel deployment.

## Response snapshot

Generated comprehensive tasks.md with 16 tasks organized across 4 phases (Foundation, Component Styling, Landing Page, Polish). Tasks include proper format with IDs, [P] parallel markers, [FR] requirement labels, and exact file paths. Identified 6 parallel execution opportunities. MVP scope defined as Phase 1 only for quick deployment.

## Outcome

- Impact: Complete task list generated for UI/UX modernization feature
- Tests: none (manual visual verification only)
- Files: 1 file updated (specs/master/tasks.md)
- Next prompts: Run /sp.implement to execute tasks, starting with Phase 1
- Reflection: Task format follows template structure with clear dependencies and parallel opportunities identified

## Evaluation notes (flywheel)

- Failure modes observed: None - tasks align with plan and spec requirements
- Graders run and results (PASS/FAIL): N/A - task generation phase
- Prompt variant (if applicable): Standard /sp.tasks workflow
- Next experiment (smallest change to try): Validate task execution time estimates after implementing first phase
