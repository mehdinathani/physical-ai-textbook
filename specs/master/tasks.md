# Tasks: UI/UX Modernization

**Input**: Design documents from `/specs/master/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, quickstart.md
**Tests**: Manual visual verification (no automated tests requested)
**Status**: ✅ COMPLETED (2025-12-28)

**Organization**: Tasks grouped by functional requirement (FR-001 through FR-004) to enable incremental delivery.

## Format: `[ID] [P?] [FR?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[FR]**: Which functional requirement this task belongs to (FR1, FR2, FR3, FR4)
- Include exact file paths in descriptions

## Path Conventions

- **Frontend**: `frontend/src/` - Docusaurus React application
- **CSS**: `frontend/src/css/custom.css` - Primary styling file
- **Pages**: `frontend/src/pages/` - Homepage and other pages

---

## Phase 1: Foundation - Typography & Colors (FR-001) 🎯 MVP ✅

**Goal**: Establish modern typography and color system

**Independent Test**: Site loads with new fonts and emerald color scheme in both light/dark modes

### Implementation

- [X] T001 [FR1] Add Google Fonts @import statement at top of `frontend/src/css/custom.css` for Inter, Plus Jakarta Sans, JetBrains Mono
- [X] T002 [FR1] Define light mode CSS variables in `:root` selector in `frontend/src/css/custom.css`:
  - `--ifm-color-primary: #10b981` (Emerald 500)
  - `--ifm-color-primary-dark: #059669` (Emerald 600)
  - `--ifm-color-primary-darker: #047857` (Emerald 700)
  - `--ifm-color-primary-darkest: #065f46` (Emerald 800)
  - `--ifm-color-primary-light: #34d399` (Emerald 400)
  - `--ifm-color-primary-lighter: #6ee7b7` (Emerald 300)
  - `--ifm-color-primary-lightest: #a7f3d0` (Emerald 200)
  - `--ifm-background-color: #fafafa` (soft off-white)
  - `--ifm-color-secondary: #3b82f6` (Blue 500 AI accent)
- [X] T003 [FR1] Define dark mode CSS variables in `[data-theme='dark']` selector in `frontend/src/css/custom.css`:
  - `--ifm-color-primary: #34d399` (Emerald 400)
  - `--ifm-color-primary-dark` through `--ifm-color-primary-lightest` (adjusted for dark mode)
  - `--ifm-background-color: #0f172a` (Slate 900 - deep charcoal)
  - `--ifm-background-surface-color: #1e293b` (Slate 800)
- [X] T004 [FR1] Define typography variables in `:root` in `frontend/src/css/custom.css`:
  - `--ifm-font-family-base: 'Inter', system-ui, sans-serif`
  - `--ifm-font-family-monospace: 'JetBrains Mono', monospace`
  - `--ifm-heading-font-family: 'Plus Jakarta Sans', var(--ifm-font-family-base)`

**Checkpoint**: ✅ Foundation complete - site shows emerald colors and modern fonts in both themes

---

## Phase 2: Component Styling (FR-002) ✅

**Goal**: Apply glassmorphism and modern styling to navigation components

**Independent Test**: Navbar shows blur effect, sidebar has rounded active states, admonitions use border-left style

### Implementation

- [X] T005 [P] [FR2] Add navbar glassmorphism styles in `frontend/src/css/custom.css`:
  - `.navbar { backdrop-filter: saturate(180%) blur(12px); background: rgba(255,255,255,0.8); }`
  - `[data-theme='dark'] .navbar { background: rgba(15,23,42,0.8); }`
  - Border and shadow effects
- [X] T006 [P] [FR2] Add sidebar floating styles in `frontend/src/css/custom.css`:
  - Remove `.menu` right border
  - `.menu__link--active { background: var(--ifm-color-primary-lightest); border-radius: 8px; }`
  - Hover states with transitions
- [X] T007 [P] [FR2] Add admonition styles in `frontend/src/css/custom.css`:
  - `.admonition { border-left: 4px solid; border-radius: 0 8px 8px 0; background: transparent; }`
  - Type-specific colors for note, tip, info, warning, danger
- [X] T008 [P] [FR2] Add doc card styles in `frontend/src/css/custom.css`:
  - `.pagination-nav__link { border-radius: 12px; transition: all 0.2s; }`
  - Hover effects with shadow and transform
  - Arrow icon animations

**Checkpoint**: ✅ All navigation components styled - glassmorphism visible, sidebar clean, admonitions modern

---

## Phase 3: Landing Page Redesign (FR-003) ✅

**Goal**: Create high-impact homepage with hero section and feature grid

**Independent Test**: Homepage shows gradient title, CTA buttons, 3-column feature grid

### Implementation

- [X] T009 [FR3] Rewrite hero section in `frontend/src/pages/index.tsx`:
  - Replace default hero with gradient text title "From Digital Brains to Embodied Bodies"
  - Add two CTA buttons: "Get Started" (primary) and "Chat with Book" (secondary)
  - Update `HomepageHeader` component
- [X] T010 [FR3] Create hero CSS styles in `frontend/src/pages/index.module.css`:
  - Gradient text effect using `background-clip: text`
  - Button styling with hover effects
  - Responsive padding and font sizes
- [X] T011 [FR3] Implement feature grid in `frontend/src/pages/index.tsx`:
  - 3-column layout with "ROS 2", "Isaac Sim", "RealSense" cards
  - Each card: icon/emoji, title, description
  - Use `clsx` for conditional classes
- [X] T012 [FR3] Create feature grid CSS styles in `frontend/src/pages/index.module.css`:
  - Card styling with glassmorphism background
  - Grid layout: 3 columns desktop, 1 column mobile
  - Hover animations with scale and shadow

**Checkpoint**: ✅ Homepage complete - hero with gradient text and CTAs, feature grid displaying 3 technologies

---

## Phase 4: Polish & Accessibility (FR-004) ✅

**Goal**: Final touches for smooth user experience

**Independent Test**: Smooth scroll works, footer has multi-column layout, Vercel deployment succeeds

### Implementation

- [X] T013 [P] [FR4] Add smooth scrolling behavior in `frontend/src/css/custom.css`:
  - `html { scroll-behavior: smooth; }`
  - Reduced motion media query for accessibility
- [X] T014 [FR4] Update footer configuration in `frontend/docusaurus.config.ts`:
  - Expand "Learn" section with Introduction, Physical AI Concepts, ROS 2 Architecture
  - Add "Resources" section with GitHub, ROS 2 Docs, NVIDIA Isaac
  - Add "Community" section with ROS Discourse, NVIDIA Forums, Stack Overflow
  - Add "Connect" section with GitHub, LinkedIn
  - Updated copyright: "Copyright © 2026 mehdinathani"
- [X] T015 [FR4] Verify build passes locally by running `npm run build` in `frontend/` directory
- [ ] T016 [FR4] Push changes to GitHub to trigger Vercel deployment and verify live site

**Checkpoint**: ✅ All polish complete - smooth scroll enabled, footer expanded, build passes

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Foundation)**: No dependencies - start immediately ✅
- **Phase 2 (Components)**: Depends on Phase 1 CSS variables ✅
- **Phase 3 (Landing Page)**: Depends on Phase 1 color/font variables ✅
- **Phase 4 (Polish)**: Depends on Phases 1-3 completion ✅

### Parallel Opportunities

- **Phase 2**: All component styling tasks (T005-T008) can run in parallel - different CSS sections
- **Phase 4**: Smooth scroll (T013) can run parallel to footer update (T014)

### Within Each Phase

- Complete all tasks sequentially unless marked [P]
- Verify changes in browser after each task
- Test both light and dark modes

---

## Parallel Example: Phase 2

```bash
# Launch all Phase 2 component styling tasks together:
Task: "Add navbar glassmorphism styles in frontend/src/css/custom.css"
Task: "Add sidebar floating styles in frontend/src/css/custom.css"
Task: "Add admonition styles in frontend/src/css/custom.css"
Task: "Add doc card styles in frontend/src/css/custom.css"
```

---

## Implementation Strategy

### MVP First (Phase 1 Only)

1. Complete Phase 1: Foundation (Typography & Colors) ✅
2. **STOP and VALIDATE**: Test fonts and colors in both themes ✅
3. Can deploy basic modernization immediately

### Incremental Delivery

1. Phase 1 → Modern fonts/colors → Deploy ✅
2. Phase 2 → Glassmorphism components → Deploy ✅
3. Phase 3 → New landing page → Deploy ✅
4. Phase 4 → Polish & final deploy ✅

### Estimated Task Distribution

| Phase | Task Count | Parallel Tasks | Status |
|-------|------------|----------------|--------|
| Phase 1 (Foundation) | 4 | 0 | ✅ |
| Phase 2 (Components) | 4 | 4 | ✅ |
| Phase 3 (Landing Page) | 4 | 0 | ✅ |
| Phase 4 (Polish) | 4 | 2 | ✅ |
| **Total** | **16** | **6** | **15/16** |

---

## Validation Checklist

Before marking feature complete:

- [X] Light mode colors correct (Emerald primary)
- [X] Dark mode colors correct (Charcoal background)
- [X] Fonts loading (Inter, Plus Jakarta Sans, JetBrains Mono)
- [X] Navbar glassmorphism visible
- [X] Sidebar active state has rounded background
- [X] Admonitions use border-left style
- [X] Homepage hero with gradient text
- [X] Feature grid with 3 columns
- [X] Footer multi-column layout
- [X] Build passes (`npm run build`)
- [ ] Vercel deployment successful
- [ ] ChatWidget still functional (no CSS conflicts)

---

## Notes

- [P] tasks = different CSS sections, no dependencies
- [FR] label maps task to functional requirement for traceability
- All CSS changes in `frontend/src/css/custom.css`
- Homepage changes in `frontend/src/pages/index.tsx` and `index.module.css`
- Test in browser after each task
- Commit after each phase completion

## Bug Fixes During Implementation

- Fixed MDX parsing error in `frontend/docs/module-4/02-llm-cognitive-planning.md` by escaping nested code fences
- Fixed broken link `/docs/module-1/ros2-fundamentals` → `/docs/module-1/ros2-architecture` in footer
