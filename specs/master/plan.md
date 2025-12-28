# Implementation Plan: Modern UI/UX Overhaul

**Branch**: `master` | **Date**: 2025-12-28 | **Spec**: [specs/master/spec.md](./spec.md)
**Input**: Feature specification from `/specs/master/spec.md`

## Summary

Complete visual redesign of the Physical AI Textbook frontend achieving a "Modern Technical/Industrial" aesthetic. Technical approach: Override Docusaurus Infima CSS variables, implement custom typography via Google Fonts, and rebuild the homepage with modern React components featuring glassmorphism and gradient effects.

## Technical Context

**Language/Version**: TypeScript 5.x (React 18, Docusaurus 3.x)
**Primary Dependencies**: @docusaurus/preset-classic, @docusaurus/plugin-ideal-image, clsx, React 18
**Storage**: N/A (static site generation)
**Testing**: Visual regression (manual), Lighthouse audits
**Target Platform**: Web (modern browsers, mobile responsive)
**Project Type**: web (frontend-only, Docusaurus static site)
**Performance Goals**: Lighthouse Performance > 90, First Contentful Paint < 1.5s
**Constraints**: No build failures on Vercel, WCAG AA compliance, Free tier services only
**Scale/Scope**: ~20 documentation pages, single landing page, ~5 custom components

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| Spec-Driven Execution | PASS | Changes originate from this spec |
| Bilingual Excellence | PASS | TypeScript only (frontend), no Python changes |
| Hackathon Velocity | PASS | Using Docusaurus built-ins + CSS overrides |
| AI-Native First | PASS | Chat widget integration preserved |
| Academic Rigor | N/A | UI changes, not content changes |
| Modular Architecture | PASS | CSS variables for easy customization |
| Context Awareness | PASS | Building on existing Docusaurus structure |
| Agent & Skill Discovery | PASS | Using frontend-design skill patterns |
| Clean UX | TARGET | Primary goal of this feature |
| Code Quality | PASS | TypeScript strict mode maintained |
| Performance Efficiency | PASS | Async font loading, minimal CSS additions |
| Zero Build Failures | GATE | Must verify before deployment |

**Pre-Design Gate**: PASSED

## Project Structure

### Documentation (this feature)

```text
specs/master/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # N/A for CSS-only feature
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/sp.tasks command)
```

### Source Code (repository root)

```text
frontend/
├── docusaurus.config.ts    # Font loading configuration
├── src/
│   ├── css/
│   │   └── custom.css      # PRIMARY: All CSS variable overrides
│   ├── pages/
│   │   ├── index.tsx       # Homepage redesign
│   │   └── index.module.css # Homepage styles
│   └── components/
│       └── HomepageFeatures/ # Feature grid component (new)
└── static/
    └── img/                 # Any new icons/images
```

**Structure Decision**: Web application (frontend only). All changes confined to `frontend/` directory, primarily CSS and React components.

## Complexity Tracking

No constitution violations to justify.

---

## Phase 0: Research

### Research Tasks Completed

1. **Docusaurus Infima CSS Variables**: Reviewed official Docusaurus theming documentation
2. **Google Fonts Integration**: Standard `@import` or `<link>` in head
3. **Glassmorphism CSS Patterns**: `backdrop-filter: blur()` with semi-transparent backgrounds
4. **Color Accessibility**: Verified #10b981 (Emerald 500) passes WCAG AA on both backgrounds

### Decisions

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Use CSS variables for all colors | Enables easy theme switching, Docusaurus native | Hardcoded colors (rejected: maintainability) |
| Google Fonts via CSS @import | Simple, works with SSG | Self-hosted fonts (rejected: complexity) |
| Keep Docusaurus Infima base | Preserve existing functionality | Custom CSS framework (rejected: breaking changes) |
| Emerald #10b981 as primary | Modern, high contrast, "Physical" feel | Blue (too generic), Purple (not technical) |

---

## Phase 1: Design

### Typography System

```css
/* Fonts */
--ifm-font-family-base: 'Inter', system-ui, -apple-system, sans-serif;
--ifm-font-family-monospace: 'JetBrains Mono', 'Fira Code', monospace;
--ifm-heading-font-family: 'Plus Jakarta Sans', var(--ifm-font-family-base);
```

### Color System

```css
/* Light Mode */
:root {
  --ifm-color-primary: #10b981;           /* Emerald 500 */
  --ifm-color-primary-dark: #059669;      /* Emerald 600 */
  --ifm-color-primary-darker: #047857;    /* Emerald 700 */
  --ifm-color-primary-darkest: #065f46;   /* Emerald 800 */
  --ifm-color-primary-light: #34d399;     /* Emerald 400 */
  --ifm-color-primary-lighter: #6ee7b7;   /* Emerald 300 */
  --ifm-color-primary-lightest: #a7f3d0;  /* Emerald 200 */
  --ifm-background-color: #fafafa;        /* Soft off-white */
  --ifm-color-secondary: #3b82f6;         /* Blue 500 - AI accent */
}

/* Dark Mode */
[data-theme='dark'] {
  --ifm-color-primary: #34d399;           /* Emerald 400 */
  --ifm-background-color: #0f172a;        /* Slate 900 - Deep charcoal */
  --ifm-background-surface-color: #1e293b; /* Slate 800 */
}
```

### Component Patterns

**Glassmorphism Navbar**:
```css
.navbar {
  backdrop-filter: blur(12px);
  background: rgba(255, 255, 255, 0.8);
}
[data-theme='dark'] .navbar {
  background: rgba(15, 23, 42, 0.8);
}
```

**Active Sidebar Item**:
```css
.menu__link--active {
  background: var(--ifm-color-primary-lightest);
  border-radius: 8px;
}
```

**Modern Admonitions**:
```css
.admonition {
  border-left: 4px solid;
  border-radius: 0 8px 8px 0;
  background: transparent;
}
```

### Homepage Layout

```
┌─────────────────────────────────────────────────┐
│                    NAVBAR                        │
├─────────────────────────────────────────────────┤
│                                                  │
│     "From Digital Brains to Embodied Bodies"    │
│              (Gradient Text H1)                 │
│                                                  │
│     [Get Started]    [Chat with Book]           │
│                                                  │
├─────────────────────────────────────────────────┤
│   ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│   │  ROS 2   │ │Isaac Sim │ │RealSense │       │
│   │  icon    │ │  icon    │ │  icon    │       │
│   │  text    │ │  text    │ │  text    │       │
│   └──────────┘ └──────────┘ └──────────┘       │
├─────────────────────────────────────────────────┤
│  Docs  │  Community  │  More                    │
│        │             │                          │
│        │             │                          │
└─────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation (Typography & Colors)
- T001: Update docusaurus.config.ts for Google Fonts
- T002: Overhaul custom.css with new variables

### Phase 2: Component Styling
- T003: Navbar glassmorphism
- T004: Sidebar floating style
- T005: Admonition redesign
- T006: Doc card hover effects

### Phase 3: Landing Page
- T007-T009: Homepage rewrite with hero and feature grid

### Phase 4: Polish
- T010: Smooth scrolling
- T011: Big footer redesign
- T012: Deployment verification

---

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| Font loading delays | Use `font-display: swap`, preconnect hints |
| Dark mode contrast issues | Test with WCAG checker before merge |
| Breaking existing ChatWidget | Preserve existing CSS, test widget after changes |

---

## Next Steps

1. Run `/sp.tasks` to generate detailed task list
2. Execute Phase 1 (Foundation) first
3. Verify build passes before proceeding to Phase 2
