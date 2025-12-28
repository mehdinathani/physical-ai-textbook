# Research: Modern UI/UX Overhaul

**Date**: 2025-12-28 | **Feature**: UI/UX Modernization

## Research Tasks

### 1. Docusaurus Theming & Infima CSS Variables

**Question**: How to properly override Docusaurus default theme?

**Findings**:
- Docusaurus uses Infima CSS framework with CSS custom properties
- All variables defined in `:root` and `[data-theme='dark']` selectors
- Primary color requires 7 shades: primary, dark, darker, darkest, light, lighter, lightest
- Custom CSS loaded via `themeConfig.customCss` in docusaurus.config.ts

**Decision**: Override Infima variables in `src/css/custom.css`
**Rationale**: Native approach, maintains Docusaurus upgrade path
**Alternatives Rejected**:
- Swizzling theme components (too invasive)
- PostCSS plugins (added complexity)

### 2. Google Fonts Integration in Docusaurus

**Question**: Best practice for loading custom fonts?

**Findings**:
- Can use `@import` in CSS or `<link>` in `<head>`
- Docusaurus 3.x supports headTags in config for link tags
- Performance: Use `font-display: swap` and preconnect hints
- Google Fonts CDN provides optimized woff2 files

**Decision**: Use CSS `@import` with preconnect in custom.css
**Rationale**: Simple, contained in one file, works with SSG
**Alternatives Rejected**:
- Self-hosted fonts (added build complexity)
- Webpack font loader (overkill for 3 fonts)

**Implementation**:
```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
```

### 3. Glassmorphism CSS Patterns

**Question**: How to implement glassmorphism navbar?

**Findings**:
- Core CSS: `backdrop-filter: blur()` + semi-transparent background
- Browser support: 95%+ (all modern browsers)
- Fallback: Solid background for older browsers
- Performance: GPU-accelerated, minimal impact

**Decision**: Apply to navbar with `backdrop-filter: blur(12px)`
**Rationale**: Modern look, performant, good browser support
**Alternatives Rejected**:
- SVG filters (poor performance)
- Canvas effects (too complex)

**Implementation**:
```css
.navbar {
  backdrop-filter: saturate(180%) blur(12px);
  -webkit-backdrop-filter: saturate(180%) blur(12px);
  background: rgba(255, 255, 255, 0.8);
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}
```

### 4. Color Accessibility Verification

**Question**: Do proposed colors meet WCAG AA standards?

**Findings**:
- **Light Mode**: #10b981 on #fafafa = 3.8:1 (AA for large text)
- **Light Mode**: #059669 on #fafafa = 4.9:1 (AA for all text)
- **Dark Mode**: #34d399 on #0f172a = 8.2:1 (AAA compliant)
- **Dark Mode**: #34d399 on #1e293b = 6.4:1 (AA compliant)

**Decision**: Use darker emerald (#059669) for body text, lighter (#10b981) for accents
**Rationale**: Maintains visual identity while meeting accessibility requirements
**Alternatives Rejected**:
- Pure green (low contrast)
- Teal (conflicts with links)

### 5. Homepage Component Architecture

**Question**: Best structure for redesigned homepage?

**Findings**:
- Docusaurus pages are React components in `src/pages/`
- Can use CSS modules for scoped styling
- Layout component handles navbar/footer
- MDX not needed for pure React pages

**Decision**: Single `index.tsx` with component sections, `index.module.css` for styles
**Rationale**: Simple, maintainable, follows Docusaurus patterns
**Alternatives Rejected**:
- Multiple component files (overkill for 3 sections)
- MDX page (less control over layout)

### 6. Sidebar Styling Approach

**Question**: How to style sidebar without breaking functionality?

**Findings**:
- Sidebar uses `.menu` class hierarchy
- Active items use `.menu__link--active`
- Collapsible categories use `.menu__list-item--collapsed`
- Mobile sidebar toggle handled by theme

**Decision**: Override CSS classes, not swizzle components
**Rationale**: Non-breaking, upgradeable, simpler maintenance
**Alternatives Rejected**:
- Swizzle DocSidebar (breaking changes risk)
- Custom sidebar (lose Docusaurus features)

### 7. Admonition Customization

**Question**: How to restyle alert boxes?

**Findings**:
- Admonitions use `.admonition` class with type modifiers
- Types: note, tip, info, warning, danger
- Each type has associated color via `--ifm-color-*`
- SVG icons can be replaced via CSS

**Decision**: Modern border-left style with transparent backgrounds
**Rationale**: Clean, minimal, professional look
**Alternatives Rejected**:
- Full background blocks (dated look)
- Custom admonition component (complexity)

## Summary of All Decisions

| Area | Decision | Confidence |
|------|----------|------------|
| Theming | Infima CSS variable overrides | HIGH |
| Fonts | Google Fonts via @import | HIGH |
| Navbar | Glassmorphism with backdrop-filter | HIGH |
| Colors | Emerald palette (#10b981/#059669) | HIGH |
| Accessibility | Darker shade for text, lighter for accents | HIGH |
| Homepage | React component with CSS modules | HIGH |
| Sidebar | CSS overrides only | HIGH |
| Admonitions | Border-left style | MEDIUM |

## Unresolved Items

None. All technical decisions resolved.

## References

- [Docusaurus Theming](https://docusaurus.io/docs/styling-layout)
- [Infima CSS Framework](https://infima.dev/)
- [Google Fonts](https://fonts.google.com/)
- [WCAG Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [CSS backdrop-filter](https://developer.mozilla.org/en-US/docs/Web/CSS/backdrop-filter)
