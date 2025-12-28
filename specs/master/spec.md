# Feature Specification: Modern UI/UX Overhaul

**Version**: 1.0.0 | **Date**: 2025-12-28 | **Status**: Draft

## Summary

Complete visual redesign of the Physical AI Textbook frontend to achieve a "Modern Technical/Industrial" aesthetic, inspired by Stripe and Linear documentation styles.

## Design Philosophy

**Vibe**: Clean, high-contrast, "Stripe-like" or "Linear-like" documentation style.

### Color Palette
- **Primary**: Deep Emerald/Forest Green (representing "Physical" world) mixed with Electric Blue accents (AI)
- **Light Mode Background**: Slight off-white (paper texture feel)
- **Dark Mode Background**: Deep Charcoal (not pitch black, softer on eyes)

### Typography
- **Headings**: 'Plus Jakarta Sans' or 'Inter' (Modern, geometric)
- **Body**: 'Inter' (High readability)
- **Code**: 'JetBrains Mono' (Developer standard)

## Functional Requirements

### FR-001: Typography & Colors Foundation
- Update `docusaurus.config.ts` to load Google Fonts
- Override Infima CSS variables in `src/css/custom.css`
- Define modern Teal/Emerald primary color (#10b981)
- Softer dark mode background (#0f172a instead of black)

### FR-002: Component Styling
- **Navbar**: Glassmorphism effect (backdrop-blur), updated logo/title font weight
- **Sidebar**: Remove right border, rounded background highlights for active items
- **Admonitions**: Modern border-left styling instead of full blocks
- **Doc Cards**: Interactive tile styling with hover effects and shadows

### FR-003: Landing Page Redesign
- **Hero Section**: Large gradient text title, CTA buttons ("Get Started", "Chat with Book")
- **Feature Grid**: 3-column layout showcasing ROS 2, Isaac Sim, RealSense with icons

### FR-004: Polish & Accessibility
- Smooth scrolling behavior
- "Big Footer" with columns (Community, Docs, Socials)
- WCAG AA contrast compliance

## Non-Functional Requirements

### NFR-001: Performance
- Maintain current load times (fonts async loaded)
- CSS bundle size increase < 10KB

### NFR-002: Accessibility
- WCAG AA contrast ratios maintained in both themes
- Keyboard navigation preserved

### NFR-003: Responsiveness
- Mobile sidebar collapse works correctly
- Feature grid collapses to single column on mobile

## Testing Strategy
- Verify Dark/Light mode switching preserves contrast
- Ensure responsiveness on mobile (sidebar collapse)
- Visual regression testing via screenshots

## Success Criteria
- [ ] All typography updates applied consistently
- [ ] Color palette implemented for both light/dark modes
- [ ] Homepage redesigned with hero and feature grid
- [ ] Glassmorphism navbar effect working
- [ ] Footer redesigned to multi-column layout
- [ ] No accessibility regressions
