# Quickstart: UI/UX Modernization

**Feature**: Modern UI/UX Overhaul | **Date**: 2025-12-28

## Prerequisites

- Node.js 18+ installed
- pnpm or npm package manager
- Git repository cloned

## Development Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Dev server runs at `http://localhost:3000`

## Key Files to Modify

| File | Purpose |
|------|---------|
| `src/css/custom.css` | CSS variable overrides, component styles |
| `src/pages/index.tsx` | Homepage layout and components |
| `src/pages/index.module.css` | Homepage-specific styles |
| `docusaurus.config.ts` | Font preloading (optional) |

## Implementation Order

### Step 1: Typography & Colors (custom.css)

Add at top of `src/css/custom.css`:
```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
```

Update `:root` variables for light mode, `[data-theme='dark']` for dark mode.

### Step 2: Component Styles (custom.css)

Add navbar, sidebar, admonition styles after variable definitions.

### Step 3: Homepage (index.tsx)

Replace default hero with gradient text header and feature grid.

### Step 4: Verify Build

```bash
npm run build
```

Must complete with no errors before proceeding.

## Testing Checklist

- [ ] Light mode colors correct
- [ ] Dark mode toggle works
- [ ] Fonts loading (check DevTools Network)
- [ ] Homepage hero displays
- [ ] Feature grid responsive
- [ ] Sidebar active states work
- [ ] Navbar glassmorphism visible
- [ ] Build completes successfully
- [ ] Vercel preview deploys

## Common Issues

### Fonts not loading
- Check @import URL is correct
- Verify no CORS errors in console
- Test with DevTools network throttling

### Dark mode broken
- Ensure all variables defined in both `:root` and `[data-theme='dark']`
- Check CSS specificity conflicts

### Build fails
- Run `npm run clear` to clear cache
- Check TypeScript errors in index.tsx
- Verify all imports exist

## Deployment

After local verification:
```bash
git add .
git commit -m "feat: Modern UI/UX overhaul"
git push origin master
```

Vercel auto-deploys from master branch.

## Rollback

If issues in production:
```bash
git revert HEAD
git push origin master
```
