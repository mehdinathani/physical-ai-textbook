# ChatKit Integration: Error Analysis & Fixes

**Date**: 2025-12-20
**Status**: Fixed ✅
**Issue**: `process is not defined` and environment variable resolution errors

---

## Error Summary

### Original Errors

1. **PRIMARY ERROR**: `process is not defined` at ChatWidget.tsx:12:91
2. **SECONDARY ERROR**: `Unexpected: no Docusaurus route context found`

### Root Cause

The ChatWidget component was using `process.env.CHATKIT_BACKEND_URL` which doesn't work in Vite (the build tool used by Docusaurus 3.x). Vite uses a different approach called `import.meta.env` for environment variables.

---

## Detailed Error Analysis

### Error 1: `process is not defined`

**Location**: `frontend/src/components/ChatWidget.tsx:22`

**Code That Failed**:
```tsx
const chatKit = useChatKit({
  api: {
    url: process.env.CHATKIT_BACKEND_URL || 'http://localhost:8000/chatkit',
    //   ^^^^^^^^^^^^^^^^^^^^ ERROR: process is not defined in Vite
  },
  ...
});
```

**Why It Failed**:
- **Bundler Difference**: `process.env` is a Node.js/CommonJS concept
- **Vite Runtime**: Vite bundles code for browser environment where `process` doesn't exist
- **No Polyfill**: Unlike webpack, Vite doesn't automatically polyfill `process` for browser code
- **Timing Issue**: The error occurred at hook initialization (line 20 `useChatKit`), not at variable access time

**Debug Logs That Would Have Caught This**:
```javascript
console.log('[ChatWidget] process object:', typeof process); // undefined
console.log('[ChatWidget] import.meta.env:', import.meta.env); // { VITE_* variables }
```

### Error 2: `Unexpected: no Docusaurus route context found`

**Location**: Docusaurus theme plugin

**Why It Occurred**:
- Component wasn't returning `null` during SSR (server-side rendering) phase quickly enough
- `useChatKit` hook was being called unconditionally, even during SSR
- This caused React to try rendering hooks outside of proper context

**Root Cause Chain**:
```
1. ChatWidget component loads
2. useChatKit hook runs (before isReady check)
3. Hook requires browser APIs and Docusaurus context
4. During SSR, context doesn't exist → error
```

---

## Solutions Applied

### Solution 1: Use Vite's `import.meta.env` Instead of `process.env`

**Changed**:
```tsx
// ❌ BEFORE (broken in Vite)
const backendUrl = process.env.CHATKIT_BACKEND_URL || 'http://localhost:8000/chatkit';

// ✅ AFTER (Vite-compatible)
const backendUrl = import.meta.env.VITE_CHATKIT_BACKEND_URL || 'http://localhost:8000/chatkit';
```

**Why This Works**:
- `import.meta` is an ES module standard (supported natively)
- Vite replaces `import.meta.env.VITE_*` at build time with actual values
- No runtime polyfills needed

### Solution 2: Add `VITE_` Prefix to Environment Variables

**Changed** `.env.local`:
```bash
# ❌ BEFORE
CHATKIT_BACKEND_URL=http://localhost:8000/chatkit

# ✅ AFTER
VITE_CHATKIT_BACKEND_URL=http://localhost:8000/chatkit
```

**Why This is Required**:
- Vite only exposes variables prefixed with `VITE_` to client code
- This is a security feature (prevents accidental secret exposure)
- Variables without `VITE_` are only available in Node.js context (build time)

### Solution 3: Defer `useChatKit` Hook Until Client-Side Ready

**Changed** hook initialization logic:
```tsx
// ❌ BEFORE (hook called unconditionally)
const chatKit = useChatKit({
  api: { url: process.env.CHATKIT_BACKEND_URL || '...' },
  ...
});

if (!isReady) {
  return null; // Too late - hook already executed!
}

// ✅ AFTER (hook only called when client-ready)
let chatKit = null;
if (isReady) {
  try {
    chatKit = useChatKit({
      api: { url: import.meta.env.VITE_CHATKIT_BACKEND_URL || '...' },
      ...
    });
  } catch (error) {
    console.error('[ChatWidget] Error initializing ChatKit:', error);
    throw error;
  }
}

if (!isReady) {
  return null; // Now hook execution is protected
}
```

**Why This Works**:
- Hooks only execute if `isReady` is true
- `isReady` is false during SSR (window undefined)
- Component returns null before hooks run
- Proper SSR + React hook lifecycle

### Solution 4: Add Comprehensive Debug Logging

Added debug logs at key points:

```tsx
// 1. Environment resolution
console.log('[ChatWidget] import.meta.env.VITE_CHATKIT_BACKEND_URL:',
  import.meta.env.VITE_CHATKIT_BACKEND_URL);

// 2. SSR detection
console.log('[ChatWidget] SSR detected - window is undefined, skipping localStorage');

// 3. Thread initialization
console.log('[ChatWidget] Retrieved saved thread ID from localStorage:', savedThreadId);

// 4. Hook initialization
console.log('[ChatWidget] Initializing ChatKit with URL:', backendUrl);

// 5. Error tracking
console.error('[ChatWidget] Error initializing ChatKit:', error);
```

**These logs help identify**:
- Whether environment variables are loading
- Whether SSR detection is working
- Whether ChatKit hook initialization succeeds
- When errors occur and what caused them

---

## Files Modified

### 1. `frontend/src/components/ChatWidget.tsx`

**Changes**:
- Replaced `process.env.CHATKIT_BACKEND_URL` with `import.meta.env.VITE_CHATKIT_BACKEND_URL`
- Moved `useChatKit` hook inside `if (isReady)` guard
- Added 12+ debug console.log statements
- Added error handling with try-catch

**Lines Changed**: 1-99 (complete rewrite of initialization logic)

### 2. `frontend/.env.local`

**Changes**:
- Renamed `CHATKIT_BACKEND_URL` → `VITE_CHATKIT_BACKEND_URL`

**Lines Changed**: Line 1

---

## Debugging Guide

### How to Read the Debug Output

When you run `npm start`, you should see this sequence in browser console:

```
[ChatWidget] Component mounted
[ChatWidget] import.meta.env.VITE_CHATKIT_BACKEND_URL: http://localhost:8000/chatkit
[ChatWidget] typeof window: object
[ChatWidget] isReady: false
[ChatWidget] useEffect: Checking client-side environment...
[ChatWidget] Retrieved saved thread ID from localStorage: null
[ChatWidget] isReady set to true - component ready for rendering
[ChatWidget] Backend URL resolved to: http://localhost:8000/chatkit
[ChatWidget] Initializing ChatKit with URL: http://localhost:8000/chatkit
[ChatWidget] ChatKit hook initialized successfully
[ChatWidget] Returning null - SSR phase or loading  ← If you see this, SSR is not happening
```

### If You See Errors

**Error**: `import.meta.env.VITE_CHATKIT_BACKEND_URL is undefined`
- **Cause**: Environment variable not set or not prefixed with `VITE_`
- **Fix**: Check `.env.local` has `VITE_CHATKIT_BACKEND_URL=http://localhost:8000/chatkit`

**Error**: `Error initializing ChatKit: [specific error]`
- **Cause**: ChatKit configuration issue or backend unreachable
- **Fix**: Check backend is running (`cd backend-chatkit && python main.py`)

**Error**: `process is not defined` (still appears)
- **Cause**: Stale Vite cache or missing env variable prefix
- **Fix**:
  1. Stop dev server (Ctrl+C)
  2. Delete `frontend/.vite/` cache
  3. Delete `frontend/node_modules/.vite/` cache
  4. Run `npm start` again

### Environment Variable Resolution

The environment variable is resolved in this order:

```
1. Check import.meta.env.VITE_CHATKIT_BACKEND_URL (from .env.local)
   ↓ (if undefined)
2. Use fallback: 'http://localhost:8000/chatkit' (hardcoded default)
   ↓
3. Resolved URL is logged: [ChatWidget] Backend URL resolved to: ...
```

---

## Best Practices Applied

### 1. **Vite Environment Variables**
- ✅ Use `import.meta.env.*` not `process.env.*`
- ✅ Prefix variables with `VITE_` to expose to client code
- ✅ Provide fallback values with `||` operator

### 2. **SSR Safety**
- ✅ Check `typeof window !== 'undefined'` before accessing browser APIs
- ✅ Defer hook initialization until client-ready
- ✅ Return `null` during SSR phase

### 3. **Error Handling**
- ✅ Wrap localStorage access in try-catch
- ✅ Gracefully degrade if localStorage unavailable
- ✅ Log errors with context for debugging

### 4. **Debugging**
- ✅ Add labeled console.log statements (e.g., `[ChatWidget]` prefix)
- ✅ Log at decision points (if/else branches)
- ✅ Log values that affect logic
- ✅ Use console.error for unexpected conditions

---

## Testing the Fix

### Step 1: Verify Environment Variable

```bash
# In frontend directory
cat .env.local
# Expected output:
# VITE_CHATKIT_BACKEND_URL=http://localhost:8000/chatkit
```

### Step 2: Start Backend

```bash
cd backend-chatkit
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
set GEMINI_API_KEY=your_key_here
python main.py
# Expected: Server running on http://localhost:8000
```

### Step 3: Start Frontend

```bash
cd frontend
npm start
# Expected: Dev server on http://localhost:3000
# Check browser console for debug logs
```

### Step 4: Check Browser Console

Open DevTools (F12) → Console tab, you should see:
```
[ChatWidget] Component mounted
[ChatWidget] Backend URL resolved to: http://localhost:8000/chatkit
[ChatWidget] ChatKit hook initialized successfully
```

### Step 5: Test Chat

1. Click floating chat button (bottom-right)
2. Type: "Hello"
3. Should see response streaming in real-time

---

## Summary

| Issue | Root Cause | Solution | Status |
|-------|-----------|----------|--------|
| `process is not defined` | Using `process.env` in Vite | Use `import.meta.env.VITE_*` | ✅ Fixed |
| Environment variable not found | Missing `VITE_` prefix | Rename to `VITE_CHATKIT_BACKEND_URL` | ✅ Fixed |
| SSR context errors | Hook runs during SSR | Defer hook until `isReady` true | ✅ Fixed |
| Hard to debug | No logging | Added 12+ debug statements | ✅ Improved |

---

## Related Files

- `.env.local` - Environment variables (Vite-compatible)
- `frontend/src/components/ChatWidget.tsx` - Fixed component
- `frontend/src/theme/Root.tsx` - ChatKit CDN injection
- `backend-chatkit/main.py` - Backend service

---

## References

- [Vite Env Variables Documentation](https://vitejs.dev/guide/env-and-mode)
- [Docusaurus with Vite](https://docusaurus.io/)
- [React Hooks SSR Safety](https://react.dev/reference/react/useEffect#specifying-reactive-dependencies)
