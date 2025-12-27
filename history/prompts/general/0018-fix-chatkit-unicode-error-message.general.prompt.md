---
id: 0018
title: Fix ChatKit Unicode Error Message
stage: general
date: 2025-12-27
surface: agent
model: claude-opus-4-5
feature: none
branch: master
user: MEHDI
command: /sp.clarify
labels: ["chatkit", "bug-fix", "unicode", "windows"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - backend-chatkit/main.py
tests:
 - Manual testing of ChatKit UI after fix
---

## Prompt

its working fine, but on ui after agent response there is a message bubble "There was an error while generating the assistant's response." why it is? and how to remove it?

## Response snapshot

### Root Cause Identified:
The error message "There was an error while generating the assistant's response." appeared in the chat UI because the backend was crashing with a `UnicodeEncodeError` when trying to print emoji characters (like 😊) to the Windows console (which uses cp1252 encoding).

### The Problem:
When Gemini generated responses containing emojis or other Unicode characters, the debug `print()` statements in `backend-chatkit/main.py` would crash, causing the streaming response to be interrupted. ChatKit's frontend detected this as a failed response and displayed the error message.

### The Fix:
Created a `safe_print()` helper function that handles Unicode encoding errors gracefully:

```python
def safe_print(message: str) -> None:
    """Print with Unicode support, handling Windows console encoding issues."""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback: encode to ASCII with replacement for problematic characters
        print(message.encode('ascii', 'replace').decode('ascii'))
```

Replaced all print statements that could contain user-generated content or AI responses with `safe_print()` at lines:
- Line 124: Store debug logging for thread items
- Line 245: Event update logging in respond()
- Line 251: Assistant message content preview
- Line 283: Accumulated text logging

### Result:
The ChatKit backend server now runs successfully without crashing on Unicode characters. The error message no longer appears in the UI, even when Gemini uses emojis in its responses.

## Outcome

- ✅ Impact: Fixed critical bug causing error messages in ChatKit UI on Windows systems
- 🧪 Tests: Manual testing confirmed - error message no longer appears when agent responds with emojis
- 📁 Files: Modified backend-chatkit/main.py (added safe_print function and updated 4 print statements)
- 🔁 Next prompts: Test deployment to ensure fix works in production environment
- 🧠 Reflection: Windows console encoding (cp1252) limitations require special handling for Unicode in Python print statements

## Evaluation notes (flywheel)

- Failure modes observed: UnicodeEncodeError when printing emojis to Windows console, causing stream interruption and UI error display
- Graders run and results (PASS/FAIL): Manual testing PASS - ChatKit UI now works correctly with Unicode responses
- Prompt variant (if applicable): N/A - Bug fix based on console log analysis
- Next experiment (smallest change to try): Consider disabling debug logging entirely in production, or using proper logging framework with UTF-8 encoding
