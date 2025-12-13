# Quickstart: Intelligent Content Adaptation System

**Feature**: Content Adaptation System
**Date**: 2025-12-13

## Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Google Gemini API key (for development)

### Backend Setup
1. Navigate to the backend directory
2. Install dependencies: `pip install fastapi uvicorn python-multipart google-generativeai`
3. Set environment variables:
   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```
4. Run the development server: `uvicorn main:app --reload`

### Frontend Setup
1. Navigate to the frontend directory
2. Install dependencies: `npm install`
3. Run the development server: `npm start`

## Key Components

### Backend API
- Main transformation endpoint: `POST /api/transform`
- Handles Urdu translation and content personalization
- Preserves markdown formatting, code blocks, and images

### Frontend Components
- `ChapterTools` component: Toolbar with translation and personalization options
- Injected globally via Docusaurus Root theme component
- Manages local state for user preferences

## Running Tests
- Backend: `pytest tests/`
- Frontend: `npm test`

## API Usage Example
```javascript
// Transform content to Urdu
const response = await fetch('/api/transform', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    sourceContent: '# Introduction to Robotics...',
    transformationType: 'urdu-translation',
    preserveFormatting: true
  })
});
const result = await response.json();
```

## Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key for content transformation
- `BACKEND_URL`: Backend API URL (for development vs production)