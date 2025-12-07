# PhysAI Foundations - Monorepo Architecture Plan

## 1. Overview

### 1.1. Architecture Strategy
Implement a monorepo architecture in the `physai-foundations` root directory with separate frontend and backend applications that work together to deliver the Physical AI & Humanoid Robotics Textbook with enhanced functionality.

### 1.2. Project Structure
```
physai-foundations/
├── frontend/              # Docusaurus-based textbook frontend
│   ├── docs/             # Textbook content (Module 1 & 2)
│   ├── src/
│   │   ├── components/   # Custom React components
│   │   ├── pages/        # Additional pages including chat interface
│   │   └── css/          # Custom styling
│   ├── static/           # Static assets
│   ├── docusaurus.config.js
│   └── package.json
├── backend/              # FastAPI backend service
│   ├── app/
│   │   ├── api/          # API routes
│   │   ├── models/       # Data models
│   │   ├── services/     # Business logic
│   │   └── database/     # Database connections
│   ├── requirements.txt
│   └── main.py
├── specs/                # Existing specifications
│   └── textbook/
├── .gitignore
├── README.md
└── package.json          # Root package for monorepo management
```

## 2. Frontend Application (`/frontend`)

### 2.1. Framework & Technology
- **Framework:** Docusaurus (TypeScript)
- **Purpose:** Serve textbook content and provide chat interface
- **Key Libraries:** `better-auth` (client), React, various Docusaurus plugins

### 2.2. Features
- Textbook content delivery (Introduction, Module 1, Module 2)
- High-contrast, accessible reading experience
- Sidebar navigation and pagination
- Integrated chat interface for AI-powered learning assistance
- Authentication integration via `better-auth`

### 2.3. Components
- Textbook content pages (Markdown/MDX)
- Interactive code examples
- Chat interface component
- Authentication UI components
- Navigation and layout components

## 3. Backend Service (`/backend`)

### 3.1. Framework & Technology
- **Framework:** FastAPI (Python)
- **Database:** Qdrant (Vector DB for embeddings), Neon Postgres (User data)
- **Purpose:** Handle RAG logic, database connections, and auth verification

### 3.2. API Endpoints
- `/api/chat` - Handle chat requests with RAG capabilities
- `/api/auth` - Authentication endpoints
- `/api/content` - Content-related APIs (if needed)
- `/api/embeddings` - Embedding generation and retrieval

### 3.3. AI Stack
- **Embeddings:** Google Gemini (Free Tier) or equivalent
- **Orchestration:** OpenAI Agents SDK / ChatKit
- **RAG Implementation:** Retrieval-Augmented Generation for textbook content

## 4. Integration Strategy

### 4.1. Frontend-Backend Communication
- REST API calls from frontend to backend
- Example: `POST http://localhost:8000/api/chat`
- Authentication headers for protected endpoints
- CORS configuration for local development

### 4.2. Data Flow
1. User interacts with textbook content in frontend
2. Chat requests sent to backend API
3. Backend processes with RAG using textbook embeddings
4. AI response returned to frontend
5. Authentication managed through better-auth client

## 5. Development Workflow

### 5.1. Local Development
- Run frontend: `cd frontend && npm run start`
- Run backend: `cd backend && uvicorn main:app --reload`
- Both services run on separate ports during development

### 5.2. Build & Deployment
- Frontend builds to static assets
- Backend deployed as API service
- Both can be containerized for deployment
- Environment configuration for different stages

## 6. Security Considerations

### 6.1. Authentication
- `better-auth` for frontend authentication
- JWT tokens for API authentication
- Secure session management

### 6.2. Data Protection
- Secure API endpoints
- Rate limiting
- Input validation
- Database security

## 7. Performance & Scalability

### 7.1. Frontend Performance
- Static site generation for textbook content
- Optimized asset loading
- Efficient component rendering

### 7.2. Backend Performance
- Vector database optimization for RAG
- Caching strategies
- Database indexing
- API response optimization

## 8. Future Considerations

### 8.1. Phase 2 Features
- Enhanced chatbot functionality
- Advanced authentication features
- Additional AI capabilities
- User progress tracking

### 8.2. Maintainability
- Clear separation of concerns between frontend and backend
- Consistent coding standards
- Comprehensive documentation
- Testing strategies for both applications