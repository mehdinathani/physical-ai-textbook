<!--
Sync Impact Report:
- Version change: 1.0.0 → 1.1.0
- Added sections: Spec-Driven Execution, Bilingual Excellence, Hackathon Velocity, AI-Native First principles
- Modified principles: Academic Rigor → Spec-Driven Execution, Hackathon Efficiency → Hackathon Velocity, Code Quality → Bilingual Excellence, Performance Efficiency → AI-Native First
- Templates requiring updates: ✅ .specify/templates/plan-template.md, ✅ .specify/templates/spec-template.md, ✅ .specify/templates/tasks-template.md
- Follow-up TODOs: None
-->
# PhysAI Foundations Constitution

## Core Principles

### Spec-Driven Execution
All code changes must originate from updated `specs/` files (Plan -> Task -> Implement). Every feature, bug fix, and enhancement must be traced back to a documented specification in the `specs/` directory to ensure alignment with project goals and maintainable development practices.

### Bilingual Excellence
Maintain strict separation of concerns: TypeScript for Frontend (Docusaurus) and Python for Backend (FastAPI). Ensure code quality, documentation, and architecture follow language-specific best practices while maintaining interoperability between systems.

### Hackathon Velocity
Prioritize "Working Deployment" over "Perfect Abstraction." Focus on delivering functional solutions that meet the core requirements within time constraints. Choose proven, simple solutions that can be deployed quickly while maintaining extensibility for future enhancements.

### AI-Native First
Leverage LLMs (Gemini/OpenAI) for both content generation and the RAG intelligence layer. Build systems that natively integrate AI capabilities from the ground up, enabling intelligent content delivery and interactive learning experiences.

### Academic Rigor
The content (textbook) must be technically accurate, professional, and structured for university students. All documentation, examples, and explanations must maintain high educational standards with clear learning objectives, verifiable facts, and proper scientific foundations. Content must accurately reflect official ROS 2, NVIDIA Isaac, and VLA documentation.

### Modular Architecture
Build the book structure (Docusaurus) so that adding the Chatbot and Auth in Phase 2 will be seamless. Maintain clear separation of concerns with well-defined interfaces between components to enable future expansion and parallel development.

### Context Awareness
Always check `specs/tasks.md` before starting work to maintain history and ensure alignment with project requirements. Follow established patterns and maintain consistency with existing architecture and codebase.

### Clean UX
Prioritize readability, high contrast, and intuitive navigation in all user-facing interfaces. Implement consistent design patterns, accessible color schemes, and clear information architecture to enhance the learning experience. Chat widgets must be unobtrusive, responsive, and Z-index optimized.

### Code Quality
Enforce strict typing, appropriate error handling, and follow established best practices for maintainability and extensibility. All code must be clean, well-documented, modular, and testable. Use TypeScript strict mode and Python best practices consistently.

### Performance Efficiency
Optimize for fast loading times and smooth interactions. Minimize bundle sizes, implement proper caching strategies, and ensure responsive design across all device types to support the educational experience. Maintain chatbot response latency under 3 seconds.

## Development Standards
All contributions must follow TypeScript strict mode, include appropriate unit tests for custom functionality, and maintain accessibility standards (WCAG AA minimum). Documentation must be comprehensive and updated with each feature addition. Code must be modular, commented, and lint-free.

## Technology Stack
Primary technologies include Docusaurus (UI), FastAPI (API), Qdrant (Vector DB), TypeScript for type safety, Python for backend services, React for component development, and standard web technologies (HTML5, CSS3) for presentation. External dependencies should be evaluated for security, maintenance, and compatibility before integration. Must utilize Free Tier services only (Google Gemini, Qdrant Cloud Free, Render Free).

## Content Requirements
Textbook content must accurately reflect official ROS 2, NVIDIA Isaac, and VLA documentation. All examples, tutorials, and explanations must be technically precise and professionally structured for university-level education. Content should be expandable for Urdu translation and personalization features.

## Governance
This constitution serves as the authoritative guide for all development decisions in the PhysAI Foundations project. All feature implementations, architectural decisions, and code contributions must align with these principles. Deviations require explicit justification and team consensus.

Zero-tolerance for build failures on Vercel (Frontend) or Render (Backend). All deployments must pass automated checks before merging.

Version changes follow semantic versioning: MAJOR for principle changes, MINOR for additions, PATCH for clarifications. Amendments require documentation of rationale and team approval.

**Version**: 1.1.0 | **Ratified**: 2025-12-06 | **Last Amended**: 2025-12-13
