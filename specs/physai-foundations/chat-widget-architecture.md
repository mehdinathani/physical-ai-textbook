# Phase 3 Architecture: Chat Widget Implementation

## Overview
This document outlines the architecture for the floating "Chat with AI" widget that will be integrated into the Docusaurus-based Physical AI & Humanoid Robotics Textbook website. The widget provides users with an AI-powered chat interface that can answer questions about the textbook content.

## 1. Tech Stack

### Frontend Technologies
- **React**: Component-based UI library for building the chat interface
- **TypeScript**: Type-safe development for improved maintainability
- **Tailwind CSS**: Utility-first CSS framework for responsive styling
- **Docusaurus**: Static site generator for the textbook website

### Dependencies
- **react-markdown**: For rendering markdown content from AI responses
- **remark-gfm**: GitHub Flavored Markdown support for react-markdown
- **lucide-react**: Icon library for UI elements (if needed)

### Icons
- Using inline SVG icons instead of external icon libraries for better performance and smaller bundle size

## 2. Component Architecture

### ChatWidget Component Structure
```
ChatWidget (Main Component)
├── Floating Action Button (FAB)
├── Chat Window Container
│   ├── Header (with title and close button)
│   ├── Messages Container (scrollable area)
│   │   ├── User Message Component
│   │   └── Assistant Message Component
│   └── Input Area
│       ├── Text Input
│       └── Send Button
```

### Component Responsibilities
- **ChatWidget**: Main container managing state and orchestrating the UI
- **Message Display**: Renders user and AI messages with proper styling
- **Input Area**: Handles user input and submission
- **FAB**: Toggles the visibility of the chat window

## 3. State Management

### React State Hooks
The component uses simple React state management with the following state variables:

```typescript
const [isOpen, setIsOpen] = useState(false);          // Chat window visibility
const [messages, setMessages] = useState<Message[]>(); // Message history
const [inputValue, setInputValue] = useState('');      // Current input text
const [isLoading, setIsLoading] = useState(false);     // API request status
```

### Message Interface
```typescript
interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}
```

### State Flow
1. User toggles chat → `isOpen` state changes
2. User types message → `inputValue` state updates
3. User sends message → `isLoading` set to true, message added to `messages`
4. API response received → AI message added to `messages`, `isLoading` set to false

## 4. API Integration

### Backend Communication
- **API Endpoint**: `POST /api/chat`
- **Request Format**: `{ message: string }`
- **Response Format**: `{ response: string }`
- **Base URL**: `http://localhost:8000` (placeholder, will be configurable for production)

### API Client Implementation
- Uses native `fetch` API for HTTP requests
- Error handling with try-catch blocks
- Loading state management during API calls
- Proper error message display for failed requests

### API Request Flow
1. User submits message via chat input
2. Component sends POST request to backend API
3. Component shows loading indicator
4. Backend processes request and returns response
5. Component adds response to message history
6. Loading indicator is removed

## 5. Docusaurus Integration Strategy

### Global Component Injection
The chat widget is integrated using Docusaurus' theme swizzling capability:

#### Approach: Root Component
- Create `src/theme/Root.tsx` component
- This component automatically wraps the entire Docusaurus application
- ChatWidget is rendered as a sibling to the main content
- Ensures the widget appears on every page of the textbook

#### Component Injection Method
```
Docusaurus App
├── Root Component (wraps entire app)
│   ├── Original Content (children)
│   └── ChatWidget (floating component)
```

### Styling Integration
- Uses Tailwind CSS for styling that adapts to Docusaurus' dark/light mode
- Z-index management to ensure widget appears above other content
- Responsive design that works on all device sizes
- CSS variables to match Docusaurus' color scheme

## 6. UI/UX Design

### Floating Action Button (FAB)
- Fixed position at bottom-right corner
- Circular design with chat icon
- Smooth hover animations
- Accessible with proper ARIA labels

### Chat Window
- Sliding animation when opening/closing
- Header with title and close button
- Scrollable message area with auto-scroll to latest message
- Different styling for user vs AI messages
- Loading indicators during API requests

### Responsive Design
- Full functionality on desktop and mobile
- Appropriate sizing for different screen dimensions
- Touch-friendly controls for mobile devices

## 7. Security Considerations

### Input Sanitization
- Messages are properly escaped to prevent XSS
- Content Security Policy compliance

### API Security
- No sensitive data stored in client-side state
- API calls use HTTPS in production
- Rate limiting considerations for future implementation

## 8. Performance Optimization

### Rendering Optimization
- Efficient state updates to prevent unnecessary re-renders
- Virtualized message list for large conversation histories (future enhancement)
- Memoization of components where appropriate

### Bundle Size
- Minimal external dependencies
- Tree-shaking friendly imports
- Lazy loading considerations for future enhancements

## 9. Error Handling

### Network Errors
- Graceful degradation when API is unavailable
- User-friendly error messages
- Retry mechanism considerations

### User Experience
- Loading states during API requests
- Clear feedback for user actions
- Error recovery options

## 10. Future Enhancements

### Potential Additions
- Conversation history persistence
- Typing indicators
- File/image upload capabilities
- Voice input support
- Multi-language support
- Conversation threading

This architecture provides a solid foundation for the chat widget while maintaining compatibility with the Docusaurus framework and ensuring good performance and user experience.