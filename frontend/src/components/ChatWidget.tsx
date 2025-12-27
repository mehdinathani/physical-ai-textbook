import React, { useState, useEffect } from 'react';
import { ChatKit, useChatKit } from '@openai/chatkit-react';
import { Bot } from 'lucide-react';

const ChatWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [initialThread, setInitialThread] = useState<string | undefined>(undefined);
  const [isReady, setIsReady] = useState(false);
  const [backendUrl, setBackendUrl] = useState<string>('http://localhost:8000/chatkit');
  const [domainKey, setDomainKey] = useState<string>('localhost');

  // ============================================================
  // DEBUGGING & ENVIRONMENT: Initialize client-side state
  // ============================================================
  useEffect(() => {
    console.log('[ChatWidget] Component mounted on client');
    console.log('[ChatWidget] typeof window:', typeof window);
    console.log('[ChatWidget] typeof import.meta:', typeof (globalThis as any).import?.meta);

    // CRITICAL: Only access import.meta.env inside useEffect (client-side only)
    try {
      // Use dynamic property access to avoid webpack parsing errors
      const envBackendUrl = (globalThis as any).__VITE_CHATKIT_BACKEND_URL__ ||
                           (typeof (globalThis as any).import !== 'undefined' &&
                            typeof (globalThis as any).import.meta !== 'undefined'
                            ? (globalThis as any).import.meta.env?.VITE_CHATKIT_BACKEND_URL
                            : undefined) ||
                           'http://localhost:8000/chatkit';

      const envDomainKey = (globalThis as any).__VITE_CHATKIT_DOMAIN_KEY__ ||
                          (typeof (globalThis as any).import !== 'undefined' &&
                           typeof (globalThis as any).import.meta !== 'undefined'
                           ? (globalThis as any).import.meta.env?.VITE_CHATKIT_DOMAIN_KEY
                           : undefined) ||
                          'localhost';

      console.log('[ChatWidget] VITE_CHATKIT_BACKEND_URL from env:', envBackendUrl);
      console.log('[ChatWidget] VITE_CHATKIT_DOMAIN_KEY from env:', envDomainKey);
      setBackendUrl(envBackendUrl);
      setDomainKey(envDomainKey);
    } catch (error) {
      console.warn('[ChatWidget] Could not access import.meta.env, using defaults:', error);
      setBackendUrl('http://localhost:8000/chatkit');
      setDomainKey('localhost');
    }
  }, []);

  // SSR-safe loading: Load thread ID from localStorage only on client
  useEffect(() => {
    console.log('[ChatWidget] useEffect: Checking client-side environment...');

    if (typeof window === 'undefined') {
      console.log('[ChatWidget] SSR detected - window is undefined, skipping localStorage');
      return;
    }

    try {
      const savedThreadId = localStorage.getItem('physai-chatkit-thread-id');
      console.log('[ChatWidget] Retrieved saved thread ID from localStorage:', savedThreadId);
      setInitialThread(savedThreadId || undefined);
      setIsReady(true);
      console.log('[ChatWidget] isReady set to true - component ready for rendering');
    } catch (error) {
      console.error('[ChatWidget] Error accessing localStorage:', error);
      console.log('[ChatWidget] Proceeding without thread persistence');
      setIsReady(true); // Still proceed even if localStorage fails
    }
  }, []);

  console.log('[ChatWidget] Backend URL resolved to:', backendUrl);

  // ============================================================
  // CRITICAL: Always call hooks unconditionally (React Rules)
  // useChatKit must be called every render, not conditionally
  // ============================================================
  let chatKit: any = null;
  let hookError: any = null;

  try {
    console.log('[ChatWidget] 🔄 About to call useChatKit hook...');
    console.log('[ChatWidget]   backendUrl:', backendUrl);
    console.log('[ChatWidget]   initialThread:', initialThread);

    // ============================================================
    // CRITICAL: ChatKit API configuration
    // CustomApiConfig requires BOTH url AND domainKey (validation requirement)
    // For localhost, domain verification is skipped automatically
    // ============================================================
    console.log('[ChatWidget]   Configuring useChatKit...');
    console.log('[ChatWidget]   Using backend URL:', backendUrl);

    // CustomApiConfig with required fields
    chatKit = useChatKit({
      api: {
        url: backendUrl,
        domainKey: domainKey,  // Domain key from environment (localhost for dev, production key for prod)
      },
      theme: 'dark',
      initialThread: initialThread || undefined,
    });

    console.log('[ChatWidget]   useChatKit initialized successfully (development mode)');
    console.log('[ChatWidget]   API config:', { url: backendUrl, mode: 'development' });
    console.log('[ChatWidget]   Theme:', 'dark');

    console.log('[ChatWidget] ✅ useChatKit hook called successfully');
    console.log('[ChatWidget]   chatKit type:', typeof chatKit);
    console.log('[ChatWidget]   chatKit keys:', chatKit ? Object.keys(chatKit) : 'N/A');
    console.log('[ChatWidget]   chatKit.control exists:', chatKit ? !!chatKit.control : 'N/A');
    if (chatKit && chatKit.control) {
      console.log('[ChatWidget]   control.options:', chatKit.control.options);
    }
  } catch (error) {
    console.error('[ChatWidget] ❌ ERROR calling useChatKit hook:', error);
    console.error('[ChatWidget]   Error message:', error instanceof Error ? error.message : String(error));
    console.error('[ChatWidget]   Error stack:', error instanceof Error ? error.stack : 'N/A');
    hookError = error;
    chatKit = null;
  }

  const toggleChat = () => {
    console.log('[ChatWidget] Toggle chat - isOpen:', isOpen, '-> ', !isOpen);
    setIsOpen(!isOpen);
  };

  const handleNewChat = () => {
    console.log('[ChatWidget] Starting new chat - clearing thread ID');
    try {
      localStorage.removeItem('physai-chatkit-thread-id');
      // Reload the page to start fresh
      window.location.reload();
    } catch (error) {
      console.error('[ChatWidget] Error clearing thread ID:', error);
    }
  };

  console.log('[ChatWidget] ✅ RENDERING WIDGET - isReady:', isReady, 'chatKit exists:', !!chatKit, 'isOpen:', isOpen);

  // ALWAYS render the button, even if chatKit isn't ready yet
  // The chat window only appears when both chatKit is ready AND isOpen is true
  return (
    <div
      role="complementary"
      aria-label="AI Chat Widget"
      style={{ position: 'fixed', bottom: '20px', right: '20px', zIndex: 9999, display: 'block' }}
    >
      {/* Floating Action Button - ALWAYS VISIBLE */}
      {!isOpen && (
        <button
          onClick={toggleChat}
          style={{
            backgroundColor: '#2563eb',
            color: 'white',
            borderRadius: '50%',
            width: '56px',
            height: '56px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
            cursor: 'pointer',
            border: 'none',
            transition: 'background-color 0.2s'
          }}
          aria-label="Open chat"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        </button>
      )}

      {/* Chat Window - Only when chatKit is ready AND isOpen is true */}
      {isOpen && chatKit && chatKit.control && (
        <div
          style={{
            position: 'fixed',
            bottom: '80px',
            right: '20px',
            width: '384px',
            maxWidth: '90vw',
            height: '500px',
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: 'white',
            border: '1px solid #e5e7eb',
            borderRadius: '0.5rem',
            boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
            overflow: 'hidden'
          }}
        >
          {/* Header */}
          <div style={{ backgroundColor: '#2563eb', color: 'white', padding: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
            <h3 style={{ fontWeight: '600', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Bot width={20} height={20} />
              PhysAI Assistant
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <button
                onClick={handleNewChat}
                style={{ backgroundColor: 'rgba(255, 255, 255, 0.2)', border: 'none', color: 'white', cursor: 'pointer', padding: '0.5rem 0.75rem', borderRadius: '0.25rem', fontSize: '0.875rem', fontWeight: '500' }}
                aria-label="Start new chat"
                title="Start new chat"
              >
                New Chat
              </button>
              <button
                onClick={toggleChat}
                style={{ backgroundColor: 'transparent', border: 'none', color: 'white', cursor: 'pointer', padding: '0.25rem', borderRadius: '50%' }}
                aria-label="Close chat"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          </div>

          {/* ChatKit Component */}
          <div style={{ flex: '1 1 auto', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <ChatKit
              control={chatKit.control}
              style={{
                display: 'flex',
                flexDirection: 'column',
                width: '100%',
                height: '100%',
                flex: '1 1 0'
              }}
            />
          </div>
        </div>
      )}

      {/* Loading state when chat is open but chatKit isn't ready */}
      {isOpen && (!chatKit || !chatKit.control) && (
        <div
          style={{
            position: 'fixed',
            bottom: '80px',
            right: '20px',
            width: '384px',
            maxWidth: '90vw',
            height: '500px',
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: 'white',
            border: '1px solid #e5e7eb',
            borderRadius: '0.5rem',
            boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
            overflow: 'hidden'
          }}
        >
          <div style={{ backgroundColor: '#2563eb', color: 'white', padding: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
            <h3 style={{ fontWeight: '600', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Bot width={20} height={20} />
              PhysAI Assistant
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <button
                onClick={handleNewChat}
                style={{ backgroundColor: 'rgba(255, 255, 255, 0.2)', border: 'none', color: 'white', cursor: 'pointer', padding: '0.5rem 0.75rem', borderRadius: '0.25rem', fontSize: '0.875rem', fontWeight: '500' }}
                aria-label="Start new chat"
                title="Start new chat"
              >
                New Chat
              </button>
              <button
                onClick={toggleChat}
                style={{ backgroundColor: 'transparent', border: 'none', color: 'white', cursor: 'pointer', padding: '0.25rem', borderRadius: '50%' }}
                aria-label="Close chat"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          </div>
          <div style={{ padding: '1rem', textAlign: 'center', color: '#6b7280' }}>
            Loading chat...
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatWidget;
