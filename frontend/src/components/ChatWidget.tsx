import React, { useState } from 'react';
import { ChatKit, useChatKit } from '@openai/chatkit-react';

const ChatWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  // Note: ChatKit expects a different API structure than our backend
  // We'll need to implement a custom API adapter or use a different approach
  // For now, using a placeholder configuration
  const { control } = useChatKit({
    api: {
      getClientSecret: async () => {
        // This is a placeholder - in a real implementation,
        // we would need to adapt our backend to match ChatKit's expected format
        // or create an intermediate service to translate between formats
        return 'placeholder-secret';
      },
    },
  });

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div style={{
      position: 'fixed',
      bottom: '20px',
      right: '20px',
      zIndex: 9999
    }} role="complementary" aria-label="AI Chat Widget">
      {/* Floating Action Button */}
      {!isOpen && (
        <button
          onClick={toggleChat}
          style={{
            backgroundColor: '#2563eb', // Bright Blue
            color: 'white',
            borderRadius: '50%',
            width: '60px',
            height: '60px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
          }}
          aria-label="Open chat"
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              toggleChat();
            }
          }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            bottom: '90px',
            right: '20px',
            display: 'flex',
            flexDirection: 'column',
            height: '500px',
            width: '380px',
            maxWidth: '100%',
            backgroundColor: 'white',
            color: 'inherit',
            borderRadius: '8px',
            boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04)',
            border: '1px solid #e5e7eb',
            overflow: 'hidden'
          }}
          role="dialog"
          aria-modal="true"
          aria-labelledby="chat-title"
        >
          {/* Header */}
          <div className="bg-blue-600 text-white p-4 flex justify-between items-center">
            <h3 id="chat-title" className="font-semibold">Chat with AI</h3>
            <button
              onClick={toggleChat}
              className="text-white hover:text-gray-200 focus:outline-none focus:ring-2 focus:ring-white focus:ring-opacity-50 rounded-full p-1"
              aria-label="Close chat"
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  toggleChat();
                }
              }}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          </div>

          {/* ChatKit Component */}
          <div className="flex-1 overflow-hidden">
            <ChatKit
              control={control}
              className="h-full w-full"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatWidget;