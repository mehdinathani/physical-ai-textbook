import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Utility function for conditional class names
const cn = (...inputs: unknown[]) => {
  return twMerge(clsx(inputs));
};

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
}

// Dynamically import useChat to avoid SSR issues
const loadUseChat = async () => {
  const { useChat } = await import('ai');
  return useChat;
};

const ChatWidgetContent: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [chatData, setChatData] = useState<{
    messages: any[];
    input: string;
    handleInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
    isLoading: boolean;
    setInput: (input: string) => void;
  } | null>(null);

  // Dynamically import useChat hook to avoid SSR issues
  useEffect(() => {
    const loadChat = async () => {
      const useChat = await loadUseChat();
      const chatHook = useChat({
        api: 'https://physai-backend.onrender.com/api/chat',
        onError: (error) => {
          console.error('Chat error:', error);
        }
      });
      setChatData(chatHook);
    };

    if (typeof window !== 'undefined') {
      loadChat();
    }
  }, []);

  // Extract chat data from state
  const {
    messages = [],
    input = '',
    handleInputChange = () => {},
    handleSubmit = () => {},
    isLoading = false,
    setInput = () => {}
  } = chatData || {};

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  // Scroll to bottom of messages when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    handleSubmit(e);
  };

  return (
    <div
      className="fixed bottom-5 right-5 z-50"
      role="complementary"
      aria-label="AI Chat Widget"
    >
      {/* Floating Action Button */}
      {!isOpen && (
        <button
          onClick={toggleChat}
          className="bg-blue-600 text-white rounded-full w-14 h-14 flex items-center justify-center shadow-lg hover:bg-blue-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
          aria-label="Open chat"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-20 right-5 w-96 max-w-[90vw] h-[500px] flex flex-col bg-white border border-gray-200 rounded-lg shadow-xl overflow-hidden">
          {/* Header */}
          <div className="bg-blue-600 text-white p-4 flex justify-between items-center">
            <h3 className="font-semibold flex items-center gap-2">
              <Bot className="w-5 h-5" />
              PhysAI Assistant
            </h3>
            <button
              onClick={toggleChat}
              className="text-white hover:text-gray-200 focus:outline-none focus:ring-2 focus:ring-white focus:ring-opacity-50 rounded-full p-1"
              aria-label="Close chat"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          </div>

          {/* Messages Container */}
          <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
            {!chatData ? (
              <div className="h-full flex flex-col items-center justify-center text-gray-500">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
                  <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                </div>
                <p className="text-center mt-2">Loading chat...</p>
              </div>
            ) : messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-gray-500">
                <Bot className="w-12 h-12 mb-3 text-blue-500" />
                <p className="text-center">Ask me anything about Physical AI & Robotics!</p>
                <p className="text-sm mt-2 text-center">I can help explain concepts, provide examples, and answer questions.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message: any) => (
                  <div
                    key={message.id}
                    className={cn(
                      'flex items-start gap-3',
                      message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                    )}
                  >
                    {message.role === 'assistant' ? (
                      <div className="bg-blue-100 p-2 rounded-full flex items-center justify-center">
                        <Bot className="w-4 h-4 text-blue-600" />
                      </div>
                    ) : (
                      <div className="bg-gray-200 p-2 rounded-full flex items-center justify-center">
                        <User className="w-4 h-4 text-gray-600" />
                      </div>
                    )}
                    <div
                      className={cn(
                        'max-w-[75%] rounded-lg p-3',
                        message.role === 'user'
                          ? 'bg-blue-500 text-white rounded-tr-none'
                          : 'bg-white border border-gray-200 rounded-tl-none'
                      )}
                    >
                      {message.content}
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 p-2 rounded-full flex items-center justify-center">
                      <Bot className="w-4 h-4 text-blue-600" />
                    </div>
                    <div className="bg-white border border-gray-200 rounded-lg p-3 rounded-tl-none">
                      <div className="flex space-x-2">
                        <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
                        <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-200 p-4 bg-white">
            {chatData ? (
              <form onSubmit={onSubmit} className="flex gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={handleInputChange}
                  placeholder="Type your message..."
                  className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className={cn(
                    'bg-blue-600 text-white rounded-lg px-4 py-2 flex items-center justify-center',
                    'hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50',
                    'disabled:opacity-50 disabled:cursor-not-allowed',
                    'transition-colors duration-200'
                  )}
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            ) : (
              <div className="flex gap-2">
                <input
                  type="text"
                  value=""
                  onChange={() => {}}
                  placeholder="Loading chat..."
                  className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled
                />
                <button
                  type="button"
                  disabled
                  className="bg-gray-400 text-white rounded-lg px-4 py-2 flex items-center justify-center"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            )}
            <p className="text-xs text-gray-500 mt-2 text-center">
              PhysAI Assistant • Powered by Gemini
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

const ChatWidget: React.FC = () => {
  const [isClient, setIsClient] = React.useState(false);

  React.useEffect(() => {
    setIsClient(true);
  }, []);

  // Only render the content on the client side
  if (!isClient) {
    return (
      <div
        className="fixed bottom-5 right-5 z-50"
        role="complementary"
        aria-label="AI Chat Widget"
      >
        {/* Floating Action Button - placeholder during SSR */}
        <div className="bg-blue-600 text-white rounded-full w-14 h-14 flex items-center justify-center shadow-lg opacity-0" aria-label="Open chat">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        </div>
      </div>
    );
  }

  return <ChatWidgetContent />;
};

export default ChatWidget;