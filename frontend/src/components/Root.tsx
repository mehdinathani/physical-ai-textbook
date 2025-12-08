import React from 'react';
import ChatWidget from './components/ChatWidget';

// Root component that wraps the entire application
const Root: React.FC<{children: React.ReactNode}> = ({ children }) => {
  return (
    <>
      {children}
      <ChatWidget />
    </>
  );
};

export default Root;