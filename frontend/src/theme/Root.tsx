import React from 'react';
import ChatWidget from '../components/ChatWidget';

// Root component that wraps the entire Docusaurus application
const Root = ({ children }: { children: React.ReactNode }) => {
  return (
    <>
      {children}
      <ChatWidget />
    </>
  );
};

export default Root;