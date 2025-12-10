import React from 'react';
import ChatWidget from '@site/src/components/ChatWidget';

// Default implementation, that you can customize
export default function Root({children}) {
  console.log('Root Layout Mounted - Chat Widget should be visible');
  return (
    <>
      {children}
      <ChatWidget />
    </>
  );
}