import React from 'react';
import { useLocation } from '@docusaurus/router';
import ChatWidget from '@site/src/components/ChatWidget';
import ChapterTools from '@site/src/components/ChapterTools';

// Default implementation, that you can customize
export default function Root({children}) {
  console.log('Root Layout Mounted - Chat Widget should be visible');
  const location = useLocation();

  // Check if we're on a documentation page
  const isDocPage = location.pathname.includes('/docs/');

  return (
    <>
      {isDocPage && <ChapterTools />}
      {children}
      <ChatWidget />
    </>
  );
}