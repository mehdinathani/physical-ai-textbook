import React from 'react';
import Head from '@docusaurus/Head';
import BrowserOnly from '@docusaurus/BrowserOnly';
import { useLocation } from '@docusaurus/router';
import ChatWidget from '@site/src/components/ChatWidget';
import ChapterTools from '@site/src/components/ChapterTools';

// ============================================================
// CHAPTER TOOLS WRAPPER: Safely uses useLocation hook
// ============================================================
function ChapterToolsWrapper() {
  console.log('[Root] ChapterToolsWrapper rendering');
  try {
    const location = useLocation();
    const isDocPage = location.pathname.includes('/docs/');

    console.log('[Root] Current path:', location.pathname);
    console.log('[Root] isDocPage:', isDocPage);

    if (isDocPage) {
      return <ChapterTools />;
    }
    return null;
  } catch (error) {
    console.error('[Root] Error in ChapterToolsWrapper:', error);
    return null;
  }
}

// ============================================================
// ROOT COMPONENT: Docusaurus layout wrapper
// ============================================================
export default function Root({children}) {
  console.log('[Root] Component mounted');

  return (
    <>
      {/* ============================================================ */}
      {/* CDN SCRIPT INJECTION: Load ChatKit library */}
      {/* ============================================================ */}
      <Head>
        <script
          src="https://cdn.platform.openai.com/deployments/chatkit/chatkit.js"
          async
          onLoad={() => console.log('[Root] ChatKit CDN script loaded')}
          onError={() => console.error('[Root] Failed to load ChatKit CDN script')}
        />
      </Head>

      {/* Page content */}
      {children}

      {/* ============================================================ */}
      {/* CHAT WIDGET: Browser-only component (SSR-safe) */}
      {/* ============================================================ */}
      <BrowserOnly
        fallback={<div style={{ display: 'none' }} />}
      >
        {() => {
          console.log('[Root] BrowserOnly rendering ChatWidget');
          try {
            return <ChatWidget />;
          } catch (error) {
            console.error('[Root] Error rendering ChatWidget:', error);
            return null;
          }
        }}
      </BrowserOnly>

      {/* ============================================================ */}
      {/* CHAPTER TOOLS: Only on documentation pages */}
      {/* Wrapped in BrowserOnly - hooks called in separate component */}
      {/* ============================================================ */}
      <BrowserOnly
        fallback={<div style={{ display: 'none' }} />}
      >
        {() => {
          console.log('[Root] BrowserOnly rendering ChapterToolsWrapper');
          return <ChapterToolsWrapper />;
        }}
      </BrowserOnly>
    </>
  );
}
