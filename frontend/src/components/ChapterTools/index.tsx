import React, { useState, useEffect } from 'react';
import './ChapterTools.css';

interface ContentTransformationRequest {
  source_content: string;
  transformation_type: 'urdu-translation' | 'hardware-personalization' | 'software-personalization';
  preserve_formatting: boolean;
}

interface ContentTransformationResponse {
  transformed_content: string;
  transformation_type: string;
  processing_time_ms: number;
}

const ChapterTools: React.FC = () => {
  const [originalContent, setOriginalContent] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [currentTransformation, setCurrentTransformation] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Effect to save original content when component mounts
  useEffect(() => {
    // Save original content when component mounts
    const mainContent = document.querySelector('main article');
    if (mainContent) {
      setOriginalContent(mainContent.innerHTML || '');
    } else {
      const contentContainer = document.querySelector('[role="main"]');
      if (contentContainer) {
        setOriginalContent(contentContainer.innerHTML || '');
      }
    }
  }, []);

  // Function to get the current page content as markdown
  const getCurrentPageContent = (): string => {
    // Try to get content from the main documentation area
    const mainContent = document.querySelector('main article');
    if (mainContent) {
      return mainContent.innerHTML || '';
    }

    // Fallback to the main content container
    const contentContainer = document.querySelector('[role="main"]');
    if (contentContainer) {
      return contentContainer.innerHTML || '';
    }

    // Fallback to body content
    return document.body.innerHTML || '';
  };

  // Function to replace the page content
  const replacePageContent = (newContent: string) => {
    // Try to find the main content area to replace
    const mainContent = document.querySelector('main article');
    if (mainContent) {
      // Set the innerHTML to preserve HTML structure
      mainContent.innerHTML = newContent;
      return;
    }

    // Fallback approach
    const contentContainer = document.querySelector('[role="main"]');
    if (contentContainer) {
      contentContainer.innerHTML = newContent;
      return;
    }

    // If we can't find a specific container, we'll show the content differently
    alert('Content transformed! (Note: Actual content replacement requires more specific DOM targeting)');
  };

  // Function to handle content transformation
  const handleTransform = async (transformationType: 'urdu-translation' | 'hardware-personalization' | 'software-personalization') => {
    setIsProcessing(true);
    setError(null);
    setCurrentTransformation(transformationType);

    try {
      // Get the current page content
      const content = getCurrentPageContent();

      // Save original content if not already saved
      if (!originalContent) {
        setOriginalContent(content);
      }

      // Prepare the transformation request
      const request: ContentTransformationRequest = {
        source_content: content,
        transformation_type: transformationType,
        preserve_formatting: true
      };

      // Call the backend API
      const response = await fetch('/api/transform', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const result: ContentTransformationResponse = await response.json();

      // Replace the page content with the transformed content
      replacePageContent(result.transformed_content);
    } catch (err) {
      console.error('Transformation error:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsProcessing(false);
    }
  };

  // Function to reset to original content
  const handleReset = () => {
    if (originalContent) {
      // Restore original content
      const mainContent = document.querySelector('main article');
      if (mainContent) {
        mainContent.innerHTML = originalContent;
      }

      const contentContainer = document.querySelector('[role="main"]');
      if (contentContainer) {
        contentContainer.innerHTML = originalContent;
      }

      setCurrentTransformation(null);
      setOriginalContent('');
    }
  };

  // Check if we're on a documentation page vs landing page
  useEffect(() => {
    // This is a simplified check - in a real implementation,
    // you might need a more sophisticated way to detect doc pages
    const isDocPage = window.location.pathname.includes('/docs/');
    if (!isDocPage) {
      // Hide the toolbar on non-doc pages
      const toolbar = document.querySelector('.chapter-tools');
      if (toolbar) {
        toolbar.style.display = 'none';
      }
    }
  }, []);

  return (
    <div className="chapter-tools">
      <div className="chapter-tools-container">
        <div className="chapter-tools-buttons">
          <button
            className={`tool-btn ${currentTransformation === 'urdu-translation' ? 'active' : ''}`}
            onClick={() => handleTransform('urdu-translation')}
            disabled={isProcessing}
            title="Translate to Urdu"
          >
            🇵🇰 Urdu
          </button>

          <button
            className={`tool-btn ${currentTransformation === 'hardware-personalization' ? 'active' : ''}`}
            onClick={() => handleTransform('hardware-personalization')}
            disabled={isProcessing}
            title="Hardware Engineer View"
          >
            ⚙️ Hardware
          </button>

          <button
            className={`tool-btn ${currentTransformation === 'software-personalization' ? 'active' : ''}`}
            onClick={() => handleTransform('software-personalization')}
            disabled={isProcessing}
            title="Software Engineer View"
          >
            💻 Software
          </button>

          <button
            className="tool-btn reset-btn"
            onClick={handleReset}
            disabled={!originalContent && !currentTransformation}
            title="Reset to original content"
          >
            🔄 Reset
          </button>
        </div>

        {isProcessing && (
          <div className="processing-indicator">
            Processing... <span className="spinner">⏳</span>
          </div>
        )}

        {error && (
          <div className="error-message">
            Error: {error}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChapterTools;