import React, { useState, useEffect } from 'react';
import './RagChatbot.css';

const RagChatbot = () => {
  const [selectedText, setSelectedText] = useState('');
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversation, setConversation] = useState([]);
  const [isMinimized, setIsMinimized] = useState(false);

  // Function to handle text selection from the textbook
  const handleTextSelection = () => {
    const selectedTextObj = window.getSelection();
    const text = selectedTextObj.toString().trim();

    if (text) {
      setSelectedText(text);
      // Optionally store in state or send to backend immediately
      console.log('Selected text:', text);
    }
  };

  // Function to send query to the backend
  const sendQueryToBackend = async (selectedText, query) => {
    setIsLoading(true);

    try {
      // First, store the selected text
      const textResponse = await fetch('http://localhost:8000/api/text-selection/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          selected_text: selectedText,
          query: query
        }),
      });

      if (!textResponse.ok) {
        throw new Error(`Text storage failed: ${textResponse.statusText}`);
      }

      // Then, query the chatbot
      const queryResponse = await fetch('http://localhost:8000/api/query/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          selected_text: selectedText,
          query: query
        }),
      });

      if (!queryResponse.ok) {
        throw new Error(`Query failed: ${queryResponse.statusText}`);
      }

      const data = await queryResponse.json();
      return data;
    } catch (error) {
      console.error('Error communicating with backend:', error);
      return {
        response: `Error: ${error.message}. Please make sure the backend server is running.`,
        context_used: selectedText
      };
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedText || !query.trim()) {
      alert('Please select text from the textbook and enter a query.');
      return;
    }

    // Add current query to conversation
    const newMessage = {
      query: query,
      selectedText: selectedText,
      timestamp: new Date().toLocaleTimeString()
    };

    setConversation(prev => [...prev, newMessage]);

    // Send to backend
    const result = await sendQueryToBackend(selectedText, query);

    setResponse(result.response);

    // Clear query input
    setQuery('');
  };

  // Add event listener for text selection
  useEffect(() => {
    const handleMouseUp = () => {
      handleTextSelection();
    };

    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  return (
    <div className={`rag-chatbot ${isMinimized ? 'minimized' : ''}`}>
      <div className="chatbot-header" onClick={() => setIsMinimized(!isMinimized)}>
        <h3>Physical AI & Robotics Chatbot</h3>
        <button className="minimize-btn">
          {isMinimized ? '+' : '−'}
        </button>
      </div>

      {!isMinimized && (
        <div className="chatbot-content">
          <div className="selected-text-preview">
            <h4>Selected Text:</h4>
            <p className="text-preview">
              {selectedText ? `"${selectedText.substring(0, 100)}${selectedText.length > 100 ? '...' : ''}"` :
               'No text selected. Highlight text in the textbook to use as context.'}
            </p>
          </div>

          <form onSubmit={handleSubmit} className="query-form">
            <div className="input-group">
              <label htmlFor="query">Ask about the selected text:</label>
              <textarea
                id="query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your question about the selected text..."
                rows="3"
                disabled={!selectedText}
              />
            </div>

            <button
              type="submit"
              disabled={!selectedText || !query.trim() || isLoading}
              className="submit-btn"
            >
              {isLoading ? 'Processing...' : 'Ask Question'}
            </button>
          </form>

          {response && (
            <div className="response-section">
              <h4>Response:</h4>
              <div className="response-content">
                {response}
              </div>
            </div>
          )}

          {conversation.length > 0 && (
            <div className="conversation-history">
              <h4>Conversation History:</h4>
              {conversation.map((msg, index) => (
                <div key={index} className="conversation-item">
                  <p><strong>Q:</strong> {msg.query}</p>
                  <p className="timestamp">{msg.timestamp}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default RagChatbot;