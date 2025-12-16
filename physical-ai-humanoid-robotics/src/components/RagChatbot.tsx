import React, { useState, useRef } from 'react';
import './RagChatbot.css';

const RagChatbot = () => {
  const [selectedText, setSelectedText] = useState('');
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [showChat, setShowChat] = useState(false);
  const textRef = useRef(null);

  // Function to handle text selection
  const handleTextSelection = () => {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    if (selectedText.length > 0) {
      setSelectedText(selectedText);
      setShowChat(true);
    }
  };

  // Function to send query to the backend
  const sendQuery = async () => {
    if (!query.trim() || !selectedText) {
      alert('Please select some text and enter a query first.');
      return;
    }

    setIsLoading(true);

    try {
      // First, store the selected text
      const textResponse = await fetch('/api/text-selection/', {
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
        throw new Error('Failed to store selected text');
      }

      // Then, send the query
      const queryResponse = await fetch('/api/query/', {
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
        throw new Error('Failed to get response from chatbot');
      }

      const data = await queryResponse.json();
      setResponse(data.response);

      // Add to conversation history
      const newConversation = {
        query: query,
        response: data.response,
        timestamp: new Date().toLocaleTimeString()
      };
      setConversationHistory(prev => [newConversation, ...prev]);

      // Clear the query input
      setQuery('');
    } catch (error) {
      console.error('Error:', error);
      setResponse('Sorry, there was an error processing your request. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Enter key press in query input
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  };

  // Function to clear conversation
  const clearConversation = () => {
    setConversationHistory([]);
    setResponse('');
    setQuery('');
  };

  return (
    <div className="rag-chatbot">
      {/* Floating button to open chat */}
      {!showChat && (
        <button
          className="chatbot-toggle"
          onClick={() => setShowChat(true)}
        >
          💬 AI Assistant
        </button>
      )}

      {showChat && (
        <div className="chatbot-container">
          <div className="chatbot-header">
            <h3>Physical AI & Robotics Assistant</h3>
            <div className="header-actions">
              <button
                className="clear-btn"
                onClick={clearConversation}
                title="Clear conversation"
              >
                🗑️
              </button>
              <button
                className="close-btn"
                onClick={() => setShowChat(false)}
                title="Close chat"
              >
                ✕
              </button>
            </div>
          </div>

          <div className="chatbot-content">
            {selectedText && (
              <div className="selected-text-preview">
                <strong>Selected Text:</strong>
                <p>"{selectedText.length > 100 ? selectedText.substring(0, 100) + '...' : selectedText}"</p>
              </div>
            )}

            <div className="input-section">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question about the selected text..."
                rows="3"
              />
              <button
                onClick={sendQuery}
                disabled={isLoading || !query.trim() || !selectedText}
                className="send-btn"
              >
                {isLoading ? 'Sending...' : 'Send'}
              </button>
            </div>

            {response && (
              <div className="response-section">
                <div className="response">
                  <strong>AI Response:</strong>
                  <p>{response}</p>
                </div>
              </div>
            )}

            {conversationHistory.length > 0 && (
              <div className="conversation-history">
                <h4>Conversation History:</h4>
                {conversationHistory.map((item, index) => (
                  <div key={index} className="history-item">
                    <div className="history-query">
                      <strong>Q:</strong> {item.query}
                    </div>
                    <div className="history-response">
                      <strong>A:</strong> {item.response}
                    </div>
                    <small className="timestamp">{item.timestamp}</small>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Instructional overlay for text selection */}
      <div className="selection-instruction">
        <p>Select text in the textbook to ask questions about it</p>
      </div>
    </div>
  );
};

export default RagChatbot;