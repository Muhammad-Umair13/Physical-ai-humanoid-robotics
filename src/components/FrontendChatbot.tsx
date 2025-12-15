import React, { useState, useRef, useEffect } from 'react';
import './FrontendChatbot.css';

const FrontendChatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm your Physical AI & Robotics assistant. Ask me anything about the textbook content!", sender: 'bot', timestamp: new Date() }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Function to send user question to backend
  const handleSend = async () => {
    if (inputValue.trim() === '' || isLoading) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Send question to Python backend
      const res = await fetch(
        'https://muhamadumair.app.n8n.cloud/webhook/chatbot',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: inputValue })
        }
      );

      const data = await res.json();


      // Add backend response as bot message
      const botMessage = {
        id: Date.now() + 1,
        text: data.answer || data.output || data.text || "No response from AI",
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      // Show error if backend fails
      const botMessage = {
        id: Date.now() + 1,
        text: "Error contacting backend.",
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
    } finally {
      setIsLoading(false);
      setInputValue('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className={`frontend-chatbot ${isOpen ? 'active' : ''}`}>
      {!isOpen ? (
        <button
          className="chatbot-toggle"
          onClick={() => setIsOpen(true)}
          aria-label="Open chatbot"
        >
          <span className="chatbot-icon">🤖</span>
        </button>
      ) : (
        <div className="chatbot-container">
          <div className="chatbot-header">
            <h3>Physical AI Assistant</h3>
            <div className="header-actions">
              <button
                className="minimize-btn"
                onClick={() => setIsOpen(false)}
                aria-label="Minimize chatbot"
              >
                −
              </button>
            </div>
          </div>

          <div className="chatbot-messages">
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.sender}`}>
                <div className="message-content">{message.text}</div>
                <div className="message-timestamp">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message bot">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chatbot-input">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about robotics concepts..."
              rows={1}
              disabled={isLoading}
            />
            <button
              onClick={handleSend}
              disabled={inputValue.trim() === '' || isLoading}
              className="send-button"
            >
              ➤
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FrontendChatbot;
