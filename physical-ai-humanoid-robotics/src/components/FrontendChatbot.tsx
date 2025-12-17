import React, { useEffect } from 'react';
import '@n8n/chat/style.css';
import { createChat } from '@n8n/chat';

const FrontendChatbot = () => {
  useEffect(() => {
    // Initialize the n8n chat with your webhook URL
    createChat({
      webhookUrl: 'https://muhamadumair.app.n8n.cloud/webhook/0c6908d3-bcdb-420f-98ba-a97ca8615595/chat'
    });
  }, []);

  return (
    <div
      id="n8n-chat-container"
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        zIndex: 1000
      }}
    />
  );
};

export default FrontendChatbot;
