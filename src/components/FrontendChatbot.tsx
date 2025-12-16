import React, { useEffect } from 'react';
import '@n8n/chat/style.css';
import { createChat } from '@n8n/chat';

const FrontendChatbot = () => {
  useEffect(() => {
    // Initialize the n8n chat with the webhook URL
    createChat({
      webhookUrl: 'https://muhamadumair.app.n8n.cloud/webhook/0c6908d3-bcdb-420f-98ba-a97ca8615595/chat'
    });
  }, []);

  // The n8n chat widget will be injected into the DOM automatically
  // We return an empty div as the container for the chat widget
  return <div />;
};

export default FrontendChatbot;
