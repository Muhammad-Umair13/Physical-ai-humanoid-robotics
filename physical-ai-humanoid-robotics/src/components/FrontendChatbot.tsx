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

  // Knowledge base for the chatbot
  const knowledgeBase = {
    ros: [
      "ROS (Robot Operating System) is a flexible framework for writing robot software. It provides services designed for a heterogeneous computer cluster such as hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.",
      "ROS uses a distributed architecture where multiple processes (nodes) can communicate with each other through messages passed over topics, services, and actions.",
      "ROS 2 is the next generation of ROS, designed to provide scalability, security, and maintainability for commercial applications."
    ],
    'ros2': [
      "ROS 2 (Robot Operating System 2) is the next generation of ROS, designed to provide scalability, security, and maintainability for commercial applications.",
      "ROS 2 uses DDS (Data Distribution Service) as the middleware for communication between nodes.",
      "ROS 2 introduces features like Quality of Service (QoS) policies, security, and multi-language support."
    ],
    'nodes': [
      "In ROS, a node is an executable that uses ROS to communicate with other nodes.",
      "Nodes are typically written in C++ or Python and use the ROS client library (roscpp or rospy).",
      "Nodes publish messages to topics and subscribe to topics to receive messages."
    ],
    'topics': [
      "Topics in ROS are named buses over which nodes exchange messages.",
      "Topics enable asynchronous message passing between nodes using a publish/subscribe communication pattern.",
      "Messages are published to topics by publisher nodes and received by subscriber nodes."
    ],
    'services': [
      "Services in ROS provide a request/reply communication pattern.",
      "Unlike topics which are asynchronous, services are synchronous and block until a response is received.",
      "Services are useful for operations that require a direct response, such as calculations or configuration changes."
    ],
    'actions': [
      "Actions in ROS provide a way to send a goal to a server and receive feedback during execution.",
      "Actions are ideal for long-running tasks that may take a significant amount of time to complete.",
      "Actions include goal, feedback, and result messages for comprehensive task management."
    ],
    'digital twin': [
      "A digital twin is a virtual representation of a physical system that mirrors its properties and behaviors.",
      "In robotics, digital twins are used for simulation, testing, and validation before deploying to real robots.",
      "Gazebo and Unity are popular platforms for creating digital twins of robotic systems."
    ],
    'gazebo': [
      "Gazebo is a 3D simulation environment that provides realistic physics, high-quality graphics, and convenient programmatic interfaces.",
      "Gazebo integrates well with ROS and provides a wide range of sensors and models for robotics simulation.",
      "Gazebo is commonly used for testing robot algorithms in a safe and cost-effective environment."
    ],
    'navigation': [
      "Robot navigation involves path planning, localization, and obstacle avoidance.",
      "The ROS Navigation Stack provides tools for autonomous navigation of mobile robots.",
      "Navigation includes global path planning and local path planning to reach goals while avoiding obstacles."
    ],
    'slam': [
      "SLAM (Simultaneous Localization and Mapping) is the computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location.",
      "SLAM is essential for autonomous robots operating in unknown environments.",
      "Common SLAM approaches include EKF SLAM, FastSLAM, and graph-based SLAM."
    ],
    'humanoid': [
      "Humanoid robots are robots that resemble the human body structure, typically having a head, torso, two arms, and two legs.",
      "Humanoid robots present unique challenges in balance, locomotion, and control compared to wheeled robots.",
      "Examples include ASIMO, Pepper, and Atlas robots developed by various companies."
    ],
    'ai': [
      "Artificial Intelligence in robotics involves perception, planning, control, and learning capabilities.",
      "Machine learning and deep learning are increasingly used in robotics for tasks like object recognition and motion planning.",
      "AI enables robots to adapt to new situations and improve their performance over time."
    ],
    'default': [
      "I'm here to help you learn about Physical AI and Humanoid Robotics. Try asking about ROS, navigation, SLAM, or humanoid robots.",
      "Feel free to ask specific questions about the textbook content. I can help explain concepts related to robotics!",
      "The Physical AI & Humanoid Robotics textbook covers four main modules: ROS 2, Digital Twin, AI-Robot Brain, and VLA."
    ]
  };

  // Function to find relevant response based on keywords
  const findResponse = (userInput: string) => {
    const input = userInput.toLowerCase();

    // Check for specific keywords
    for (const [keyword, responses] of Object.entries(knowledgeBase)) {
      if (keyword !== 'default' && input.includes(keyword)) {
        return responses[Math.floor(Math.random() * responses.length)];
      }
    }

    // If no specific keyword found, return a default response
    const defaultResponses = knowledgeBase.default;
    return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
  };

  const handleSend = () => {
    if (inputValue.trim() === '' || isLoading) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Simulate bot thinking time
    setTimeout(() => {
      const botResponse = findResponse(inputValue);
      const botMessage = {
        id: Date.now() + 1,
        text: botResponse,
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
      setIsLoading(false);
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Scroll to bottom of messages
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
              <div
                key={message.id}
                className={`message ${message.sender}`}
              >
                <div className="message-content">
                  {message.text}
                </div>
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