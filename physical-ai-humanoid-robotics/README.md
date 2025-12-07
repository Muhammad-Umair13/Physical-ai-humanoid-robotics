# Physical AI & Humanoid Robotics Textbook with Integrated RAG Chatbot

This project provides a comprehensive textbook on Physical AI and Humanoid Robotics, augmented with an integrated RAG (Retrieval-Augmented Generation) chatbot. The textbook is structured into four modules: ROS 2, Digital Twin (Gazebo/Unity), AI-Robot Brain (NVIDIA Isaac, VSLAM, navigation, locomotion), and VLA (Voice commands, LLM planning, capstone autonomous humanoid).

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Modules](#modules)
- [RAG Chatbot](#rag-chatbot)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Contributing](#contributing)

## Features

- Comprehensive textbook on Physical AI and Humanoid Robotics
- Four detailed modules covering ROS 2, Digital Twin, AI-Robot Brain, and VLA
- Interactive RAG chatbot for Q&A based on selected textbook content
- Reproducible labs and exercises for hands-on learning
- Capstone project demonstrating autonomous humanoid capabilities
- Deployable on GitHub Pages

## Project Structure

```
physical-ai-humanoid-robotics/
├── docs/
│   ├── module1-ros2/           # ROS 2: Nodes, Topics, Services, Python agents, URDF
│   │   ├── 01-introduction-to-ros2.md
│   │   ├── 02-ros2-architecture.md
│   │   ├── 03-nodes-topics-services-actions.md
│   │   ├── 04-launch-files-parameters.md
│   │   └── labs/
│   │       └── lab1-ros2-node-sensor-integration.md
│   ├── module2-digital-twin/   # Digital Twin: Gazebo simulation, sensors, Unity visualization
│   ├── module3-ai-robot-brain/ # AI-Robot Brain: NVIDIA Isaac, VSLAM, navigation, locomotion
│   └── module4-vla/            # VLA: Voice commands (Whisper), LLM planning, capstone
├── static/
│   ├── images/
│   └── diagrams/
├── src/
│   └── components/
│       └── RagChatbot.tsx      # Integrated RAG chatbot component
├── backend/                    # FastAPI backend for RAG functionality
│   ├── main.py                 # Main API server
│   ├── qdrant_store.py         # Qdrant vector store implementation
│   ├── requirements.txt        # Backend dependencies
│   └── README.md               # Backend documentation
├── sidebars.js
├── docusaurus.config.ts
└── package.json
```

## Modules

### Module 1: ROS 2
- Introduction to ROS 2 concepts
- Architecture and communication patterns
- Nodes, topics, services, and actions
- Launch files and parameters
- Practical labs for sensor integration

### Module 2: Digital Twin
- Gazebo simulation fundamentals
- Sensor integration in simulation
- Unity visualization setup
- Digital twin architecture

### Module 3: AI-Robot Brain
- NVIDIA Isaac SDK fundamentals
- VSLAM concepts and implementation
- Navigation algorithms
- Locomotion and perception systems

### Module 4: VLA (Vision-Language-Action)
- Voice command processing with Whisper
- LLM planning systems
- Integrated VLA system
- Capstone autonomous humanoid project

## RAG Chatbot

The integrated RAG (Retrieval-Augmented Generation) chatbot provides interactive Q&A based on selected textbook content:

- **Backend**: FastAPI server with Qdrant vector store
- **Frontend**: React component integrated into textbook pages
- **Functionality**: Answers questions based only on selected text context
- **Technology**: Uses embeddings to find relevant content and generate responses

### Backend Features
- Text selection storage in Qdrant vector database
- Semantic search for relevant content
- Context-aware question answering
- RESTful API endpoints

### Frontend Features
- Floating AI assistant button
- Text selection detection
- Conversation history
- Real-time responses

## Installation

### Prerequisites
- Node.js (v16 or higher)
- Python (v3.8 or higher)
- Qdrant vector database

### Setup

1. Install Docusaurus dependencies:
   ```bash
   npm install
   ```

2. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Set up Qdrant (for RAG functionality):
   - Install Qdrant locally or use Docker:
   ```bash
   docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
   ```

## Usage

### Running the Textbook

1. Start the Docusaurus development server:
   ```bash
   npm start
   ```

2. Open your browser to `http://localhost:3000`

### Running the Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Start the FastAPI server:
   ```bash
   python main.py
   ```

3. The API will be available at `http://localhost:8000`

### Using the RAG Chatbot

1. Select text in any textbook chapter
2. Click the floating AI Assistant button
3. Ask questions about the selected text
4. The chatbot will respond based on the selected content

## Development

### Adding New Content

1. Create new markdown files in the appropriate module directory
2. Update `sidebars.js` to include the new content in the navigation
3. Ensure proper frontmatter is included in markdown files

### Backend Development

The backend is built with FastAPI and uses Qdrant for vector storage:

1. API endpoints are defined in `backend/main.py`
2. Qdrant integration is in `backend/qdrant_store.py`
3. Update dependencies in `backend/requirements.txt` as needed

### Frontend Development

The frontend uses React components integrated with Docusaurus:

1. The RAG chatbot component is in `src/components/RagChatbot.tsx`
2. Styles are in `src/components/RagChatbot.css`
3. Components are integrated through Docusaurus' component system

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
