# RAG Chatbot Backend for Physical AI & Humanoid Robotics Textbook

This backend provides a Retrieval-Augmented Generation (RAG) chatbot API for the Physical AI & Humanoid Robotics textbook. It allows users to select text from the textbook and ask questions about the selected content.

## Features

- Text selection storage in Qdrant vector database
- Semantic search for relevant content
- Context-aware question answering
- RESTful API endpoints

## API Endpoints

- `GET /` - Root endpoint
- `POST /api/text-selection/` - Store selected text in vector store
- `POST /api/query/` - Query the chatbot with selected text context
- `GET /api/health` - Health check endpoint

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Make sure Qdrant is running (default: http://localhost:6333)
3. Run the server: `python main.py`

## Dependencies

- FastAPI
- Qdrant Client
- Sentence Transformers