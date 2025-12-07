from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from qdrant_store import QdrantRAGStore

app = FastAPI(
    title="RAG Chatbot API for Physical AI & Humanoid Robotics Textbook",
    description="API for handling text selection and Q&A based on selected textbook content",
    version="1.0.0"
)

# Models for request/response
class TextSelectionRequest(BaseModel):
    selected_text: str
    query: str

class ChatbotResponse(BaseModel):
    response: str
    context_used: str

class VectorStoreResponse(BaseModel):
    success: bool
    message: str

# Initialize the Qdrant store
rag_store = QdrantRAGStore()

@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API for Physical AI & Humanoid Robotics Textbook"}

@app.post("/api/text-selection/", response_model=VectorStoreResponse)
def store_selected_text(request: TextSelectionRequest):
    """
    Store selected text in the vector store for later retrieval
    """
    try:
        # Store the selected text in Qdrant
        text_id = rag_store.store_text(
            request.selected_text,
            metadata={"query": request.query, "source": "textbook_selection"}
        )
        return VectorStoreResponse(
            success=True,
            message=f"Text stored successfully with ID: {text_id}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/", response_model=ChatbotResponse)
def query_chatbot(request: TextSelectionRequest):
    """
    Query the chatbot with selected text as context
    """
    try:
        # First, store the selected text if it's new
        text_id = rag_store.store_text(
            request.selected_text,
            metadata={"query": request.query, "source": "current_selection"}
        )

        # Retrieve similar content from the store based on the query
        # This simulates the RAG process: retrieve relevant context, then generate response
        results = rag_store.retrieve_similar(request.query)

        if results:
            # Use the most relevant text as context
            context_text = results[0]["text"]
            response = f"Based on the textbook content: '{context_text[:200]}...', I can answer your query: '{request.query}'. The system has found relevant information to address your question about this Physical AI and Humanoid Robotics topic."
        else:
            # If no similar content found, respond accordingly
            response = f"I've stored your selected text: '{request.selected_text[:100]}...'. However, I couldn't find directly relevant content to answer your specific query: '{request.query}'. Please make sure your query relates to the selected text or try rephrasing."

        return ChatbotResponse(
            response=response,
            context_used=request.selected_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "RAG Chatbot API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)