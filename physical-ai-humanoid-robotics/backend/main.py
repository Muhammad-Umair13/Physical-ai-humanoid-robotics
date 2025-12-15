# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware

WEBHOOK_URL = "https://muhamadumair.app.n8n.cloud/webhook/chatbot"

app = FastAPI()

# Allow CORS so frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    prompt: str

@app.post("/ask")
def ask_question(q: Question):
    try:
        response = requests.post(
            WEBHOOK_URL,
            json={"prompt": q.prompt},
            timeout=60
        )
        response.raise_for_status()
        
        # Parse n8n response
        try:
            data = response.json()
            if isinstance(data, list):
                first_item = data[0].get("json", {})
                answer = first_item.get("answer", "")
            else:
                answer = data.get("answer", "")
        except ValueError:
            answer = response.text

        if not answer:
            return {"answer": "No response from n8n."}
        return {"answer": answer}

    except requests.exceptions.RequestException as e:
        return {"answer": f"Error contacting n8n: {e}"}
