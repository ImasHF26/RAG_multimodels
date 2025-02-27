import requests
from app.models import insert_text_file_into_chroma

API_URL = "http://127.0.0.1:8000/rag"

def chat_with_bot(query: str, model: str) -> dict:
    payload = {"query": query, "model": model}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"response": f"Erreur lors de la connexion à l'API : {str(e)}", "context": []}