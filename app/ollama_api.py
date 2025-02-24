from fastapi import FastAPI, UploadFile, File
from app.models import search_with_reranking, generate_response, insert_text_file_into_chroma
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exemple d'utilisation pour ajouter un fichier texte au démarrage (optionnel)
insert_text_file_into_chroma("Faq.txt")

class ChatRequest(BaseModel):
    query: str
    model: str

class ChatResponse(BaseModel):
    response: str

@app.post("/rag")
async def rag_endpoint(request: ChatRequest):
    top_docs = search_with_reranking(request.query)
    print('top_docs', top_docs)
    response = generate_response(request.query, top_docs, request.model)
    return {"query": request.query, "response": response, "context": top_docs}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint pour importer un fichier texte et l'indexer dans ChromaDB."""
    if not file.filename.endswith(".txt"):
        return {"error": "Seuls les fichiers .txt sont supportés."}
    
    file_path = f"temp_{file.filename}"
    try:
        # Sauvegarder le fichier temporairement
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Indexer dans ChromaDB
        insert_text_file_into_chroma(file_path)
        return {"message": f"Fichier '{file.filename}' importé et indexé avec succès !"}
    except Exception as e:
        return {"error": f"Erreur lors de l'importation : {str(e)}"}
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/welcome")
async def welcome():
    return {"message": "Hello! Type 'exit' to quit."}