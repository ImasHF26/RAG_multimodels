from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Importation relative
from .models import search_with_reranking, generate_response, insert_text_file_into_chroma

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger un fichier au démarrage (optionnel)
# insert_text_file_into_chroma("Faq.txt")

class ChatRequest(BaseModel):
    query: str
    model: str

class ChatResponse(BaseModel):
    response: str

@app.post("/rag")
async def rag_endpoint(request: ChatRequest):
    top_docs = search_with_reranking(request.query)
    response = generate_response(request.query, top_docs, request.model)
    return {"query": request.query, "response": response, "context": top_docs}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint pour importer divers types de fichiers et les indexer dans ChromaDB."""
    supported_extensions = ["txt", "docx", "csv", "pdf", "json"]
    file_extension = file.filename.split('.')[-1].lower()
    
    if file_extension not in supported_extensions:
        raise HTTPException(status_code=400, detail=f"Seuls les fichiers {', '.join(supported_extensions)} sont supportés.")
    
    file_path = f"temp_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        result = insert_text_file_into_chroma(file_path)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        return {"message": result["message"], "chunks_created": result["chunks_created"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'importation : {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/welcome")
async def welcome():
    return {"message": "Hello! Type 'exit' to quit."}