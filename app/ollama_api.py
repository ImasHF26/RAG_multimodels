from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
from app.models import search_with_reranking_and_filters, generate_response, insert_text_file_into_chroma

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    model: str
    filters: dict = {}

class ChatResponse(BaseModel):
    response: str
    context: list = []

@app.post("/rag")
async def rag_endpoint(request: ChatRequest):
    logger.debug(f"Requête reçue: {request.dict()}")
    
    try:
        # Extraire les filtres
        filters = request.filters or {}
        
        # Log des filtres appliqués
        if filters:
            logger.info(f"Filtres appliqués: {filters}")
        
        # Recherche avec filtres
        top_docs = search_with_reranking_and_filters(
            request.query,
            filters=filters
        )
        
        # Génération de la réponse
        response = generate_response(request.query, top_docs, request.model)
        
        return {
            "query": request.query, 
            "response": response, 
            "context": top_docs
        }
    except Exception as e:
        logger.error(f"Erreur dans l'endpoint RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de la requête: {str(e)}")

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    domaine: str = Form(None, description="Domaine du document (ex: science, informatique)"),
    theme: str = Form(None, description="Thème du document (ex: Cours, Tuto, TD, TP)")
):
    """Endpoint pour importer divers types de fichiers et les indexer dans ChromaDB."""
    supported_extensions = ["txt", "docx", "csv", "pdf", "json"]
    file_extension = file.filename.split('.')[-1].lower()
    
    if file_extension not in supported_extensions:
        raise HTTPException(status_code=400, detail=f"Seuls les fichiers {', '.join(supported_extensions)} sont supportés.")
    
    file_path = f"temp_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        result = insert_text_file_into_chroma(file_path, domaine=domaine, theme=theme, filename=file.filename)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        return {
            "message": result["message"],
            "chunks_created": result["chunks_created"],
            "chunks_indexed": result["chunks_indexed"],
            "domaine": domaine,
            "theme": theme
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'importation du fichier: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'importation : {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/health")
async def health_check():
    """Endpoint de vérification de l'état du serveur."""
    return {"status": "ok", "message": "Le serveur API est opérationnel."}

@app.get("/welcome")
async def welcome():
    return {"message": "Hello! Type 'exit' to quit."}