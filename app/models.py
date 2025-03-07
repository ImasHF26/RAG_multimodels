import chromadb
import uuid
import os
import hashlib
import ollama
from typing import List, Dict
import pandas as pd
from docx import Document
import PyPDF2
import json
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialisation de ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("rag_docs")

# Chargement du modèle d'embedding
def load_embedding_model():
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.debug("Embedding model loaded successfully")
        return embedding_model
    except Exception as e:
        logger.error(f"Erreur lors du chargement de SentenceTransformer : {str(e)}")
        raise ValueError(f"Erreur lors du chargement du modèle d'embedding : {str(e)}")

# Calcul du hash du fichier
def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as file:
        while chunk := file.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Validation du fichier
def validate_file(file_path: str) -> tuple[bool, str]:
    if not os.path.exists(file_path):
        return False, f"Le fichier {file_path} n'existe pas."
    if not os.path.isfile(file_path):
        return False, f"{file_path} n'est pas un fichier valide."
    if os.path.getsize(file_path) == 0:
        return False, f"Le fichier {file_path} est vide."
    return True, "Fichier valide."

# Extraction de texte à partir d'un fichier
def extract_text_from_file(file_path: str) -> str:
    extension = file_path.split('.')[-1].lower()
    logger.debug(f"Extracting text from {file_path} (extension: {extension})")
    try:
        if extension == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif extension == "docx":
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        elif extension == "pdf":
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if isinstance(extracted, str):
                        text += extracted + "\n"
                return text.strip() or ""
        elif extension == "csv":
            df = pd.read_csv(file_path)
            return df.to_string()
        elif extension == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return json.dumps(data, ensure_ascii=False)
        else:
            raise ValueError(f"Type de fichier non supporté : {extension}")
    except Exception as e:
        logger.error(f"Erreur dans extract_text_from_file pour {file_path} : {str(e)}")
        raise ValueError(f"Erreur lors de l'extraction du texte : {str(e)}")

# Découpage du texte en chunks
def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    logger.debug(f"Chunking text (length: {len(text)})")
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        word_length = len(word) + 1
        if current_length + word_length <= max_chunk_size:
            current_chunk.append(word)
            current_length += word_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    logger.debug(f"Created {len(chunks)} chunks")
    return chunks

# Vérification des chevauchements
def check_overlap(new_chunk: str, existing_embeddings: List[List[float]], threshold: float = 0.9) -> bool:
    model = load_embedding_model()
    try:
        new_embedding = model.encode(new_chunk, convert_to_numpy=True)
        for existing_emb in existing_embeddings:
            similarity = cosine_similarity(new_embedding.reshape(1, -1), np.array(existing_emb).reshape(1, -1))[0][0]
            if similarity >= threshold:
                logger.debug(f"Overlap detected: similarity = {similarity}")
                return True
        return False
    except Exception as e:
        logger.error(f"Erreur dans check_overlap : {str(e)}")
        raise ValueError(f"Erreur dans check_overlap : {str(e)}")

# Insertion d'un fichier texte dans ChromaDB
# Dans app/models.py
def insert_text_file_into_chroma(file_path: str, chunk_size: int = 500, domaine: str = None, theme: str = None, filename: str = None, size: float = None, date: str = None) -> dict:
    is_valid, validation_message = validate_file(file_path)
    if not is_valid:
        return {"success": False, "message": validation_message}
    
    file_hash = compute_file_hash(file_path)
    existing_docs = collection.get(where={"file_hash": file_hash}, include=["metadatas"])
    if existing_docs["metadatas"]:
        return {"success": False, "message": f"Le fichier {file_path} est déjà présent dans ChromaDB."}
    
    try:
        text = extract_text_from_file(file_path)
        if not isinstance(text, str) or not text.strip():
            return {"success": False, "message": f"Le fichier {file_path} ne contient aucun texte exploitable."}
    except ValueError as e:
        return {"success": False, "message": str(e)}
    
    chunks = chunk_text(text, max_chunk_size=chunk_size)
    if not chunks:
        return {"success": False, "message": f"Aucun chunk créé à partir du fichier {file_path}."}
    
    doc_ids, documents, embeddings, metadatas = [], [], [], []
    all_existing = collection.get(include=["embeddings"])
    existing_embeddings = all_existing["embeddings"] if all_existing["embeddings"] is not None else []
    
    model = load_embedding_model()
    for i, chunk in enumerate(chunks):
        if existing_embeddings is not None and len(existing_embeddings) > 0 and check_overlap(chunk, existing_embeddings):
            logger.debug(f"Skipping chunk {i} due to overlap")
            continue
        
        try:
            chunk_embedding = model.encode(chunk, batch_size=32, convert_to_numpy=True)
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            documents.append(chunk)
            embeddings.append(chunk_embedding.tolist())
            metadata = {
                "source": file_path,
                "file_hash": file_hash,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            if domaine:
                metadata["domaine"] = domaine
            if theme:
                metadata["theme"] = theme
            if filename:
                metadata["filename"] = filename
            if size is not None:  # Ajout de la taille
                metadata["size"] = size
            if date:  # Ajout de la date
                metadata["date"] = date
            metadatas.append(metadata)
        except Exception as e:
            logger.error(f"Erreur lors de l'encodage du chunk {i}: {str(e)}")
            continue
    
    if not doc_ids:
        return {"success": False, "message": "Aucun nouveau contenu ajouté (chevauchement détecté ou erreur)."}
    
    try:
        collection.add(ids=doc_ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        indexed_docs = collection.get(ids=doc_ids, include=["documents"])
        if not indexed_docs["documents"] or len(indexed_docs["documents"]) != len(doc_ids):
            return {"success": False, "message": f"Échec de l'indexation pour {file_path}."}
        
        return {
            "success": True,
            "message": f"Le fichier {file_path} a été indexé avec succès.",
            "chunks_created": len(chunks),
            "chunks_indexed": len(doc_ids),
            "domaine": domaine,
            "theme": theme,
            "filename": filename
        }
    except Exception as e:
        logger.error(f"Erreur dans insert_text_file_into_chroma : {str(e)}")
        return {"success": False, "message": f"Erreur lors de l'indexation : {str(e)}"}

# Recherche avec reranking et filtres
def search_with_reranking_and_filters(query: str, top_k: int = 3, filters: dict = None) -> List[Dict]:
    model = load_embedding_model()
    try:
        query_embedding = model.encode(query, convert_to_numpy=True).tolist()
    except Exception as e:
        logger.error(f"Erreur lors de l'encodage de la requête : {str(e)}")
        raise ValueError(f"Erreur lors de l'encodage de la requête : {str(e)}")
    
    # Construire la clause where pour les filtres
    where_clause = {}
    if filters:
        for key, value in filters.items():
            if value:  # Ignorer les valeurs vides
                where_clause[key] = value
    
    # Effectuer la recherche avec les filtres si présents
    if where_clause:
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=top_k * 2,
            where=where_clause
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=top_k * 2
        )
    
    if not results.get("documents"):
        logger.info("Aucun résultat trouvé pour la requête.")
        return []
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    
    ranked_results = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(docs, metas, distances)
    ]
    
    return sorted(ranked_results, key=lambda x: x["distance"])[:top_k]

# Recherche simple (pour compatibilité avec l'ancien code)
def search_with_reranking(query: str, top_k: int = 3) -> List[Dict]:
    return search_with_reranking_and_filters(query, top_k)

# Génération de réponse
def generate_response(query: str, context: List[Dict], model: str = "qwen2.5:3b") -> str:
    if not isinstance(context, list):
        raise ValueError(f"Context doit être une liste, reçu : {type(context)}")
    context_text = "\n".join([
        f"{item['text']} (Source: {item['metadata']['source']}, Domaine: {item['metadata'].get('domaine', 'N/A')}, Thème: {item['metadata'].get('theme', 'N/A')}, Fichier: {item['metadata'].get('filename', 'N/A')})"
        for item in context if "text" in item and "metadata" in item
    ])
    prompt = f"Contexte : {context_text}\n\nQuestion : {query}\nInstruction : Répondez uniquement en vous basant sur le contexte fourni. Indiquez la source si possible. Ne rajoutez aucune information externe. Répondez toujours en français.\nRéponse :"
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])["message"]["content"]
    return response