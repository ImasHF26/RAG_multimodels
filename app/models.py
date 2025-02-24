import chromadb
import uuid
import os
import hashlib
import ollama
from typing import List, Dict

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("rag_docs")

def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as file:
        while chunk := file.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
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
    return chunks

def insert_text_file_into_chroma(file_path: str, chunk_size: int = 500) -> dict:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    if not os.path.exists(file_path):
        return {"success": False, "message": f"Le fichier {file_path} n'existe pas."}
    file_hash = compute_file_hash(file_path)
    existing_docs = collection.get(where={"file_hash": file_hash}, include=["metadatas"])
    if existing_docs["metadatas"]:
        return {"success": False, "message": f"Le fichier {file_path} est déjà présent dans ChromaDB."}
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            text = file.read()
    except UnicodeDecodeError:
        return {"success": False, "message": f"Erreur d'encodage pour {file_path}. Assurez-vous que le fichier est en UTF-8."}
    chunks = chunk_text(text, max_chunk_size=chunk_size)
    doc_ids, documents, embeddings, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        chunk_embedding = embedding_model.encode(chunk, batch_size=32).tolist()
        doc_id = str(uuid.uuid4())
        doc_ids.append(doc_id)
        documents.append(chunk)
        embeddings.append(chunk_embedding)
        metadatas.append({
            "source": file_path,
            "file_hash": file_hash,
            "chunk_index": i,
            "total_chunks": len(chunks)
        })
    collection.add(ids=doc_ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return {
        "success": True,
        "message": f"Le fichier {file_path} a été ajouté avec succès.",
        "chunks_created": len(chunks)
    }

def search_with_reranking(query: str, top_k: int = 3) -> List[Dict]:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k * 2)
    if "documents" not in results or not results["documents"]:
        print("Aucun résultat trouvé.")
        return []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    ranked_results = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(docs, metas, distances)
    ]
    ranked_results = sorted(ranked_results, key=lambda x: x["distance"])[:top_k]
    return ranked_results

def generate_response(query: str, context: List[Dict], model: str = "qwen2:1.5b") -> str:
    if not isinstance(context, list):
        raise ValueError(f"Context doit être une liste, reçu : {type(context)}")
    context_text = "\n".join([item["text"] for item in context if "text" in item])
    prompt = f"Contexte : {context_text}\n\nQuestion : {query}\nInstruction : Répondez uniquement en vous basant sur le contexte fourni. \n assurez-vous toujours de fournir des sources fiables, indiquez leur source.\n Ne rajoutez aucune information externe. Répondez toujours en français.\nRéponse :"
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])["message"]["content"]
    return response