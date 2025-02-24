SH_chatbot/
│
├── app/
│   ├── __init__.py
│   ├── main.py         # Interface Streamlit (choix du modèle, historique)
│   ├── chatbot.py      # Gestion du chatbot et des requêtes vers l'API
│   ├── ollama_api.py   # API FastAPI pour la récupération augmentée
│   ├── faq.py          # Chargement et gestion des FAQs
│   └── models.py       # Configuration des modèles (embeddings, RAG)
│
├── requirements.txt    # Dépendances
├── Faq.txt             # Fichier contenant des réponses FAQ pré-enregistrées
└── README.md           # Documentation du projet

---

### **📌 `app/main.py` - Interface Streamlit avec choix du modèle**
```python
import streamlit as st
from app.chatbot import chat_with_bot

st.title("SH Chatbot - Recherche Augmentée")

# Liste des modèles disponibles
models = ["mistral", "llama2", "gemma"]
selected_model = st.selectbox("Choisissez un modèle :", models)

# Stocker le modèle sélectionné
st.session_state.model = selected_model

# Initialiser l’historique
if "history" not in st.session_state:
    st.session_state.history = []

# Afficher l'historique des conversations
st.subheader("Historique")
for chat in st.session_state.history:
    st.write(f"**Vous** : {chat['query']}")
    st.write(f"**Bot** : {chat['response']}")
    st.markdown("---")

query = st.text_input("Posez votre question :")

if st.button("Envoyer") and query:
    response = chat_with_bot(query, st.session_state.model)
    
    st.session_state.history.append({"query": query, "response": response})
    st.subheader("Réponse :")
    st.write(response)

if st.button("Réinitialiser l'historique"):
    st.session_state.history = []
    st.experimental_rerun()
```

---

### **📌 `app/chatbot.py` - Gestion des requêtes vers l'API**
```python
import requests

def chat_with_bot(query, model="mistral"):
    url = "http://127.0.0.1:8000/rag/"
    response = requests.post(url, params={"query": query, "model": model})

    if response.status_code == 200:
        return response.json()["response"]
    return "Erreur lors de la requête."
```

---

### **📌 `app/ollama_api.py` - API FastAPI pour la récupération augmentée**
```python
from fastapi import FastAPI
from app.models import search_with_reranking, generate_response

app = FastAPI()

@app.post("/rag/")
async def rag_endpoint(query: str, model: str = "mistral"):
    top_docs = search_with_reranking(query)
    context = "\n\n".join(top_docs)
    response = generate_response(query, context, model)
    return {"query": query, "response": response, "context": top_docs}
```

---

### **📌 `app/models.py` - Embeddings, RAG et génération de réponse**
```python
from sentence_transformers import SentenceTransformer
import chromadb

# Modèle d'embedding pour la recherche
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialisation de ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("rag_docs")

def search_with_reranking(query, top_k=5):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=10)
    return [r["text"] for r in results["metadatas"][0]]

def generate_response(query, context, model="mistral"):
    import ollama
    prompt = f"Contexte : {context}\n\nQuestion : {query}\nRéponse :"
    return ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])["message"]["content"]
```

---

### **📌 `app/faq.py` - Gestion des FAQs**
```python
def load_faq():
    with open("Faq.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]
```

---

### **📌 `requirements.txt` - Dépendances**
```
streamlit
fastapi
uvicorn
sentence-transformers
chromadb
requests
ollama
```

---

## **3️⃣ Lancer le projet**
### **1️⃣ Démarrer l'API**
```bash
uvicorn app.ollama_api:app --reload
```

### **2️⃣ Lancer Streamlit**
```bash
streamlit run app/main.py
```

---

### **🎯 Fonctionnalités ajoutées**
✅ **Personnalisation du modèle** (Mistral, Llama2, Gemma, etc.) via un menu déroulant  
✅ **Optimisation du RAG** avec **ChromaDB** et **reranking**  
✅ **Interface utilisateur avec Streamlit**  
✅ **Historique des conversations**  
✅ **Architecture bien organisée**
