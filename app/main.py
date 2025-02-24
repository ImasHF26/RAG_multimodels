import streamlit as st
import requests
import json
from models import insert_text_file_into_chroma

API_URL = "http://127.0.0.1:8000/rag"

def chat_with_bot(query: str, model: str) -> dict:
    payload = {"query": query, "model": model}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"response": f"Erreur lors de la connexion à l'API : {str(e)}", "context": []}

st.set_page_config(page_title="SH Chatbot - Recherche Augmentée", layout="wide")
st.title("🤖 SH Chatbot - Recherche Augmentée")
st.markdown("Posez vos questions et enrichissez la base de connaissances avec vos propres fichiers !")

if "history" not in st.session_state:
    st.session_state.history = []

col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.subheader("📜 Historique")
    if not st.session_state.history:
        st.info("Aucune conversation pour le moment.")
    else:
        for chat in st.session_state.history:
            with st.expander(f"{chat['query'][:30]}...", expanded=False):
                st.markdown(f"**Vous** : {chat['query']}")
                st.markdown(f"**Bot** : {chat['response']}")
                if "context" in chat:
                    st.markdown("**Contexte** :")
                    for doc in chat["context"]:
                        st.write(f"- {doc['text'][:50]}... ({doc['distance']:.2f})")
            st.markdown("---")

with col2:
    st.subheader("💬 Conversation en cours")
    query = st.text_area("Entrez votre question :", height=150, placeholder="Ex. : Quelles sont les dates importantes ?")
    if st.button("🚀 Envoyer", type="primary") and query.strip():
        with st.spinner("Réponse en cours de génération..."):
            result = chat_with_bot(query, st.session_state.get("model", "qwen2:1.5b"))
            response = result.get("response", "Erreur : aucune réponse reçue")
            context = result.get("context", [])
            st.session_state.history.append({"query": query, "response": response, "context": context})
            st.markdown("### Réponse :")
            st.write(response)
            with st.expander("Voir le contexte utilisé", expanded=False):
                if context:
                    for doc in context:
                        st.write(f"- {doc['text'][:100]}... (Score: {doc['distance']:.2f})")
                else:
                    st.write("Aucun contexte disponible.")
    elif query.strip() == "":
        st.warning("Veuillez entrer une question valide.")

with col3:
    st.subheader("⚙️ Paramètres")
    models = ["qwen2.5:3b", "llama3.2:latest", "deepseek-r1:1.5b", "qwen2:1.5b"]
    selected_model = st.selectbox("Modèle :", models, help="Choisissez le modèle de langage.")
    st.session_state.model = selected_model
    if st.button("🗑️ Réinitialiser"):
        st.session_state.history = []
        st.success("Historique réinitialisé !")
    st.markdown("---")
    st.subheader("📂 Importer un fichier")
    uploaded_file = st.file_uploader("Ajoutez un fichier texte (.txt)", type=["txt"])
    if uploaded_file is not None:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            result = insert_text_file_into_chroma(file_path)
            if result["success"]:
                st.success(f"{result['message']} ({result['chunks_created']} chunks créés)")
            else:
                st.error(result["message"])
        except Exception as e:
            st.error(f"Erreur : {str(e)}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

st.markdown("---")
st.caption("RAG est réalisée par H. Sami  • Date : 23 février 2025")