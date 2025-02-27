import streamlit as st
import requests
import os
import json
from app.models import insert_text_file_into_chroma
from app.chatbot import chat_with_bot
import logging
import time

# Désactiver les logs de débogage d'asyncio
logging.getLogger('asyncio').setLevel(logging.WARNING)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Définir la configuration de la page en premier
st.set_page_config(page_title="SH Chatbot - Recherche Augmentée", layout="wide", initial_sidebar_state="auto")

# CSS personnalisé pour un chat classique, l’historique et le mode sombre avec défilement
st.markdown("""
    <style>
    .chat-message {
        margin: 5px 0;
        padding: 10px;
        border-radius: 8px;
        max-width: 80%;
    }
    .user-message {
        background-color: #DCF8C6;
        align-self: flex-end;
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background-color: #E8E8E8;
        align-self: flex-start;
        margin-right: auto;
        text-align: left;
    }
    .chat-container {
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 5px;
        background-color: #f5f5f5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 400px; /* Hauteur fixe pour la conversation */
        overflow-y: auto; /* Défilement vertical dans la conversation */
    }
    .typing-indicator {
        font-style: italic;
        color: #888;
        margin: 5px 0;
    }
    .dark-mode .chat-message.user-message {
        background-color: #4CAF50;
        color: white;
    }
    .dark-mode .chat-message.bot-message {
        background-color: #555;
        color: white;
    }
    .dark-mode .chat-container {
        background-color: #333;
        border-color: #555;
    }
    .dark-mode {
        background-color: #222;
        color: white;
    }
    /* Limiter la hauteur de l’application avec défilement global si nécessaire */
    [data-testid="stApp"] {
        max-height: 800px !important; /* Hauteur maximale de l’application */
        overflow-y: auto !important; /* Défilement global si contenu dépasse */
    }
    /* Limiter la hauteur des colonnes avec défilement local si nécessaire */
    [data-testid="column"] {
        max-height: 100% !important; /* Hauteur maximale égale à la hauteur disponible */
        overflow-y: auto !important; /* Défilement local si contenu dépasse */
    }
    /* Style pour les barres de défilement */
    .chat-container::-webkit-scrollbar, [data-testid="stApp"]::-webkit-scrollbar, [data-testid="column"]::-webkit-scrollbar {
        width: 8px;
    }
    .chat-container::-webkit-scrollbar-track, [data-testid="stApp"]::-webkit-scrollbar-track, [data-testid="column"]::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    .chat-container::-webkit-scrollbar-thumb, [data-testid="stApp"]::-webkit-scrollbar-thumb, [data-testid="column"]::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    .chat-container::-webkit-scrollbar-thumb:hover, [data-testid="stApp"]::-webkit-scrollbar-thumb:hover, [data-testid="column"]::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    /* Style pour la zone de saisie */
    .stTextInput > div > input {
        border-radius: 8px;
        padding: 8px;
    }
    /* Style pour le bouton Envoyer */
    .stButton > button {
        background-color: #ff4444; /* Rouge vif */
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #cc0000; /* Rouge plus foncé au survol */
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 SH Chatbot - Recherche Augmentée")
st.markdown("Posez vos questions et enrichissez la base de connaissances avec vos propres fichiers !")

if "history" not in st.session_state:
    st.session_state.history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "last_query" not in st.session_state:
    st.session_state.last_query = None  # Initialisation de last_query
if "model" not in st.session_state:
    st.session_state.model = "qwen2:1.5b"

col1, col2, col3 = st.columns([1, 3, 1])

# Appliquer le mode sombre si activé
if st.session_state.dark_mode:
    st.markdown('<div class="dark-mode">', unsafe_allow_html=True)

with col1:
    st.subheader("📜 Historique détaillé")
    # Conteneur pour l'historique avec barre de défilement
    history_container = st.container()
    with history_container:
        st.markdown("""
            <div style="height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
        """, unsafe_allow_html=True)
        if not st.session_state.history:
            st.info("Aucune conversation pour le moment.")
        else:
            # Afficher les 10 derniers messages pour limiter la longueur
            for i, chat in enumerate(st.session_state.history[-10:]):
                with st.expander(f"{chat['query'][:30]}...", expanded=False):
                    st.markdown(f"**Vous** : {chat['query']}")
                    st.markdown(f"**Bot** : {chat['response']}")
                    if "context" in chat:
                        st.markdown("**Contexte** :")
                        for doc in chat["context"]:
                            st.write(f"- {doc['text'][:50]}... ({doc['distance']:.2f})")
                    if st.button("🗑️ Supprimer", key=f"del_history_{i}"):
                        st.session_state.history.pop(-10 + i)  # Supprimer le message correspondant
                        st.rerun()
                st.markdown("---")
        st.markdown('</div>', unsafe_allow_html=True)
    # Bouton pour exporter l'historique
    if st.session_state.history and st.button("📥 Exporter l'historique"):
        st.download_button(
            label="Télécharger",
            data=json.dumps(st.session_state.history, ensure_ascii=False),
            file_name="chat_history.json",
            mime="application/json"
        )

with col2:
    st.subheader("💬 Conversation")
    # Conteneur pour la conversation avec barre de défilement
    chat_container = st.container()
    with chat_container:
        st.markdown("""
            <div class="chat-container" style="height: 400px; overflow-y: auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        # Afficher les 10 derniers messages pour limiter la longueur
        for chat in st.session_state.history[-10:]:  # Limiter à 10 derniers messages
            st.markdown(f'<div class="chat-message user-message">👤 {chat["query"][:200]}...</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="chat-message bot-message">🤖 {chat["response"]}<br><details><summary>Contexte</summary>{"<br>".join([f"- {doc["text"][:50]}... (Score: {doc["distance"]:.2f})" for doc in chat["context"]]) if "context" in chat else "Aucun contexte"}</details></div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Formulaire pour la saisie avec "Entrée" et vérification anti-redondance
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("Entrez votre question :", placeholder="Ex. : Quelles sont les dates importantes ?")
        submit_button = st.form_submit_button("🚀 Envoyer", type="primary", help="Envoyer votre question")

    if (submit_button or query.strip()) and query and query != st.session_state.last_query:
        st.session_state.last_query = query  # Marquer la dernière question soumise
        with st.spinner("Réponse en cours de génération..."):
            try:
                # Ajouter un indicateur de saisie temporaire
                st.session_state.history.append({"query": query, "response": "Le bot est en train de taper...", "context": []})
                # Simuler un délai pour la réponse (ajustez selon votre besoin)
                time.sleep(1)  # Simuler le temps de réponse
                result = chat_with_bot(query, st.session_state.model)
                response = result.get("response", "Erreur : aucune réponse reçue")
                context = result.get("context", [])
                # Remplacer le dernier message par la vraie réponse
                st.session_state.history[-1] = {"query": query, "response": response, "context": context}
                st.rerun()  # Rafraîchir l'affichage final une seule fois
            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse : {str(e)}")

with col3:
    st.subheader("⚙️ Paramètres")
    models = ["qwen2.5:3b", "llama3.2:latest", "deepseek-r1:1.5b", "qwen2:1.5b"]
    selected_model = st.selectbox("Modèle :", models, help="Choisissez le modèle de langage.")
    if selected_model != st.session_state.get("model", "qwen2.5:3b"):
        st.session_state.model = selected_model
        st.rerun()  # Rafraîchir uniquement si le modèle change

    dark_mode = st.checkbox("Mode sombre", value=st.session_state.get("dark_mode", False))
    if dark_mode != st.session_state.get("dark_mode", False):
        st.session_state.dark_mode = dark_mode
        st.rerun()  # Rafraîchir uniquement si le mode sombre change

    if st.button("🗑️ Réinitialiser"):
        st.session_state.history = []
        st.success("Historique réinitialisé !")
        st.rerun()

    st.markdown("---")
    st.subheader("📂 Importer un fichier")
    uploaded_file = st.file_uploader("Ajoutez un fichier (.txt, .docx, .csv, .pdf, .json)", type=["txt", "docx", "csv", "pdf", "json"])
    if uploaded_file is not None:
        file_path = f"temp_{uploaded_file.name}"
        with st.spinner("Validation et indexation du fichier..."):
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                logger.debug("Fichier écrit sur disque")
                result = insert_text_file_into_chroma(file_path)
                logger.debug("Indexation terminée")
                if result["success"]:
                    st.success(f"{result['message']} ({result['chunks_indexed']} chunks indexés sur {result['chunks_created']} créés)")
                else:
                    st.error(result["message"])
            except RuntimeError as re:
                st.error(f"Erreur d'exécution : {str(re)}")
            except Exception as e:
                st.error(f"Erreur inattendue : {str(e)}")
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

# Fermer la balise div du mode sombre si activé
if st.session_state.dark_mode:
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("RAG réalisé par H. Sami • Date : 23 février 2025")