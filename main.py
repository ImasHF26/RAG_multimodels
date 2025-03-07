import sys
import os
import subprocess
import time
import threading
import logging
from datetime import datetime
from typing import List, Dict, Any

# Gestion sécurisée des importations
try:
    import gradio as gr
except ImportError as e:
    print(f"Erreur: Impossible d'importer gradio: {e}")
    print("Installez gradio avec: pip install gradio")
    sys.exit(1)

try:
    import requests
except ImportError as e:
    print(f"Erreur: Impossible d'importer requests: {e}")
    print("Installez requests avec: pip install requests")
    sys.exit(1)

try:
    from app.models import insert_text_file_into_chroma, collection
    from app.chatbot import chat_with_bot
except ImportError as e:
    print(f"Erreur: Impossible d'importer les modules personnalisés: {e}")
    print("Vérifiez que app/models.py et app/chatbot.py existent et sont corrects.")
    sys.exit(1)

class EnhancedChatbotApplication:
    """
    Classe améliorée de l'application Chatbot avec interface à onglets
    séparant les fonctionnalités d'indexation et de chat.
    """
    
    def __init__(self):
        """Initialise l'application avec la configuration de base et la journalisation."""
        # Configuration du logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # État de l'application
        self.api_running = False
        self.demo = None
        self.chat_history = []
        
        # Initialiser les domaines et thèmes
        self.unique_domains, self.unique_themes = self.get_unique_domains_and_themes()
        
        # Modèles disponibles avec leurs descriptions
        self.available_models = {
            "qwen2.5:3b": "Équilibré - Bon ratio performance/rapidité",
            "llama3.2:latest": "Performant - Réponses détaillées",
            "deepseek-r1:1.5b": "Rapide - Réponses concises",
            "qwen2:1.5b": "Très rapide - Idéal pour des questions simples"
        }
        
    def check_dependencies(self):
        """Vérifie si toutes les dépendances nécessaires sont installées."""
        required_packages = {
            "gradio": "Interface utilisateur",
            "chromadb": "Base de données vectorielle",
            "sentence_transformers": "Modèle d'embedding",
            "ollama": "Interface avec les modèles de langage",
            "fastapi": "API REST",
            "PyPDF2": "Traitement des fichiers PDF",
            "docx": "Traitement des fichiers Word",
            "sklearn": "Calcul de similarité cosinus",
            "numpy": "Opérations matricielles",
            "pandas": "Manipulation de données",
            "requests": "Appels API HTTP",
            "uvicorn": "Serveur ASGI"
        }
        
        missing_packages = []
        for package, description in required_packages.items():
            try:
                __import__(package.split(".")[0])
            except ImportError:
                missing_packages.append(f"{package} ({description})")
        
        if missing_packages:
            self.logger.warning("⚠️ Packages manquants détectés :")
            for pkg in missing_packages:
                self.logger.warning(f"  - {pkg}")
            self.logger.warning("\nVeuillez installer les packages manquants avec la commande :")
            self.logger.warning(f"pip install {' '.join([pkg.split(' ')[0] for pkg in missing_packages])}")
            return False
        return True
    
    def is_server_running(self, port=8000):
        """Vérifie si un serveur est en cours d'exécution sur le port spécifié."""
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def start_fastapi_server(self):
        """Démarre le serveur FastAPI en arrière-plan."""
        try:
            subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "app.ollama_api:app", "--host", "127.0.0.1", 
                "--port", "8000", "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            attempts = 0
            while not self.is_server_running(8000) and attempts < 10:
                time.sleep(1)
                attempts += 1
            if self.is_server_running(8000):
                self.logger.info("✅ Serveur FastAPI démarré avec succès sur le port 8000")
                self.api_running = True
            else:
                self.logger.error("❌ Échec du démarrage du serveur FastAPI")
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du démarrage du serveur FastAPI : {str(e)}")
    
    def get_unique_domains_and_themes(self):
        """Récupère les domaines et thèmes uniques depuis ChromaDB."""
        try:
            all_docs = collection.get(include=["metadatas"])
            domains = set()
            themes = set()
            if all_docs["metadatas"]:
                for metadata in all_docs["metadatas"]:
                    if "domaine" in metadata and metadata["domaine"]:
                        domains.add(metadata["domaine"])
                    if "theme" in metadata and metadata["theme"]:
                        themes.add(metadata["theme"])
            return sorted(list(domains)), sorted(list(themes))
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des domaines et thèmes : {str(e)}")
            return [], []
    
    def chat(self, query: str, selected_domaine: str, selected_theme: str, model: str) -> tuple[str, str, Dict[str, Any]]:
        """Fonction principale du chat avec gestion des filtres de domaine et thème."""
        if not query:
            return "Veuillez entrer une question.", "", {}
        
        try:
            result = chat_with_bot(
                query, 
                model, 
                domaine=selected_domaine if selected_domaine != "Tous" else None,
                theme=selected_theme if selected_theme != "Tous" else None
            )
            response = result.get("response", "Erreur : aucune réponse reçue")
            context = result.get("context", [])
            sources_text = ""
            source_list = []
            if context:
                for i, ctx in enumerate(context, 1):
                    meta = ctx.get("metadata", {})
                    filename = meta.get('filename', 'Inconnu')
                    domaine = meta.get('domaine', 'N/A')
                    theme = meta.get('theme', 'N/A')
                    source_list.append({"filename": filename, "domaine": domaine, "theme": theme})
                    sources_text += f"{i}. {filename} (Domaine: {domaine}, Thème: {theme})\n"
            entry = {"user": query, "bot": response, "sources": source_list, "model": model}
            return response, sources_text, entry
        except Exception as e:
            error_msg = f"Erreur lors du traitement de la requête : {str(e)}"
            self.logger.error(error_msg)
            entry = {"user": query, "bot": error_msg, "sources": [], "model": model}
            return error_msg, "", entry
    
# Plus bas, dans la classe EnhancedChatbotApplication
    def process_import(self, files: List[Any], domaine: str, theme: str, custom_domaine: str, custom_theme: str) -> tuple[str, List]:
        if not files:
            return "Aucun fichier sélectionné.", []
        
        if domaine == "Nouveau" and custom_domaine:
            domaine = custom_domaine
        if theme == "Nouveau" and custom_theme:
            theme = custom_theme
        
        results = []
        for file in files:
            base_filename = os.path.basename(file.name)
            file_path = file.name  # Chemin temporaire fourni par Gradio
            try:
                # Calculer la taille en Ko
                file_size = os.path.getsize(file_path) / 1024  # Convertir octets en Ko
                # Générer la date d'indexation
                index_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                result = insert_text_file_into_chroma(
                    file_path,
                    domaine=domaine,
                    theme=theme,
                    filename=base_filename,
                    size=file_size,  # Passer la taille
                    date=index_date  # Passer la date
                )
                if result["success"]:
                    message = f"✓ {base_filename}: {result['chunks_indexed']} chunks indexés sur {result['chunks_created']} créés (Domaine: {domaine}, Thème: {theme})"
                else:
                    message = f"✗ {base_filename}: {result['message']}"
                results.append(message)
            except Exception as e:
                results.append(f"✗ {base_filename}: Erreur - {str(e)}")
        
        self.unique_domains, self.unique_themes = self.get_unique_domains_and_themes()
        return "\n".join(results), []
    
    def update_dropdowns(self):
        """Met à jour les listes déroulantes avec les domaines et thèmes actuels."""
        domains, themes = self.get_unique_domains_and_themes()
        return gr.update(choices=["Tous"] + domains), gr.update(choices=["Tous"] + themes)
    
    def update_indexing_dropdowns(self):
        """Met à jour les listes déroulantes dans l'onglet d'indexation."""
        domains, themes = self.get_unique_domains_and_themes()
        return gr.update(choices=["Nouveau"] + domains), gr.update(choices=["Nouveau"] + themes)
    
    def handle_chat(self, query: str, selected_domaine: str, selected_theme: str, model: str, chatbot: List) -> tuple[str, List, gr.update, str]:
        """Gère une interaction de chat et met à jour l'interface."""
        if not query.strip():
            return "", chatbot, gr.update(visible=False), ""
        
        response, sources_text, entry = self.chat(query, selected_domaine, selected_theme, model)
        chatbot = chatbot + [{"role": "user", "content": query}, {"role": "assistant", "content": response}]
        self.chat_history.append(entry)
        sources_visible = len(entry["sources"]) > 0
        return "", chatbot, gr.update(visible=sources_visible), sources_text
    
    def reset_chat(self) -> tuple[List, gr.update, str]:
        """Réinitialise le chat et l'historique."""
        self.chat_history = []
        return [], gr.update(visible=False), ""
    
    def build_interface(self):
        """Construit l'interface utilisateur Gradio avec onglets."""
        with gr.Blocks(title="SH Chatbot - RAG", theme="soft") as demo:
            with gr.Row(equal_height=True):
                with gr.Column(scale=6):
                    gr.Markdown("## 🤖 SH Chatbot - Recherche Augmentée")
                with gr.Column(scale=1):
                    dark_mode = gr.Checkbox(label="Mode sombre", value=False)
            
            with gr.Tabs() as tabs:
                with gr.TabItem("💬 Conversation", id="chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                show_label=False,
                                avatar_images=("👤", "🤖"),
                                height=500,
                                bubble_full_width=False,
                                render_markdown=True,
                                type="messages"  # Compatible avec Python 3.13 et Gradio récent
                            )
                            with gr.Accordion("📚 Sources et références", open=False, visible=False) as sources_accordion:
                                sources_display = gr.Textbox(
                                    label="Sources utilisées pour cette réponse",
                                    interactive=False,
                                    lines=4
                                )
                            with gr.Row():
                                query_input = gr.Textbox(
                                    label="Posez votre question",
                                    placeholder="Ex. : Quelles sont les dates importantes ?",
                                    lines=2
                                )
                                submit_button = gr.Button("Envoyer 🚀", variant="primary")
                            with gr.Accordion("⚙️ Paramètres", open=False):
                                with gr.Row():
                                    domaine_filter = gr.Dropdown(
                                        ["Tous"] + self.unique_domains,
                                        label="Filtrer par domaine",
                                        value="Tous"
                                    )
                                    theme_filter = gr.Dropdown(
                                        ["Tous"] + self.unique_themes,
                                        label="Filtrer par thème",
                                        value="Tous"
                                    )
                                model_select = gr.Dropdown(
                                    list(self.available_models.keys()),
                                    label="Modèle de langage",
                                    value=list(self.available_models.keys())[0],
                                    info="Choisissez le modèle à utiliser pour les réponses"
                                )
                                gr.HTML("""
                                <div style="margin-top: 10px; padding: 10px; background-color: #f0f7ff; border-radius: 5px;">
                                    <strong>📝 Descriptions des modèles :</strong>
                                    <ul style="margin-top: 5px; padding-left: 20px;">
                                        <li><strong>qwen2.5:3b</strong> : Équilibré - Bon ratio performance/rapidité</li>
                                        <li><strong>llama3.2:latest</strong> : Performant - Réponses détaillées</li>
                                        <li><strong>deepseek-r1:1.5b</strong> : Rapide - Réponses concises</li>
                                        <li><strong>qwen2:1.5b</strong> : Très rapide - Idéal pour des questions simples</li>
                                    </ul>
                                </div>
                                """)
                            with gr.Row():
                                clear_button = gr.Button("🗑️ Effacer la conversation")
                
                with gr.TabItem("📥 Indexation", id="indexation"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("""
                            ### Enrichissez la base de connaissances
                            
                            Ajoutez vos propres documents pour améliorer les réponses du chatbot.
                            Classez-les par domaine et thème pour faciliter la recherche.
                            """)
                            with gr.Column():  # Remplacement de gr.Box()
                                file_input = gr.File(
                                    label="Sélectionnez des fichiers à indexer",
                                    file_types=[".txt", ".docx", ".csv", ".pdf", ".json"],
                                    file_count="multiple"
                                )
                                with gr.Row():
                                    with gr.Column():
                                        domaine_input = gr.Dropdown(
                                            ["Nouveau"] + self.unique_domains,
                                            label="Domaine",
                                            value="Nouveau"
                                        )
                                        custom_domaine = gr.Textbox(
                                            label="Nouveau domaine",
                                            placeholder="Entrez un nom de domaine",
                                            visible=False
                                        )
                                    with gr.Column():
                                        theme_input = gr.Dropdown(
                                            ["Nouveau"] + self.unique_themes,
                                            label="Thème",
                                            value="Nouveau"
                                        )
                                        custom_theme = gr.Textbox(
                                            label="Nouveau thème",
                                            placeholder="Entrez un nom de thème",
                                            visible=False
                                        )
                                with gr.Row():
                                    import_button = gr.Button("📤 Indexer les fichiers", variant="primary")
                                import_output = gr.Textbox(
                                    label="Résultat de l'indexation",
                                    interactive=False,
                                    lines=6
                                )
                            gr.Markdown("""
                            #### Types de fichiers supportés
                            
                            - **Texte (.txt)** : Documents texte standard
                            - **Word (.docx)** : Documents Microsoft Word
                            - **CSV (.csv)** : Fichiers de données tabulaires
                            - **PDF (.pdf)** : Documents PDF (texte uniquement)
                            - **JSON (.json)** : Fichiers de données structurées
                            
                            #### Organisation
                            
                            - **Domaine** : Catégorie principale (ex: Finance, RH, Marketing)
                            - **Thème** : Sous-catégorie (ex: Comptabilité, Recrutement, Campagnes)
                            """)
                
                with gr.TabItem("🔍 Base de connaissances", id="knowledge_base"):
                    with gr.Row():
                        with gr.Column():
                            refresh_stats_button = gr.Button("🔄 Rafraîchir les statistiques")
                            with gr.Row():
                                with gr.Column():
                                    total_docs = gr.Number(label="Documents indexés", value=0, precision=0)
                                with gr.Column():
                                    total_chunks = gr.Number(label="Chunks disponibles", value=0, precision=0)
                                with gr.Column():
                                    total_domaines = gr.Number(label="Domaines", value=0, precision=0)
                                with gr.Column():
                                    total_themes = gr.Number(label="Thèmes", value=0, precision=0)
                            gr.Markdown("### 📊 Répartition des documents par domaine")
                            domains_chart = gr.HTML("""<div style="height: 200px; background-color: #f0f0f0; 
                                                    display: flex; justify-content: center; align-items: center;">
                                                    Les données s'afficheront après actualisation</div>""")
                            gr.Markdown("### 📊 Répartition des documents par thème")
                            themes_chart = gr.HTML("""<div style="height: 200px; background-color: #f0f0f0; 
                                                   display: flex; justify-content: center; align-items: center;">
                                                   Les données s'afficheront après actualisation</div>""")
                            gr.Markdown("### 📑 Liste des documents indexés")
                            docs_table = gr.DataFrame(
                                headers=["Fichier", "Domaine", "Thème", "Taille", "Date d'indexation"],
                                datatype=["str", "str", "str", "str", "str"],
                                row_count=(5, "fixed"),
                                col_count=(5, "fixed")
                            )
            
            # Événements
            submit_button.click(
                fn=self.handle_chat,
                inputs=[query_input, domaine_filter, theme_filter, model_select, chatbot],
                outputs=[query_input, chatbot, sources_accordion, sources_display]
            )
            query_input.submit(
                fn=self.handle_chat,
                inputs=[query_input, domaine_filter, theme_filter, model_select, chatbot],
                outputs=[query_input, chatbot, sources_accordion, sources_display]
            )
            clear_button.click(
                fn=self.reset_chat,
                inputs=[],
                outputs=[chatbot, sources_accordion, sources_display]
            )
            domaine_input.change(
                fn=lambda d: gr.update(visible=d == "Nouveau"),
                inputs=[domaine_input],
                outputs=[custom_domaine]
            )
            theme_input.change(
                fn=lambda t: gr.update(visible=t == "Nouveau"),
                inputs=[theme_input],
                outputs=[custom_theme]
            )
            import_button.click(
                fn=self.process_import,
                inputs=[file_input, domaine_input, theme_input, custom_domaine, custom_theme],
                outputs=[import_output, file_input]
            ).then(
                fn=self.update_dropdowns,
                inputs=[],
                outputs=[domaine_filter, theme_filter]
            ).then(
                fn=self.update_indexing_dropdowns,
                inputs=[],
                outputs=[domaine_input, theme_input]
            )
            def get_kb_stats(self):
                try:
                    all_docs = collection.get(include=["metadatas"])
                    total_documents = len(set(m.get("filename") for m in all_docs["metadatas"] if m.get("filename")))
                    total_chunks = len(all_docs["metadatas"])
                    domains = set(m.get("domaine") for m in all_docs["metadatas"] if m.get("domaine"))
                    themes = set(m.get("theme") for m in all_docs["metadatas"] if m.get("theme"))
                    
                    domain_counts = {}
                    for meta in all_docs["metadatas"]:
                        domain = meta.get("domaine", "Non spécifié")
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    domains_html = """
                    <div style="font-family: Arial; width: 100%;">
                    <div style="display: flex; flex-direction: column; gap: 10px;">
                    """
                    max_count = max(domain_counts.values()) if domain_counts else 1
                    colors = ["#4285F4", "#34A853", "#FBBC05", "#EA4335"]
                    for i, (domain, count) in enumerate(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)):
                        color = colors[i % len(colors)]
                        percentage = (count / max_count) * 100
                        domains_html += f"""
                        <div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                            <span>{domain}</span>
                            <span>{count} chunks</span>
                        </div>
                        <div style="width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 4px; overflow: hidden;">
                            <div style="height: 100%; width: {percentage}%; background-color: {color};"></div>
                        </div>
                        </div>
                        """
                    domains_html += "</div></div>"
                    
                    theme_counts = {}
                    for meta in all_docs["metadatas"]:
                        theme = meta.get("theme", "Non spécifié")
                        theme_counts[theme] = theme_counts.get(theme, 0) + 1
                    themes_html = """
                    <div style="font-family: Arial; width: 100%;">
                    <div style="display: flex; flex-direction: column; gap: 10px;">
                    """
                    max_count = max(theme_counts.values()) if theme_counts else 1
                    for i, (theme, count) in enumerate(sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)):
                        color = colors[i % len(colors)]
                        percentage = (count / max_count) * 100
                        themes_html += f"""
                        <div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                            <span>{theme}</span>
                            <span>{count} chunks</span>
                        </div>
                        <div style="width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 4px; overflow: hidden;">
                            <div style="height: 100%; width: {percentage}%; background-color: {color};"></div>
                        </div>
                        </div>
                        """
                    themes_html += "</div></div>"
                    
                    doc_list = []
                    seen_docs = set()
                    for meta in all_docs["metadatas"]:
                        filename = meta.get("filename")
                        if filename and filename not in seen_docs:
                            seen_docs.add(filename)
                            size = meta.get("size", "N/A")
                            if isinstance(size, (int, float)):
                                size = f"{size:.2f} Ko"  # Formatage avec 2 décimales
                            doc_list.append([
                                filename,
                                meta.get("domaine", "Non spécifié"),
                                meta.get("theme", "Non spécifié"),
                                size,
                                meta.get("date", "N/A")
                            ])
                    doc_list.sort(key=lambda x: x[0])
                    doc_list = doc_list[:15]
                    
                    return total_documents, total_chunks, len(domains), len(themes), domains_html, themes_html, doc_list
                except Exception as e:
                    self.logger.error(f"Erreur lors de la récupération des stats : {str(e)}")
                    return 0, 0, 0, 0, "<p>Erreur lors du chargement</p>", "<p>Erreur lors du chargement</p>", []
            refresh_stats_button.click(
                fn=get_kb_stats,
                inputs=[],
                outputs=[total_docs, total_chunks, total_domaines, total_themes, domains_chart, themes_chart, docs_table]
            )
            dark_mode.change(
                fn=lambda x: gr.update(theme="dark" if x else "soft"),
                inputs=[dark_mode],
                outputs=[demo]
            )
        
        self.demo = demo
        return demo
    
    def periodic_api_check(self, interval=30):
        """Vérifie périodiquement si l'API est accessible."""
        while True:
            time.sleep(interval)
            if not self.is_server_running(8000):
                self.logger.warning("API non accessible")
                if not self.api_running:
                    self.logger.info("Tentative de redémarrage de l'API...")
                    self.start_fastapi_server()
            else:
                self.logger.debug("API accessible")
    
    def run(self):
        """Démarre l'application complète avec interface et API."""
        if not self.check_dependencies():
            self.logger.error("⚠️ Certains packages sont manquants. L'application s'arrête.")
            sys.exit(1)
        
        if not self.is_server_running(8000):
            self.logger.info("📡 Démarrage du serveur API...")
            self.start_fastapi_server()
            time.sleep(3)
        
        demo = self.build_interface()
        api_check_thread = threading.Thread(target=self.periodic_api_check, daemon=True)
        api_check_thread.start()
        demo.launch(share=True)

if __name__ == "__main__":
    app = EnhancedChatbotApplication()
    app.run()