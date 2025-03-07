import requests
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000/rag"

def chat_with_bot(query: str, model: str, domaine: str = None, theme: str = None) -> dict:
    """
    Envoie une requête au chatbot avec filtrage par domaine et thème.
    
    Args:
        query: La question posée par l'utilisateur
        model: Le modèle à utiliser pour la génération de réponse
        domaine: Filtre optionnel par domaine
        theme: Filtre optionnel par thème
        
    Returns:
        dict: Contient la réponse et le contexte utilisé
    """
    payload = {
        "query": query, 
        "model": model,
        "filters": {}
    }
    
    # Ajouter les filtres s'ils sont spécifiés et ne sont pas "Tous"
    if domaine and domaine != "Tous":
        payload["filters"]["domaine"] = domaine
    if theme and theme != "Tous":
        payload["filters"]["theme"] = theme
    
    logger.info(f"Envoi de la requête avec filtres: {payload}")
    
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur de connexion API: {str(e)}")
        return {
            "response": f"Erreur lors de la connexion à l'API : {str(e)}", 
            "context": []
        }
    except Exception as e:
        logger.error(f"Erreur générale: {str(e)}")
        return {
            "response": f"Une erreur est survenue : {str(e)}",
            "context": []
        }