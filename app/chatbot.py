import requests

def chat_with_bot(query, model="qwen2.5:3b"):
    # print('chat_with_bot',query)
    url = "http://127.0.0.1:8000/rag"
    response = requests.post(url, json={"query": query, "model": model})

    if response.status_code == 200:
        return response.json()["response"]
    return "Erreur lors de la requête."
