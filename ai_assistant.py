import requests
import os

def generate_insights(summary):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mixtral-8x7b-32768",  # or llama3-70b
        "messages": [
            {"role": "system", "content": "You're a data analyst. Provide insights from data summaries."},
            {"role": "user", "content": summary}
        ],
        "temperature": 0.5
    }

    response = requests.post(url, json=payload, headers=headers, timeout=15)
    return response.json()['choices'][0]['message']['content']
