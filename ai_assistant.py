import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_insights(summary):
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mixtral-8x7b-32768",  # or "llama3-70b-8192", etc.
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data analyst that gives smart insights from statistics."
                },
                {
                    "role": "user",
                    "content": f"Here is a dataset summary:\n{summary}\nGive me smart insights in 5 bullet points."
                }
            ],
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ Error from Groq API: {e}"
