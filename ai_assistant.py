# ai_assistant.py

import requests

# 🔐 Directly use your Groq API key (replace with your real one)
GROQ_API_KEY = "your_actual_groq_api_key_here"

def generate_insights(summary):
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mixtral-8x7b-32768",  # Or use "llama3-70b-8192", "gemma-7b-it", etc.
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data analyst. Generate smart, helpful, and concise insights from a dataset summary."
                },
                {
                    "role": "user",
                    "content": f"Here is a dataset summary:\n{summary}\nPlease provide 5 smart insights."
                }
            ],
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ Error from Groq API: {e}"
