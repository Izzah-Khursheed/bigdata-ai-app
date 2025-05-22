import streamlit as st
import requests

# ✅ Access Groq API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def generate_insights(summary):
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": "You're a data analyst."},
                {"role": "user", "content": f"Here is a dataset summary:\n{summary}\nGive 5 smart insights."}
            ],
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        # Debugging (optional): st.write(result)

        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return f"❌ Groq API error: {result.get('error', result)}"

    except Exception as e:
        return f"❌ Exception occurred: {e}"
