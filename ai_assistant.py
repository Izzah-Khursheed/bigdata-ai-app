import google.generativeai as genai
import os
import streamlit as st

# ✅ Load API key securely from environment variable
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("⚠️ GEMINI_API_KEY is not set in environment variables.")

# ✅ Function to get AI-generated insight
def generate_insights(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating insight: {str(e)}"
