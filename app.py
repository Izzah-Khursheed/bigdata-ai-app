import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_data, preprocess_data, visualize_data,
    train_model, evaluate_model, available_models
)
from ai_assistant import generate_insights  # Optional

# ✅ Set Streamlit page config early
st.set_page_config(page_title="AI-Powered Data Analyzer", layout="wide")

# ✅ App title and sidebar file upload
st.title("🔍 Big Data Analysis with AI and ML")
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # ✅ Load data
    df = load_data(uploaded_file)
    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # ✅ Preprocess data
    df = preprocess_data(df)
    st.subheader("🧹 Cleaned Data")
    st.dataframe(df.head())

    # ✅ Show visualizations
    if st.button("📈 Show Visualizations"):
        st.subheader("📊 Visualizations")
        visualize_data(df)

    # ✅ Model selection UI
    st.subheader("🤖 ML Model Selection & Evaluation")

    model_name = st.selectbox(
        "Select Machine Learning Model",
        list(available_models.keys())
    )
    target = st.selectbox("Select Target Variable", df.columns)

    use_cv = st.checkbox("Use Cross-Validation", value=False)
    show_feat_importance = st.checkbox("Show Feature Importance (if supported)", value=True)

    # ✅ Train and evaluate
    if st.button("🚀 Train & Evaluate Model"):
        with st.spinner("Training your model..."):
            try:
                model, X_test, y_test, y_pred = train_model(df, target, model_name, use_cv)
                evaluate_model(y_test, y_pred)

                if show_feat_importance and hasattr(model, "feature_importances_"):
                    st.subheader("🔍 Feature Importance")
                    importance = model.feature_importances_
                    st.bar_chart(pd.Series(importance, index=X_test.columns))
            except Exception as e:
                st.error(f"❌ Error during training or evaluation: {e}")

    # ✅ AI-powered Gemini insights
    if st.checkbox("💡 Generate AI-Powered Insights"):
        with st.spinner("Generating AI-powered insights..."):
            try:
                summary = df.describe(include='all').to_string()
                insights = generate_insights(summary)
                st.success(insights)
            except Exception as e:
                st.error(f"❌ Error from Gemini API: {e}")