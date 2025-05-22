import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt  # ✅ Added for Altair charts

from utils import (
    load_data, preprocess_data, visualize_data,
    train_model, evaluate_model, available_models
)
from ai_assistant import generate_insights  # Optional

# Page config
st.set_page_config(
    page_title="AI-Powered Data Analyzer",
    page_icon="icon.png",  # Replace with your own favicon path if desired
    layout="wide"
)

st.title("🔍 Big Data Analysis with AI and ML")

# --- Sidebar for file upload ---
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    with st.expander("📊 Raw Data (Click to Expand)", expanded=True):
        df = load_data(uploaded_file)
        st.dataframe(df.head())
        
    # --- Section: Cleaned Data ---
    with st.expander("🧹 Cleaned Data (Click to Expand)"):
        df = preprocess_data(df)
        st.dataframe(df.head())

    # --- Section: Visualizations ---
    with st.expander("📈 Visualizations (Click to Expand)"):
        if st.button("Show Visualizations"):
            st.subheader("📊 Visualizations")
            visualize_data(df)

    # --- Section: ML Model Training and Evaluation ---
    with st.expander("🤖 ML Model Training & Evaluation (Click to Expand)"):
        st.subheader("Select Machine Learning Model & Target Variable")

        model_name = st.selectbox(
            "Model",
            list(available_models.keys())
        )
        target = st.selectbox("Target Variable", df.columns)

        use_cv = st.checkbox("Use Cross-Validation", value=False)
        show_feat_importance = st.checkbox("Show Feature Importance (if supported)", value=True)

        if st.button("🚀 Train & Evaluate Model"):
            with st.spinner("Training your model..."):
                try:
                    model, X_test, y_test, y_pred = train_model(df, target, model_name, use_cv)
                    evaluate_model(model, y_test, y_pred)

                    if show_feat_importance and hasattr(model, "feature_importances_"):
                        st.subheader("🔍 Feature Importance")

                        importance = model.feature_importances_
                        features = X_test.columns
                        importance_df = pd.DataFrame({
                            "Feature": features,
                            "Importance": importance
                        })

                        importance_df["Feature"] = importance_df["Feature"].astype(str)

                        chart = alt.Chart(importance_df).mark_bar().encode(
                            x=alt.X('Feature:N', sort='-y', title='Features'),
                            y=alt.Y('Importance:Q', title='Importance'),
                            color=alt.Color('Feature:N', scale=alt.Scale(scheme='category20b')),
                            tooltip=['Feature', 'Importance']
                        ).properties(
                            width=700,
                            height=400,
                            title="🎨 Feature Importance by Feature (Altair)"
                        )

                        st.altair_chart(chart, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error during training or evaluation: {e}")

    # --- Section: AI Insights ---
    with st.expander("💡 AI-Powered Insights (Click to Expand)"):
        if st.checkbox("Generate AI-Powered Insights"):
            with st.spinner("Generating AI-powered insights..."):
                try:
                    summary = df.describe(include='all').to_string()
                    insights = generate_insights(summary)
                    st.success(insights)
                except Exception as e:
                    st.error(f"❌ Error from Gemini API: {e}")

else:
    st.info("Please upload a CSV or Excel file to get started.")
