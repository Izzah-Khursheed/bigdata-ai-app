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

# Algorithm info data (add more or adjust descriptions as you like)
algorithm_info = {
    "Logistic Regression": {
        "Description": "A linear model used for binary classification tasks.",
        "Use Cases": "Spam detection, credit scoring, medical diagnosis.",
        "Pros": "Simple, interpretable, efficient.",
        "Cons": "Not suitable for complex relationships.",
        "Key Parameters": "penalty, C (regularization strength), solver."
    },
    "Linear Regression": {
        "Description": "Predicts a continuous target variable based on input features.",
        "Use Cases": "Sales forecasting, risk assessment, price prediction.",
        "Pros": "Easy to implement, interpretable.",
        "Cons": "Assumes linear relationship, sensitive to outliers.",
        "Key Parameters": "fit_intercept, normalize."
    },
    "Random Forest": {
        "Description": "An ensemble of decision trees for classification and regression.",
        "Use Cases": "Feature selection, anomaly detection, classification tasks.",
        "Pros": "Robust, handles non-linear data well, reduces overfitting.",
        "Cons": "Less interpretable, can be slow with large data.",
        "Key Parameters": "n_estimators, max_depth, min_samples_split."
    },
    "SVM": {
        "Description": "Finds the hyperplane that best separates classes.",
        "Use Cases": "Text categorization, image classification.",
        "Pros": "Effective in high-dimensional spaces.",
        "Cons": "Memory intensive, not suited for very large datasets.",
        "Key Parameters": "kernel, C, gamma."
    },
    "K-Nearest Neighbors (KNN)": {
        "Description": "Classifies a data point based on neighbors' classes.",
        "Use Cases": "Recommendation systems, pattern recognition.",
        "Pros": "Simple, no training phase.",
        "Cons": "Computationally expensive at prediction time.",
        "Key Parameters": "n_neighbors, weights, metric."
    },
    "K-Means Clustering": {
        "Description": "Partitions data into K clusters based on feature similarity.",
        "Use Cases": "Customer segmentation, image compression.",
        "Pros": "Simple, fast for small datasets.",
        "Cons": "Needs predefined K, sensitive to outliers.",
        "Key Parameters": "n_clusters, init, max_iter."
    },
    "Naive Bayes": {
        "Description": "Probabilistic classifier based on Bayes’ theorem.",
        "Use Cases": "Spam filtering, document classification.",
        "Pros": "Fast, handles high-dimensional data well.",
        "Cons": "Assumes feature independence.",
        "Key Parameters": "var_smoothing (for Gaussian NB)."
    },
    "Decision Tree": {
        "Description": "Tree-based model splitting data by feature values.",
        "Use Cases": "Credit scoring, medical diagnosis.",
        "Pros": "Easy to interpret, handles categorical data.",
        "Cons": "Prone to overfitting.",
        "Key Parameters": "max_depth, min_samples_split."
    },
    "Apriori Algorithm": {
        "Description": "Identifies frequent itemsets and association rules.",
        "Use Cases": "Market basket analysis, recommendation systems.",
        "Pros": "Simple, interpretable rules.",
        "Cons": "Can be slow on large datasets.",
        "Key Parameters": "min_support, min_confidence."
    }
}

if uploaded_file:
    df = load_data(uploaded_file)

    # Use tabs instead of expanders
    tabs = st.tabs([
        "📊 Raw Data",
        "🧹 Cleaned Data",
        "📈 Visualizations",
        "🤖 ML Model Training & Evaluation",
        "💡 AI-Powered Insights",
        "📚 Algorithm Info"
    ])

    # Tab 1: Raw Data
    with tabs[0]:
        st.subheader("📊 Raw Data")
        st.dataframe(df.head())

    # Tab 2: Cleaned Data
    with tabs[1]:
        st.subheader("🧹 Cleaned Data")
        df = preprocess_data(df)
        st.dataframe(df.head())

    # Tab 3: Visualizations
    with tabs[2]:
        st.subheader("📈 Visualizations")

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Histogram", "Bar Chart", "Pie Chart", "Line Chart"]
        )

        if st.button("Show Visualization"):
            visualize_data(df, chart_type)


    # Tab 4: ML Model Training & Evaluation
    with tabs[3]:
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

    # Tab 5: AI-Powered Insights
    with tabs[4]:
        st.subheader("💡 AI-Powered Insights")
        if st.checkbox("Generate AI-Powered Insights"):
            with st.spinner("Generating AI-powered insights..."):
                try:
                    summary = df.describe(include='all').to_string()
                    insights = generate_insights(summary)
                    st.success(insights)
                except Exception as e:
                    st.error(f"❌ Error from Groq API: {e}")

    # Tab 6: Algorithm Info
    with tabs[5]:
        st.subheader("📚 Algorithm Info")

        for algo, info in algorithm_info.items():
            with st.expander(algo):
                st.markdown(f"**Description:** {info['Description']}")
                st.markdown(f"**Use Cases:** {info['Use Cases']}")
                st.markdown(f"**Pros:** {info['Pros']}")
                st.markdown(f"**Cons:** {info['Cons']}")
                st.markdown(f"**Key Parameters:** {info['Key Parameters']}")

else:
    st.info("Please upload a CSV or Excel file to get started.")













# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import altair as alt  # ✅ Added for Altair charts

# from utils import (
#     load_data, preprocess_data, visualize_data,
#     train_model, evaluate_model, available_models
# )
# from ai_assistant import generate_insights  # Optional

# # Page config
# st.set_page_config(
#     page_title="AI-Powered Data Analyzer",
#     page_icon="icon.png",  # Replace with your own favicon path if desired
#     layout="wide"
# )

# st.title("🔍 Big Data Analysis with AI and ML")

# # --- Sidebar for file upload ---
# st.sidebar.header("📁 Upload Your Dataset")
# uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# if uploaded_file:
#     with st.expander("📊 Raw Data (Click to Expand)", expanded=True):
#         df = load_data(uploaded_file)
#         st.dataframe(df.head())
        
#     # --- Section: Cleaned Data ---
#     with st.expander("🧹 Cleaned Data (Click to Expand)"):
#         df = preprocess_data(df)
#         st.dataframe(df.head())

#     # --- Section: Visualizations ---
#     with st.expander("📈 Visualizations (Click to Expand)"):
#         if st.button("Show Visualizations"):
#             st.subheader("📊 Visualizations")
#             visualize_data(df)

#     # --- Section: ML Model Training and Evaluation ---
#     with st.expander("🤖 ML Model Training & Evaluation (Click to Expand)"):
#         st.subheader("Select Machine Learning Model & Target Variable")

#         model_name = st.selectbox(
#             "Model",
#             list(available_models.keys())
#         )
#         target = st.selectbox("Target Variable", df.columns)

#         use_cv = st.checkbox("Use Cross-Validation", value=False)
#         show_feat_importance = st.checkbox("Show Feature Importance (if supported)", value=True)

#         if st.button("🚀 Train & Evaluate Model"):
#             with st.spinner("Training your model..."):
#                 try:
#                     model, X_test, y_test, y_pred = train_model(df, target, model_name, use_cv)
#                     evaluate_model(model, y_test, y_pred)

#                     if show_feat_importance and hasattr(model, "feature_importances_"):
#                         st.subheader("🔍 Feature Importance")

#                         importance = model.feature_importances_
#                         features = X_test.columns
#                         importance_df = pd.DataFrame({
#                             "Feature": features,
#                             "Importance": importance
#                         })

#                         importance_df["Feature"] = importance_df["Feature"].astype(str)

#                         chart = alt.Chart(importance_df).mark_bar().encode(
#                             x=alt.X('Feature:N', sort='-y', title='Features'),
#                             y=alt.Y('Importance:Q', title='Importance'),
#                             color=alt.Color('Feature:N', scale=alt.Scale(scheme='category20b')),
#                             tooltip=['Feature', 'Importance']
#                         ).properties(
#                             width=700,
#                             height=400,
#                             title="🎨 Feature Importance by Feature (Altair)"
#                         )

#                         st.altair_chart(chart, use_container_width=True)

#                 except Exception as e:
#                     st.error(f"❌ Error during training or evaluation: {e}")

#     # --- Section: AI Insights ---
#     with st.expander("💡 AI-Powered Insights (Click to Expand)"):
#         if st.checkbox("Generate AI-Powered Insights"):
#             with st.spinner("Generating AI-powered insights..."):
#                 try:
#                     summary = df.describe(include='all').to_string()
#                     insights = generate_insights(summary)
#                     st.success(insights)
#                 except Exception as e:
#                     st.error(f"❌ Error from Gemini API: {e}")

# else:
#     st.info("Please upload a CSV or Excel file to get started.")
