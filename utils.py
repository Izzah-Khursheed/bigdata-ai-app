import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

# ✅ Available models
available_models = {
    "Logistic Regression": LogisticRegression,
    "Linear Regression": LinearRegression,
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
}

# ✅ Load CSV/Excel with error handling
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# ✅ Basic preprocessing: drop NA and encode categorical
def preprocess_data(df):
    df = df.dropna()

    # Encode categorical features
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# ✅ Visualizations
def visualize_data(df):
    num_df = df.select_dtypes(include='number')
    
    # Pairplot with Matplotlib
    fig = sns.pairplot(num_df)
    st.pyplot(fig)

    # Plotly histograms
    for col in num_df.columns:
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

# ✅ Train model
def train_model(df, target_col, model_name, use_cv=False):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = available_models[model_name]()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

# ✅ Evaluate model
def evaluate_model(model, y_true, y_pred):
    st.subheader("📊 Model Evaluation")

    if is_regression_model(model):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        st.write(f"**Mean Squared Error:** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")
    else:
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        st.write(f"**Accuracy:** {acc:.2f}")
        st.write("**Confusion Matrix:**")
        st.write(cm)
        st.text("Classification Report:\n" + report)

# ✅ Feature importance (for tree-based models only)
def show_feature_importance(model, X_test):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': importance
        }).sort_values(by="Importance", ascending=False)

        st.subheader("🔍 Feature Importance")
        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.warning("This model does not support feature importance.")

# ✅ Helper to check if model is for regression
def is_regression_model(model):
    return isinstance(model, LinearRegression)