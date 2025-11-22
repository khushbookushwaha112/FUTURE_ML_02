import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("🏦 Bank Customer Churn Prediction — Task 2")

MODEL_PATH = "models/clean_churn_model.pkl"
DATA_PATH = "data/bank_clean.csv"

# ---------------- SAFE CSV READER ----------------
def safe_load(file):
    try:
        return pd.read_csv(file)
    except:
        try:
            return pd.read_csv(file, encoding="latin1")
        except:
            return pd.read_csv(file, sep=";", encoding="latin1")

# ---------------- PREPROCESS (MATCH NOTEBOOK) ----------------
def preprocess(df):
    df = df.copy()

    # Drop ID columns (always drop)
    for col in ["RowNumber", "CustomerId", "Surname"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # One-hot encode Geography + Gender (same as training)
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=False)

    # Required columns in MODEL (exact order)
    required = [
        'CreditScore','Age','Tenure','Balance','NumOfProducts',
        'HasCrCard','IsActiveMember','EstimatedSalary',
        'Geography_France','Geography_Germany','Geography_Spain',
        'Gender_Female','Gender_Male'
    ]

    # Add missing columns
    for col in required:
        if col not in df.columns:
            df[col] = 0

    # Keep exact order
    df = df[required]

    return df

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found! Upload clean_churn_model.pkl in /models/")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- UI ----------------
uploaded = st.file_uploader("Upload CSV (same columns as original)", type=["csv"])

if uploaded:
    df = safe_load(uploaded)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    if st.button("Predict Churn"):
        if model is None:
            st.error("Model missing!")
        else:
            clean_df = preprocess(df)

            preds = model.predict(clean_df)
            probs = model.predict_proba(clean_df)[:, 1]

            df["predicted_churn"] = preds
            df["churn_probability"] = probs

            st.subheader("Prediction Results")
            st.dataframe(df.head(15))

            st.download_button(
                "Download Results CSV",
                df.to_csv(index=False),
                "predictions.csv"
            )

else:
    st.info("Upload a CSV to begin prediction.")
