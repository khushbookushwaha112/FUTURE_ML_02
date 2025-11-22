import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os, zipfile

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("🏦 Bank Customer Churn Prediction — Task 2")

MODEL_ZIP = "models/clean_churn_model.zip"
MODEL_PATH = "models/clean_churn_model.pkl"

# ------------------ UNZIP MODEL ------------------
def extract_model():
    if os.path.exists(MODEL_ZIP) and not os.path.exists(MODEL_PATH):
        with zipfile.ZipFile(MODEL_ZIP, 'r') as z:
            z.extractall("models")
        st.success("Model extracted successfully!")

# Extract model on startup
extract_model()

# ------------------ LOAD MODEL ------------------
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error("❌ clean_churn_model.pkl missing from /models/")
        return None

model = load_model()


# ------------------ CSV LOADER ------------------
def safe_load_csv(file):
    try:
        return pd.read_csv(file)
    except:
        return pd.read_csv(file, encoding="latin1")


# ------------------ PREPROCESS ------------------
def preprocess(df):
    df = df.copy()
    
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=False)

    required = [
        "CreditScore","Age","Tenure","Balance","NumOfProducts",
        "HasCrCard","IsActiveMember","EstimatedSalary",
        "Geography_France","Geography_Germany","Geography_Spain",
        "Gender_Female","Gender_Male"
    ]

    for col in required:
        if col not in df.columns:
            df[col] = 0

    df = df[required]
    return df


# ------------------ UI + prediction ------------------
uploaded = st.file_uploader("Upload Bank Churn CSV", type=["csv"])

if uploaded:
    df = safe_load_csv(uploaded)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    if st.button("Predict"):
        processed = preprocess(df)
        preds = model.predict(processed)
        probs = model.predict_proba(processed)[:,1]

        df["predicted_churn"] = preds
        df["churn_probability"] = probs

        st.subheader("Prediction Results")
        st.dataframe(df.head(10))

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "predictions.csv"
        )

else:
    st.info("Upload CSV to see predictions.")
