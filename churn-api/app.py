import streamlit as st
import pandas as pd
import pickle

# Load model
loaded = pickle.load(open("model.pkl", "rb"))

if not isinstance(loaded, dict):
    st.error("Invalid model file format. Please retrain and regenerate model.pkl.")
    st.stop()

required_keys = ["pipeline", "columns", "categorical_cols", "numerical_cols"]
missing_keys = [key for key in required_keys if key not in loaded]
if missing_keys:
    st.error(
        "model.pkl is missing required keys: "
        + ", ".join(missing_keys)
        + ". Please retrain and regenerate model.pkl."
    )
    st.stop()

model = loaded["pipeline"]
columns = loaded["columns"]
categorical_cols = loaded["categorical_cols"]
numerical_cols = loaded["numerical_cols"]
# Optional metadata for compatibility with older/newer model artifacts.
target = loaded.get("target")

st.title("Churn Predictor")

file = st.file_uploader("Upload CSV")

if file:
    data = pd.read_csv(file)

    # Drop ID
    if "customerID" in data.columns:
        data = data.drop("customerID", axis=1)

    # Align columns first
    data = data.reindex(columns=columns, fill_value=0)

    # 🔥 FIX TYPES EXACTLY
    for col in numerical_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)

    # Handle missing
    data = data.fillna(0)

    # Predict
    prediction = model.predict(data)
    data["Prediction"] = prediction

    st.write(data)

    st.write("Prediction Count:")
    st.write(data["Prediction"].value_counts())