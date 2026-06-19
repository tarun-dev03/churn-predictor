from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
loaded = pickle.load(open("model.pkl", "rb"))
if not isinstance(loaded, dict):
    raise RuntimeError("Invalid model.pkl format. Expected a dict.")

required_keys = ["pipeline", "columns", "categorical_cols", "numerical_cols"]
missing_keys = [key for key in required_keys if key not in loaded]
if missing_keys:
    raise RuntimeError(
        "model.pkl is missing required keys: " + ", ".join(missing_keys)
    )

model = loaded["pipeline"]
columns = loaded["columns"]
categorical_cols = loaded["categorical_cols"]
numerical_cols = loaded["numerical_cols"]
target = loaded.get("target")


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}") from exc

    # Drop target if exists
    if target and target in df.columns:
        df = df.drop(target, axis=1)

    # Align columns
    df = df.reindex(columns=columns, fill_value=0)

    # Fix types
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_cols:
        df[col] = df[col].astype(str)

    df = df.fillna(0)

    preds = model.predict(df)

    return {"predictions": preds.tolist()}