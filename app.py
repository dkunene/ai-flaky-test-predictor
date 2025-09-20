from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os

from predictor.features import preprocess
from predictor.train import train   # your function is called 'train'
from predictor.predict import predict

app = FastAPI()

MODEL_PATH = "models/model.pkl"

# Serve frontend and data
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")
app.mount("/data", StaticFiles(directory="data"), name="data")


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Flaky Test Predictor API is running 🚀"}


@app.post("/train")
async def train_model_endpoint(file: UploadFile = File(...)):
    """Train model on uploaded CSV"""
    if not os.path.exists("data"):
        os.makedirs("data")
    temp_csv = "data/temp_train.csv"

    df = pd.read_csv(file.file)
    df.to_csv(temp_csv, index=False)

    if not os.path.exists("models"):
        os.makedirs("models")
    train(temp_csv, MODEL_PATH)

    return {"status": "success", "message": "Model trained successfully"}


@app.post("/predict")
async def predict_model_endpoint(file: UploadFile = File(...)):
    """Predict using trained model"""
    if not os.path.exists(MODEL_PATH):
        return {"status": "error", "message": "No trained model found. Train first."}

    if not os.path.exists("data"):
        os.makedirs("data")
    temp_csv = "data/temp_predict.csv"

    df = pd.read_csv(file.file)
    df.to_csv(temp_csv, index=False)

    output_file = predict(MODEL_PATH, temp_csv)
    return {"status": "success", "predictions_file": f"data/{os.path.basename(output_file)}"}
