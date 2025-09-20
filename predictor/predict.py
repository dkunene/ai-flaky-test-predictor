import pandas as pd
import joblib
from predictor.features import preprocess
import os

def predict(model_path: str, data_path: str) -> str:
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X, _ = preprocess(df, training=False)

    predictions = model.predict(X)
    df["prediction"] = predictions

    output_file = "data/predictions.csv"
    df.to_csv(output_file, index=False)

    print(f"✅ Predictions saved to {output_file}")
    return output_file
