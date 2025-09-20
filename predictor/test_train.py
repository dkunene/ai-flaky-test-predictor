import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from predictor.features import preprocess

def train_model(data_path: str, model_path: str):
    df = pd.read_csv(data_path)
    X, y = preprocess(df)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved at {model_path}")
