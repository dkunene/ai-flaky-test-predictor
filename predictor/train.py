import joblib
from predictor.features import preprocess
from sklearn.ensemble import RandomForestClassifier

def train(csv_path, model_path):
    df = preprocess(csv_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, model_path)
