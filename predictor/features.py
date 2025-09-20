import pandas as pd

def preprocess(df: pd.DataFrame, training=True):
    # Example preprocessing – replace with real flaky test features
    if "label" in df.columns:
        y = df["label"]
        X = df.drop(columns=["label"])
    else:
        y = None
        X = df

    return X, y
