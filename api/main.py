from fastapi import FastAPI

app = FastAPI(title="AI Flaky Test Predictor")

@app.get("/")
def read_root():
    return {"message": "AI Flaky Test Predictor API is running"}
