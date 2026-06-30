from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# -----------------------
# CONFIG
# -----------------------
MODEL_DIR = r"C:\Users\EB-PC\Documents\ml2\archive (3)\models"
model = joblib.load("models/hybrid_lr_svm.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# -----------------------
# FASTAPI APP
# -----------------------
app = FastAPI(title="Realtime Sentiment Analysis API")

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze_sentiment(request: TextRequest):
    review = [request.text.lower()]
    vec = vectorizer.transform(review)
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    scores = dict(zip(model.classes_, proba))
    return {
        "label": pred,
        "confidence": max(proba),
        "scores": scores,
        "model": "tfidf-logistic"
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}