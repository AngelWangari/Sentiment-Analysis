import joblib
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_lr_svm.pkl")
VECT_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

def predict_sentiment(text):

    if not text or text.strip() == "":
        return "Empty input"

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    return prediction