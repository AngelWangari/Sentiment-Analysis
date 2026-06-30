
import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

# ==========================
# LOAD DATA
# ==========================

DATA_PATH = r"C:\Users\EB-PC\Documents\ml2\archive (5)\Combined Data.csv"

df = pd.read_csv(DATA_PATH, encoding="latin1")

# ==========================
# CLEAN TEXT
# ==========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["statement"] = df["statement"].fillna("").apply(clean_text)

label_mapping = {
    "Suicidal": "very negative",
    "Depression": "very negative",
    "Anxiety": "negative",
    "Stress": "negative",
    "Bipolar": "negative",
    "Personality disorder": "negative",
    "Normal": "neutral"
}

df["status"] = df["status"].map(label_mapping)
df = df.dropna(subset=["status"])

# ==========================
# SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    df["statement"],
    df["status"],
    test_size=0.2,
    random_state=42,
    stratify=df["status"]
)

# ==========================
# TFIDF (8000 FEATURES)
# ==========================

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)

# ==========================
# MODELS
# ==========================

lr = LogisticRegression(
    C=1,
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced"
)

svm = LinearSVC(
    C=0.1,
    class_weight="balanced",
    dual=False
)

ensemble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("svm", svm)
    ],
    voting="hard"
)

ensemble.fit(X_train_vec, y_train)

# ==========================
# SAVE
# ==========================

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(ensemble, "models/hybrid_lr_svm.pkl")

print("DONE")
print("Vectorizer Features:", X_train_vec.shape[1])
