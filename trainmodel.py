import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

# =====================================================
# 1. LOAD DATA
# =====================================================

DATA_PATH = r"C:\Users\EB-PC\Documents\ml2\archive (5)\Combined Data.csv"

df = pd.read_csv(DATA_PATH, encoding="latin1")

# =====================================================
# 2. CLEAN TEXT
# =====================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)

    replacements = {
        "kms": "kill myself",
        "suicide": "end my life",
        "self harm": "self_harm",
        "panic attack": "panic_attack",
        "want to die": "want_to_die"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-zA-Z\s_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["statement"] = df["statement"].fillna("").apply(clean_text)
df = df[df["statement"] != ""]

# =====================================================
# 3. LABEL MAPPING
# =====================================================

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

print("\nCLASS DISTRIBUTION")
print(df["status"].value_counts())

# =====================================================
# 4. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["statement"],
    df["status"],
    test_size=0.20,
    random_state=42,
    stratify=df["status"]
)

# =====================================================
# 5. TF-IDF
# =====================================================

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nTF-IDF SHAPE:", X_train_vec.shape)

# =====================================================
# 6. HYPERPARAMETER TUNING
# =====================================================

print("\nSTARTING GRID SEARCH...")

# ----------------------------
# Logistic Regression
# ----------------------------

lr_params = {
    "C": [0.1, 1, 10],
    "solver": ["lbfgs", "liblinear"]
}

lr_grid = GridSearchCV(
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    lr_params,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1
)

lr_grid.fit(X_train_vec, y_train)

lr = lr_grid.best_estimator_

print("\nBEST LR PARAMETERS")
print(lr_grid.best_params_)

# ----------------------------
# Naive Bayes
# ----------------------------

nb_params = {
    "alpha": [0.1, 0.5, 1.0]
}

nb_grid = GridSearchCV(
    MultinomialNB(),
    nb_params,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1
)

nb_grid.fit(X_train_vec, y_train)

nb = nb_grid.best_estimator_

print("\nBEST NB PARAMETERS")
print(nb_grid.best_params_)

# ----------------------------
# Linear SVM
# ----------------------------

svm_params = {
    "C": [0.1, 1, 10]
}

svm_grid = GridSearchCV(
    LinearSVC(
        class_weight="balanced",
        dual=False
    ),
    svm_params,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1
)

svm_grid.fit(X_train_vec, y_train)

svm = svm_grid.best_estimator_

print("\nBEST SVM PARAMETERS")
print(svm_grid.best_params_)

# =====================================================
# 7. EVALUATE MODELS
# =====================================================

models = {
    "Logistic Regression": lr,
    "Naive Bayes": nb,
    "Linear SVM": svm
}

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    predictions = model.predict(X_test_vec)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")

    print(classification_report(y_test, predictions))

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Macro F1": f1
    })

# =====================================================
# 8. SAVE MODELS
# =====================================================

os.makedirs("models", exist_ok=True)

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(lr, "models/lr.pkl")
joblib.dump(nb, "models/nb.pkl")
joblib.dump(svm, "models/svm.pkl")

# =====================================================
# 9. RESULTS SUMMARY
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\nFINAL MODEL COMPARISON")
print(results_df)

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

best_model_name = results_df.iloc[0]["Model"]

print("\nBEST MODEL:")
print(best_model_name)

print("\nTRAINING COMPLETE")