# =========================
# EDA VISUALIZATION SCRIPT
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

# =========================
# LOAD DATA
# =========================

DATA_PATH = r"C:\Users\EB-PC\Documents\ml2\archive (5)\Combined Data.csv"

df = pd.read_csv(DATA_PATH, encoding="latin1")

# =========================
# BASIC CLEANING
# =========================

df = df.copy()

df = df.dropna(subset=["statement", "status"])
df = df[df["statement"].astype(str).str.strip() != ""]

# =========================
# CREATE FEATURES
# =========================

df["word_count"] = df["statement"].astype(str).apply(
    lambda x: len(x.split())
)

df["text_length"] = df["statement"].astype(str).apply(
    len
)

print("Dataset shape after cleaning:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# =========================
# 1. CLASS DISTRIBUTION
# =========================

plt.figure(figsize=(7, 5))

sns.countplot(
    x="status",
    data=df,
    order=df["status"].value_counts().index
)

plt.title("Class Distribution")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# =========================
# 2. WORD COUNT DISTRIBUTION
# =========================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["word_count"],
    bins=50,
    kde=True
)

plt.title("Word Count Distribution")
plt.xlabel("Word Count")
plt.tight_layout()
plt.show()

# =========================
# 3. TEXT LENGTH DISTRIBUTION
# =========================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["text_length"],
    bins=50,
    kde=True
)

plt.title("Text Length Distribution")
plt.xlabel("Number of Characters")
plt.tight_layout()
plt.show()

# =========================
# 4. WORD COUNT vs SENTIMENT
# =========================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="status",
    y="word_count",
    data=df
)

plt.title("Word Count by Sentiment Class")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# =========================
# 5. TEXT LENGTH vs SENTIMENT
# =========================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="status",
    y="text_length",
    data=df
)

plt.title("Text Length by Sentiment Class")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# =========================
# 6. TF-IDF + PCA
# =========================

print("\nRunning TF-IDF...")

sample_df = df.sample(
    min(5000, len(df)),
    random_state=42
)

tfidf = TfidfVectorizer(
    max_features=3000,
    stop_words="english"
)

X = tfidf.fit_transform(
    sample_df["statement"].astype(str)
)

print("Running PCA...")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(
    X.toarray()
)

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=sample_df["status"],
    palette="Set1",
    s=20,
    alpha=0.7
)

plt.title("PCA Visualization of TF-IDF Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

# =========================
# DONE
# =========================

print("\nEDA Visualizations Complete.")