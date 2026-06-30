import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# =====================================================
# 1. LOAD DATA
# =====================================================

DATA_PATH = r"C:\Users\EB-PC\Documents\ml2\archive (5)\Combined Data.csv"

df = pd.read_csv(DATA_PATH, encoding="latin1")

print("\n================ DATA OVERVIEW ================")
print("Shape:", df.shape)
print("\nColumns:", df.columns)
print("\nFirst 5 rows:\n", df.head())

# =====================================================
# 2. CLEAN TEXT
# =====================================================

df["statement"] = df["statement"].fillna("").astype(str)
df["statement"] = df["statement"].str.lower().str.strip()
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

# =====================================================
# 4. MISSING VALUES CHECK
# =====================================================

print("\n================ MISSING VALUES ================")
print(df.isnull().sum())

# =====================================================
# 5. CLASS DISTRIBUTION
# =====================================================

print("\n================ CLASS DISTRIBUTION ================")
print(df["status"].value_counts())

print("\nPercentages:")
print(round(df["status"].value_counts(normalize=True) * 100, 2))

plt.figure()
df["status"].value_counts().plot(kind="bar")
plt.title("Sentiment Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =====================================================
# 6. TEXT LENGTH ANALYSIS
# =====================================================

df["text_length"] = df["statement"].apply(len)
df["word_count"] = df["statement"].apply(lambda x: len(x.split()))

print("\n================ TEXT LENGTH STATS ================")
print(df["text_length"].describe())

print("\n================ WORD COUNT STATS ================")
print(df["word_count"].describe())

plt.figure()
plt.hist(df["text_length"], bins=50)
plt.title("Text Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(df["word_count"], bins=50)
plt.title("Word Count Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# =====================================================
# 7. LENGTH BY CLASS
# =====================================================

print("\n================ AVG TEXT LENGTH BY CLASS ================")
print(df.groupby("status")["text_length"].mean())

plt.figure()
sns.boxplot(x="status", y="text_length", data=df)
plt.title("Text Length by Sentiment Class")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# =====================================================
# 8. MOST COMMON WORDS
# =====================================================

all_words = " ".join(df["statement"]).split()
word_freq = Counter(all_words)

print("\n================ TOP 20 WORDS ================")
print(word_freq.most_common(20))

top_words = word_freq.most_common(20)
words, counts = zip(*top_words)

plt.figure()
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.title("Top 20 Most Frequent Words")
plt.tight_layout()
plt.show()

# =====================================================
# 9. FINAL INSIGHT SUMMARY
# =====================================================

print("\n================ EDA SUMMARY ================")
print("""
- Dataset contains text statements classified into sentiment categories.
- Data is imbalanced (very negative dominates).
- Text lengths vary significantly across classes.
- Negative sentiments tend to have longer statements.
- Certain emotional keywords dominate vocabulary distribution.
""")