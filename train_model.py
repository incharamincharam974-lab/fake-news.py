import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1. Load Dataset
# -----------------------------

df = pd.read_csv("fake_news.csv")

print("Dataset loaded successfully!")
print(df.head())


# -----------------------------
# 2. Clean Text
# -----------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["text"] = df["text"].apply(clean_text)


# -----------------------------
# 3. Convert Labels
# -----------------------------

df["label"] = df["label"].map({
    "FAKE": 0,
    "REAL": 1
})


# Remove missing values
df = df.dropna()


# -----------------------------
# 4. Split Dataset
# -----------------------------

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 5. TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------
# 6. Train Model
# -----------------------------

model = LogisticRegression()

model.fit(X_train_tfidf, y_train)


# -----------------------------
# 7. Test Model
# -----------------------------

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["FAKE", "REAL"]
))


# -----------------------------
# 8. Save Model
# -----------------------------

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")



# 9. Save Test Results
# -----------------------------

with open("test_results.txt", "w") as file:
    file.write("Fake News Detector Test Results\n")
    file.write("--------------------------------\n")
    file.write(f"Accuracy: {accuracy:.2f}\n\n")
    file.write(
        classification_report(
            y_test,
            predictions,
            target_names=["FAKE", "REAL"]
        )
    )


print("\nModel saved as model.pkl")
print("Vectorizer saved as vectorizer.pkl")
print("Test results saved as test_results.txt")