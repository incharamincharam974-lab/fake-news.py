import streamlit as st
import pandas as pd
import joblib
import re

# Load trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Page settings
st.set_page_config(
    page_title="Fake News Detector",
    page_icon=""
)

# Title
st.title("Fake News Detector")

st.write(
    "Enter a news headline below to check whether "
    "the trained machine-learning model classifies "
    "it as REAL or FAKE."
)


# -----------------------------
# Single News Prediction
# -----------------------------

st.header("🔍 Check a News Headline")

headline = st.text_input(
    "Enter a news headline:",
    placeholder="Example: Government announces a new scholarship"
)

if st.button("Check News"):

    if headline.strip() == "":
        st.warning("Please enter a headline.")

    else:
        cleaned = clean_text(headline)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        confidence = max(probability) * 100

        if prediction == 1:
            st.success("Prediction: REAL")
        else:
            st.error(" Prediction: FAKE")

        st.info(
            f"Model confidence: {confidence:.2f}%"
        )


# -----------------------------
# CSV Upload
# -----------------------------

st.header("📁 Batch Check CSV")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing a 'text' column",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(data)

    if "text" not in data.columns:

        st.error(
            "Your CSV must contain a column named 'text'."
        )

    else:

        cleaned = data["text"].apply(clean_text)

        vectors = vectorizer.transform(cleaned)

        predictions = model.predict(vectors)

        probabilities = model.predict_proba(vectors)

        data["prediction"] = [
            "REAL" if p == 1 else "FAKE"
            for p in predictions
        ]

        data["confidence"] = [
            round(max(prob) * 100, 2)
            for prob in probabilities
        ]

        st.subheader("Prediction Results")

        st.dataframe(data)

        csv = data.to_csv(index=False)

        st.download_button(
            "⬇Download Results",
            csv,
            "fake_news_results.csv",
            "text/csv"
        )