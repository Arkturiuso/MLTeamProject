from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pymorphy3

app = FastAPI(title="Sentiment Analysis API")

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

nltk.download('stopwords', quiet=True)
RU_STOPWORDS = set(stopwords.words('russian'))
MORPH = pymorphy3.MorphAnalyzer()


def preprocess_tweet(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|t\.co/\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^а-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    lemmatized = [
        MORPH.parse(w)[0].normal_form
        for w in words
        if w not in RU_STOPWORDS and len(w) > 2
    ]
    return ' '.join(lemmatized)


class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict_sentiment(input: TextInput):
    clean_text = preprocess_tweet(input.text)

    if not clean_text:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "probability_positive": 0.0,
            "probability_negative": 0.0,
            "clean_text": "",
            "message": "Текст после обработки пустой"
        }

    X = vectorizer.transform([clean_text])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    sentiment = "positive" if prediction == 1 else "negative"
    confidence = float(max(proba))

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "probability_positive": float(proba[1]),
        "probability_negative": float(proba[0]),
        "clean_text": clean_text
    }


@app.get("/")
def root():
    return {"message": "Sentiment Analysis API работает!"}