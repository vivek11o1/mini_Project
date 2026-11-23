"""FastAPI service for news-topic classification (Swagger UI at /docs).

Run:
    uvicorn app:app --reload
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
# Make the local training module importable even when uvicorn is started elsewhere.
import sys  # noqa: E402

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ML_Model import MODEL_PATH, train_and_save  # noqa: E402


def _load_or_train():
    """Load the saved pipeline, training it from bundled data if it is missing."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    print("No saved model found - training from bundled sample data...")
    train_and_save()
    return joblib.load(MODEL_PATH)


model = _load_or_train()
TOPICS = [str(label) for label in model.classes_]

app = FastAPI(
    title="News Topic Classification API",
    description="Classify a news article into one of six topics using TF-IDF + SVM.",
    version="1.0.0",
)


class ArticleRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw news article text or summary")


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of article texts")


class Prediction(BaseModel):
    topic: str
    confidence: float
    probabilities: Dict[str, float]


class BatchResponse(BaseModel):
    predictions: List[Prediction]


def _predict(text: str) -> Prediction:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Article text must not be empty")
    probabilities = model.predict_proba([text])[0]
    scores = {str(label): float(score) for label, score in zip(model.classes_, probabilities)}
    topic = max(scores, key=scores.get)
    return Prediction(topic=topic, confidence=scores[topic], probabilities=scores)


@app.get("/", tags=["info"], summary="API information")
def root():
    return {
        "name": "News Topic Classification API",
        "model": "TF-IDF + SVM",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["info"], summary="Health check")
def health():
    return {"status": "ok", "model_loaded": True, "topics": TOPICS}


@app.get("/topics", tags=["info"], summary="List available topic labels")
def topics():
    return {"topics": TOPICS}


@app.post("/predict", response_model=Prediction, tags=["prediction"], summary="Classify one article")
def predict(article: ArticleRequest):
    """Return the predicted topic, its confidence, and all class probabilities."""
    return _predict(article.text)


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["prediction"],
    summary="Classify multiple articles",
)
def predict_batch(batch: BatchRequest):
    """Classify a list of articles in one request."""
    return BatchResponse(predictions=[_predict(text) for text in batch.texts])
