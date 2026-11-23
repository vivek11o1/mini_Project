"""Train and evaluate the SVM news-topic classifier.

Usage:
    python ML_Model.py train                      # train + save artifacts
    python ML_Model.py test                       # evaluate saved artifacts
    python ML_Model.py train --data my_news.csv   # use a custom labeled CSV
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# sklearn 1.9+ deprecates SVC(probability=True); keep the demo output clean.
warnings.filterwarnings(
    "ignore", message="The `probability` parameter was deprecated.*", category=FutureWarning
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "svm_vectorizer.pkl"
RESULTS_PATH = BASE_DIR / "ml_results.csv"
REPORT_PATH = BASE_DIR / "classification_report.txt"

# A labeled CSV is expected to have at least these two columns:
#   Summary  -> the article text
#   Topic    -> one of politics / technology / entertainment / health / business / sports
DATA_CANDIDATES = [BASE_DIR / "news.csv", BASE_DIR / "data" / "sample_news.csv"]


def find_dataset(path: str | None = None) -> Path:
    """Return the first available labeled CSV."""
    if path:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset not found: {candidate}")
        return candidate

    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No labeled dataset found. Add 'news.csv' with Summary and Topic columns "
        f"or keep the bundled 'data/sample_news.csv'. Looked for: {DATA_CANDIDATES}"
    )


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Load and validate a labeled news CSV."""
    data = pd.read_csv(find_dataset(path))
    required = {"Summary", "Topic"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns {sorted(required)}; missing {sorted(missing)}")

    data = data.dropna(subset=["Summary", "Topic"]).copy()
    data["Summary"] = data["Summary"].astype(str).str.strip()
    data["Topic"] = data["Topic"].astype(str).str.strip()
    data = data[(data["Summary"] != "") & (data["Topic"] != "")]

    if len(data) < 8:
        raise ValueError("Need at least 8 labeled rows to train a model.")
    if data["Topic"].nunique() < 2:
        raise ValueError("Need at least two different topics to train a classifier.")
    return data


def build_pipeline() -> Pipeline:
    """TF-IDF vectorizer + SVM classifier in one pickleable object."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", sublinear_tf=True)),
            ("svm", svm.SVC(probability=True, random_state=42)),
        ]
    )


def train_and_save(path: str | None = None) -> dict:
    """Train the SVM with a small grid search, evaluate it, and persist artifacts."""
    data = load_dataset(path)
    csv_path = find_dataset(path)
    print(f"Training on {len(data)} rows from {csv_path}")

    # Stratify only when every class has at least 2 examples.
    stratify = data["Topic"] if data["Topic"].value_counts().min() >= 2 else None
    train_data, test_data, train_labels, test_labels = train_test_split(
        data["Summary"],
        data["Topic"],
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    pipeline = build_pipeline()
    parameters = {"svm__kernel": ["linear", "rbf"], "svm__C": [0.1, 1, 10]}
    search = GridSearchCV(pipeline, parameters, cv=3, scoring="accuracy")
    search.fit(train_data, train_labels)

    # Honest hold-out evaluation with the best parameters found on the training split.
    best_estimator = search.best_estimator_
    predictions = best_estimator.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, zero_division=0)

    # Refit on every labeled row for the artifact that gets deployed.
    model = build_pipeline()
    model.set_params(**search.best_params_)
    model.fit(data["Summary"], data["Topic"])

    REPORT_PATH.write_text(report, encoding="utf-8")
    pd.DataFrame({"Actual": test_labels, "Predicted": predictions}).to_csv(
        RESULTS_PATH, index=False
    )
    joblib.dump(model, MODEL_PATH)
    joblib.dump(model.named_steps["tfidf"], VECTORIZER_PATH)

    print(f"Best parameters: {search.best_params_}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(report)
    print(f"Saved {MODEL_PATH.name}, {VECTORIZER_PATH.name}, {RESULTS_PATH.name}")
    return {"model": model, "accuracy": accuracy, "best_params": search.best_params_}


def load_model():
    """Load the trained pipeline, or return None when it has not been trained yet."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def evaluate(path: str | None = None) -> str:
    """Evaluate the already-trained model on the full labeled dataset."""
    model = load_model()
    if model is None:
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run 'python ML_Model.py train' first.")

    data = load_dataset(path)
    report = classification_report(data["Topic"], model.predict(data["Summary"]), zero_division=0)
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["train", "test"], default="train")
    parser.add_argument("--data", default=None, help="Path to a labeled CSV with Summary and Topic")
    args = parser.parse_args()

    if args.command == "train":
        train_and_save(args.data)
    else:
        evaluate(args.data)


if __name__ == "__main__":
    main()
