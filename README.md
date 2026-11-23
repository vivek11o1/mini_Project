# News Topic Classification (FastAPI)

A small, classic-ML project that classifies news articles into six topics
(`politics`, `technology`, `entertainment`, `health`, `business`, `sports`) using
**TF-IDF + Support Vector Machine (SVM)** from scikit-learn.

The model is served through **FastAPI**. Interactive Swagger documentation lives at
`http://localhost:8000/docs` — no frontend needed.

## Project structure

| File | Purpose |
|------|---------|
| `ML_Model.py` | Trains/evaluates the SVM and saves `svm_model.pkl` + `svm_vectorizer.pkl` |
| `app.py` | FastAPI service (endpoints + Swagger UI) |
| `scraper.py` | Optional scraper for **unlabeled** articles |
| `data/sample_news.csv` | Small labeled dataset bundled so the project runs immediately |
| `test_api.py` | Smoke test for the API |
| `news.csv` | Optional: your own labeled CSV (`Summary`, `Topic` columns) |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python ML_Model.py train                   # uses news.csv if present, else data/sample_news.csv
python ML_Model.py train --data my_news.csv
python ML_Model.py test                    # evaluate the saved model
```

Training writes:

- `svm_model.pkl` — full scikit-learn `Pipeline` (TF-IDF + SVC)
- `svm_vectorizer.pkl` — the fitted TF-IDF vectorizer (kept for compatibility)
- `ml_results.csv` — per-row actual vs predicted labels
- `classification_report.txt` — precision/recall/F1 per topic

If you add your own `news.csv`, give it at least these columns:

```csv
Summary,Topic
"government passes new law today",politics
```

## Run the API

```bash
uvicorn app:app --reload
```

If no model artifact exists, the app trains one from `data/sample_news.csv` on startup.

Open **http://localhost:8000/docs** for Swagger UI.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check + loaded topics |
| GET | `/topics` | List of topic labels |
| POST | `/predict` | Classify one article |
| POST | `/predict/batch` | Classify several articles |

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "The central bank lowered interest rates after inflation slowed."}'
```

Example response:

```json
{
  "topic": "business",
  "confidence": 0.87,
  "probabilities": {
    "business": 0.87,
    "politics": 0.05,
    "technology": 0.03,
    "health": 0.02,
    "sports": 0.02,
    "entertainment": 0.01
  }
}
```

## Test

```bash
python test_api.py
```

## Docker

```bash
docker build -t news-classifier .
docker run -p 8000:8000 news-classifier
# Swagger UI: http://localhost:8000/docs
```

## Notes

- `scraper.py` collects unlabeled headlines into `scraped_articles.csv`. Add a `Topic`
  column before using that data for training.
- `DL_Model.py` is an unused placeholder from the original repository; the working
  model is the SVM in `ML_Model.py`.
