# Architecture

## Training Pipeline

1. Load review dataset
2. Clean and preprocess text
3. Generate TF-IDF features
4. Train Naive Bayes classifier
5. Save model artifacts

## Inference Pipeline

1. User enters review
2. Input validation
3. Text vectorization
4. Model prediction
5. Sentiment returned to user

## Components

- `app.py` — Streamlit user interface
- `inference.py` — Prediction logic
- `model.pkl` — Trained Naive Bayes model
- `vectorizer.pkl` — TF-IDF vectorizer
- `tests/` — Automated test suite
- `.github/workflows/ci.yml` — Continuous Integration workflow