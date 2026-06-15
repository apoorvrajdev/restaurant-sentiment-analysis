# Architecture

## Training Pipeline (`train.py`)

1. Load review dataset (`data/Restaurant_Reviews.tsv`)
2. Clean and preprocess text via the shared `preprocess()` (clean → lowercase → stem → stopword removal)
3. Generate TF-IDF features (`TfidfVectorizer`, `max_features=1500`)
4. Train a Logistic Regression classifier
5. Save model artifacts and a `sha256` checksum manifest

## Inference Pipeline (`app.py` + `inference.py`)

1. User enters review
2. Input validation (`ReviewValidationError`)
3. Text preprocessing with the **same** `preprocess()` used in training
4. Text vectorization
5. Model prediction with a confidence score (`predict_proba`)
6. Sentiment + confidence returned to user

## Components

- `app.py` — Streamlit user interface
- `inference.py` — `preprocess()`, validation, integrity-checked loading, prediction
- `train.py` — Reproducible training pipeline
- `model.pkl` — Trained Logistic Regression model
- `vectorizer.pkl` — Fitted TfidfVectorizer
- `artifacts.sha256` — Checksum manifest verified on load
- `tests/` — Automated test suite
- `.github/workflows/ci.yml` — Continuous Integration workflow

## Data Flow

1. User enters a restaurant review through the Streamlit interface.
2. Input validation checks the review content.
3. The shared `preprocess()` cleans and stems the text, then the saved `TfidfVectorizer` converts it into numerical features.
4. The trained Logistic Regression model generates a sentiment prediction and confidence.
5. The prediction result is displayed to the user.

---

## Model Artifacts

| File | Purpose |
|--------|---------|
| model.pkl | Stores the trained Logistic Regression classifier |
| vectorizer.pkl | Stores the fitted TfidfVectorizer |
| artifacts.sha256 | Checksum manifest verified by `load_artifacts()` |

---

## Deployment Overview

The application can be deployed using Streamlit Cloud or a containerized environment.

Deployment workflow:

1. Clone repository
2. Install dependencies
3. Launch Streamlit application
4. Access through browser

Example:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Continuous Integration

GitHub Actions automatically:

- Installs project dependencies
- Sets up Python 3.11
- Executes automated tests
- Verifies build stability on pushes and pull requests