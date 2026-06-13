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

## Data Flow

1. User enters a restaurant review through the Streamlit interface.
2. Input validation checks the review content.
3. The saved TF-IDF vectorizer converts text into numerical features.
4. The trained Naive Bayes model generates a sentiment prediction.
5. The prediction result is displayed to the user.

---

## Model Artifacts

| File | Purpose |
|--------|---------|
| model.pkl | Stores the trained Naive Bayes classifier |
| vectorizer.pkl | Stores the fitted TF-IDF vectorizer |

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