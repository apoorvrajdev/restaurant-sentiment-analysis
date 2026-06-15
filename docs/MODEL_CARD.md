# Model Card — Restaurant Review Sentiment Classifier

## Overview

| Field | Value |
| ----- | ----- |
| Task | Binary sentiment classification of restaurant reviews |
| Model | Logistic Regression (`sklearn.linear_model.LogisticRegression`) |
| Features | TF-IDF (`TfidfVectorizer`, `max_features=1500`) |
| Selected by | Benchmarking — see [EVALUATION.md](EVALUATION.md) |
| Artifacts | `model.pkl`, `vectorizer.pkl` |
| Last trained | 2026-06-15 |
| Reproduce with | `python train.py` |

## Input & Preprocessing

The classifier consumes free-text reviews. Both training and inference apply the
same `preprocess()` function in `inference.py`:

1. Strip non-alphabetic characters and lowercase.
2. Tokenize on whitespace.
3. Remove English stopwords **except** `not` (it carries sentiment).
4. Porter-stem each remaining token.

Keeping training and inference preprocessing identical is required — divergence
silently degrades accuracy by sending out-of-vocabulary tokens to the model.

## Training Data

- Source: `data/Restaurant_Reviews.tsv`
- Size: 1,000 labeled reviews (binary: 1 = positive, 0 = negative)
- Split: 80% train / 20% test, stratified, `random_state=0` (vectorizer fit on the training split only)

## Evaluation

Measured on the 20% held-out test set:

| Metric | Value |
| ------ | ----- |
| Accuracy | 0.82 |

Confusion matrix (rows = actual, columns = predicted):

| Actual ↓ / Predicted → | Negative | Positive |
| ---------------------- | -------: | -------: |
| **Negative**           | 81       | 19       |
| **Positive**           | 17       | 83       |

Each prediction also returns a confidence score (the predicted class probability
from `predict_proba`).

## Intended Use

Educational / portfolio demonstration of an end-to-end NLP classification pipeline.
Not intended for high-stakes or production decision-making without further validation.

## Limitations

- Binary only — no neutral or mixed sentiment.
- English-language reviews only.
- Trained on a small, single-domain dataset; may not generalize to other corpora.
- TF-IDF ignores word order and most negation beyond the retained `not` token.

## Integrity & Versioning

- `artifacts.sha256` is a `sha256sum`-compatible manifest of the bundled artifacts.
- `load_artifacts()` verifies these checksums (when the manifest is present) and
  validates the loaded object types before serving predictions.
- Regenerating the model via `train.py` refreshes both the artifacts and the manifest.
