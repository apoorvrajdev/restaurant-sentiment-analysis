<h1 align="center">Restaurant Review Sentiment Analyzer</h1>

<p align="center">
  <strong>An end-to-end NLP pipeline that predicts customer sentiment from restaurant reviews — from raw text to a deployed, tested web app.</strong>
</p>

<p align="center">
  <a href="https://github.com/apoorvrajdev/restaurant-sentiment-analysis/actions/workflows/ci.yml"><img src="https://github.com/apoorvrajdev/restaurant-sentiment-analysis/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/scikit--learn-MultinomialNB-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Streamlit-deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/tests-17%20passing-success?style=flat-square" alt="Tests" />
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License: MIT" />
</p>

<p align="center">
  A compact, production-minded machine-learning project: classical NLP done correctly, with the engineering discipline — shared preprocessing, reproducible training, artifact integrity checks, tests, and CI — that keeps a model honest after it leaves the notebook.
</p>

---

## 🚀 Live Demo

**[restaurant-sentiment-analyzer.streamlit.app](https://restaurant-sentiment-analyzer.streamlit.app)**

Type any restaurant review and get an instant Positive/Negative prediction with a confidence score.

<table>
  <tr>
    <td align="center" width="50%">
      <b>✅ Positive Review</b><br/>
      <sub>A favorable review → <code>Positive</code></sub><br/><br/>
      <img src="https://github.com/user-attachments/assets/18b117ef-e768-4394-b89c-dca3e7e4586a" alt="Positive Review Prediction in Streamlit App" width="100%"/>
    </td>
    <td align="center" width="50%">
      <b>❌ Negative Review</b><br/>
      <sub>An unfavorable review → <code>Negative</code></sub><br/><br/>
      <img src="https://github.com/user-attachments/assets/f3042f90-c90c-4b6f-b41e-0bc8bfe0bf0f" alt="Negative Review Prediction in Streamlit App" width="100%"/>
    </td>
  </tr>
</table>

<sub>Screenshots are real outputs from the deployed app — the trained model, vectorizer, and Streamlit UI wired together end to end.</sub>

---

## 📌 What Is This Project?

A binary sentiment classifier for restaurant reviews, served through a small Streamlit web app. A review goes in; cleaned text is vectorized, scored by a Naive Bayes model, and returned as **Positive** or **Negative** with a confidence percentage.

It is intentionally **not** a deep-learning showcase or a large system. It is a focused demonstration that the *unglamorous* parts of an ML project are done right: the same text preprocessing runs at training and inference time, training is reproducible from a script and a seed, the serialized artifacts are integrity-checked before they are trusted, and the whole thing is covered by tests that run in CI on every push.

I built it as a portfolio reference for the difference between "a model in a notebook" and "a model you can actually ship."

---

## 💡 What This Project Demonstrates

- **Train/inference parity** — a single `preprocess()` function is shared by the training script and the live request path, eliminating the train/serve skew that silently degrades accuracy.
- **Reproducible training** — `train.py` rebuilds the model and vectorizer deterministically (`random_state=0`) from a version-controlled dataset.
- **Artifact integrity** — `load_artifacts()` verifies a `sha256sum`-compatible checksum manifest and validates object types before serving predictions (defense-in-depth for pickle loading).
- **Honest evaluation** — accuracy and a confusion matrix measured on a held-out test split, documented in a model card.
- **Robust inference** — typed input validation with clear, user-facing error messages for empty, oversized, and unrecognized input.
- **Explainable output** — every prediction carries a confidence score from `predict_proba`.
- **Tested + automated** — 17 tests (inference, artifact-loading failure paths, and the Streamlit UI via `AppTest`) gated by `ruff` lint/format and coverage in GitHub Actions.

---

## 🧠 How It Works

The project has two phases that share one preprocessing function, which is the whole point.

```
TRAINING  (train.py)                         INFERENCE  (app.py + inference.py)
───────────────────                          ─────────────────────────────────
Restaurant_Reviews.tsv                       user review (Streamlit text area)
        │                                              │
        ▼                                              ▼
   preprocess()  ◀────── same function ──────▶   preprocess()
   clean · lower · stem · stopwords             clean · lower · stem · stopwords
        │                                              │
        ▼                                              ▼
   CountVectorizer.fit                          CountVectorizer.transform
        │                                              │
        ▼                                              ▼
   MultinomialNB.fit                            MultinomialNB.predict_proba
        │                                              │
        ▼                                              ▼
   model.pkl + vectorizer.pkl  ──────────────▶  label + confidence
   + artifacts.sha256                           (validated + integrity-checked)
```

**Preprocessing (shared):** strip non-alphabetic characters → lowercase → remove English stopwords (keeping `not`, which carries sentiment) → Porter-stem each token.

**Decision:** the predicted label is the higher-probability class; its probability is reported as the confidence score.

---

## 📊 Model Performance

Measured on a 20% held-out test split (`random_state=0`). Reproduce with `python train.py`.

| Metric   | Value      |
| -------- | ---------- |
| Accuracy | **0.78**   |
| Model    | Multinomial Naive Bayes |
| Features | Bag-of-Words (`CountVectorizer`, `max_features=1500`) |

Confusion matrix (rows = actual, columns = predicted):

| Actual ↓ / Predicted → | Negative | Positive |
| ---------------------- | -------: | -------: |
| **Negative**           | 75       | 22       |
| **Positive**           | 22       | 81       |

Full details — preprocessing, training data, intended use, and limitations — live in the [model card](docs/MODEL_CARD.md).

---

## 🛠️ Tech Stack

| Layer          | Technologies                                            |
| -------------- | ------------------------------------------------------- |
| **Language**   | Python 3.10+                                            |
| **ML / NLP**   | scikit-learn (MultinomialNB, CountVectorizer), NLTK, pandas, numpy, joblib |
| **Web app**    | Streamlit                                               |
| **Quality**    | pytest, pytest-cov, ruff (lint + format)                |
| **CI**         | GitHub Actions                                          |

---

## 📁 Repository Structure

```
restaurant-sentiment-analysis/
├── app.py                  # Streamlit UI — thin: renders, calls inference
├── inference.py            # preprocess(), validation, integrity-checked loading, prediction
├── train.py                # Reproducible training; writes artifacts + checksum manifest
├── model.pkl               # Trained Multinomial Naive Bayes model
├── vectorizer.pkl          # Fitted CountVectorizer
├── artifacts.sha256        # sha256sum-compatible manifest for the artifacts
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Dev/test/lint dependencies
├── pyproject.toml          # ruff + pytest/coverage config
├── data/
│   └── Restaurant_Reviews.tsv   # 1,000 labeled reviews
├── tests/
│   ├── test_inference.py   # Prediction + validation + preprocessing parity
│   ├── test_artifacts.py   # load_artifacts() failure + integrity paths
│   └── test_app.py         # Streamlit UI via AppTest
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MODEL_CARD.md
│   └── DEPLOYMENT.md
└── .github/workflows/ci.yml
```

---

## 🏁 Quick Start

### Prerequisites

- Python **3.10+**
- Git

### Install & run

```bash
git clone https://github.com/apoorvrajdev/restaurant-sentiment-analysis.git
cd restaurant-sentiment-analysis
pip install -r requirements.txt
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### Reproduce training

The dataset lives in `data/Restaurant_Reviews.tsv`. Regenerate the model, vectorizer, and checksum manifest:

```bash
python train.py
```

### Run the test suite

```bash
pip install -r requirements-dev.txt
pytest                 # 17 tests, with coverage
ruff check .           # lint
ruff format --check .  # formatting
```

---

## 🔧 Engineering Decisions

> **Why share one `preprocess()` between training and inference?**
> Train/serve skew is the bug that quietly destroys classical-NLP accuracy: if training stems and de-stopwords text but inference doesn't, the live request sends tokens the vectorizer's vocabulary never saw, and they're silently dropped. The only durable fix is a single code path. `train.py` and `inference.py` import the same `preprocess()`, so the text entering the vectorizer at inference is processed byte-for-byte like the training corpus.

> **Why Multinomial Naive Bayes and not Gaussian?**
> The features are word counts from a Bag-of-Words vectorizer — discrete, sparse, non-negative. `MultinomialNB` is built for exactly that distribution; `GaussianNB` assumes continuous, normally-distributed features and is a poor fit for count data. Switching to it lifted held-out accuracy.

> **Why Bag-of-Words instead of a transformer?**
> On 1,000 reviews, a large language model would be overkill, slower to serve, and far harder to explain. A linear-time, fully interpretable classical pipeline is the honest engineering choice at this scale — and it keeps the deployed app small enough to run free on Streamlit Cloud.

> **Why checksum and type-validate the model artifacts?**
> `joblib`/pickle loading executes arbitrary code, so artifacts are a supply-chain surface. `load_artifacts()` verifies a committed `sha256` manifest and confirms the loaded objects actually expose `predict_proba`/`transform` before serving — so a corrupted or swapped artifact fails loudly instead of producing silent garbage.

---

## 🗺️ Roadmap

### ✅ Completed

- [x] Streamlit app with live Positive/Negative prediction
- [x] Shared `preprocess()` across training and inference (train/serve parity)
- [x] Switched Gaussian → **Multinomial Naive Bayes** (accuracy 0.73 → 0.78)
- [x] Reproducible `train.py` + version-controlled dataset
- [x] Prediction **confidence score** in the UI, with prediction logging
- [x] Artifact integrity: `sha256` manifest + object-type validation on load
- [x] Model card documenting data, metrics, and limitations
- [x] Test suite (17 tests) covering inference, artifact failures, and the UI
- [x] CI quality gates: `ruff` lint/format + pytest coverage
- [x] Upgraded `scikit-learn` (1.2.2 → 1.7.x) and retrained artifacts

### 🔜 Next Up

- [ ] Expand the dataset and analyze misclassifications

### 🔭 Future Direction

- [ ] Neutral / mixed sentiment class beyond binary
- [ ] Batch analysis and sentiment-trend visualizations
- [ ] Benchmark alternative features/classifiers and tune hyperparameters

---

## ⚠️ Limitations

- Binary sentiment only — no neutral or mixed class.
- English-language reviews only.
- Trained on a small, single-domain dataset; may not generalize to other corpora.
- Bag-of-Words ignores word order and most negation beyond the retained `not` token.

---

## 📄 License & Contact

Released under the [MIT License](LICENSE).

**Built by [apoorvrajdev](https://github.com/apoorvrajdev)** — reach me at [apoorvrajmgr@gmail.com](mailto:apoorvrajmgr@gmail.com).

<p align="center">
  <em>A portfolio project demonstrating production-minded machine-learning engineering.</em>
</p>
