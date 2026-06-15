# Restaurant Sentiment Analysis

![CI](https://github.com/apoorvrajdev/restaurant-sentiment-analysis/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

End-to-end **NLP classification pipeline** that predicts customer sentiment from restaurant reviews. Built the full workflow from raw text preprocessing to deployed web application using classical machine learning.

**Key results:** 78% accuracy on a 1,000-review dataset, with a live Streamlit app serving real-time predictions.

---

## 🚀 Live Demo

[![Open App](https://img.shields.io/badge/Open%20Live%20App-Streamlit-red)](https://restaurant-sentiment-analyzer.streamlit.app)

Try the live sentiment analyzer here: **[restaurant-sentiment-analyzer.streamlit.app](https://restaurant-sentiment-analyzer.streamlit.app)**

Users can type a restaurant review and instantly receive a sentiment prediction.

---

## 📸 Application Showcase

Real inference outputs captured from the deployed Streamlit application. Each example below shows a live prediction the model produced on customer-style review text.

### Live Prediction Examples

<table>
  <tr>
    <td align="center" width="50%">
      <b>✅ Positive Review — Classified Correctly</b><br/>
      <sub>Input: a favorable restaurant review · Output: <code>Positive</code></sub><br/><br/>
      <img src="https://github.com/user-attachments/assets/18b117ef-e768-4394-b89c-dca3e7e4586a" alt="Positive Review Prediction in Streamlit App" width="100%"/>
    </td>
    <td align="center" width="50%">
      <b>❌ Negative Review — Classified Correctly</b><br/>
      <sub>Input: an unfavorable restaurant review · Output: <code>Negative</code></sub><br/><br/>
      <img src="https://github.com/user-attachments/assets/f3042f90-c90c-4b6f-b41e-0bc8bfe0bf0f" alt="Negative Review Prediction in Streamlit App" width="100%"/>
    </td>
  </tr>
</table>

> Screenshots above are real outputs from the live, deployed application — not mockups. They demonstrate that the trained model, vectorizer, and Streamlit interface are fully wired together end-to-end.

---

## 📊 Model Validation

### Confusion Matrix (Test Set)

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e195b02-707b-47db-8c04-bed48ac126a4" alt="Confusion Matrix — Restaurant Sentiment Model" width="60%"/>
</p>

<p align="center"><sub>Confusion matrix generated during model evaluation on the held-out test split.</sub></p>

**Model Accuracy: 78%**

| Actual ↓ / Predicted → | Negative | Positive |
|------------------------|---------:|---------:|
| **Negative**           | 75       | 22       |
| **Positive**           | 22       | 81       |

The model uses Multinomial Naive Bayes over Bag-of-Words features, with identical text preprocessing at training and inference time. It performs reasonably well for a small dataset and demonstrates a full NLP classification workflow.

---

## 📂 Dataset

The dataset contains **1000 restaurant reviews** labeled as **positive or negative**.

The goal of the dataset is to train a machine learning model capable of understanding customer sentiment from text reviews.

---

## 🛠️ Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Jupyter Notebook
- Streamlit

---

## 🔄 Project Workflow
The project follows a complete **machine learning pipeline**:
1. Data preprocessing and text cleaning
2. Tokenization and stopword removal
3. Feature extraction using **Bag of Words** (`CountVectorizer`)
4. Model training using a **Multinomial Naive Bayes classifier**
5. Model evaluation using **confusion matrix and accuracy**
6. Saving the trained model and vectorizer
7. Deploying the model as a **web application using Streamlit**

---

## 🗂️ Project Structure

```text
restaurant-sentiment-analysis/
├── app.py                # Streamlit web interface
├── inference.py          # Preprocessing, validation, and prediction
├── train.py              # Reproducible training script
├── requirements.txt
├── model.pkl             # Trained Multinomial Naive Bayes model
├── vectorizer.pkl        # Fitted CountVectorizer
├── data/
│   └── Restaurant_Reviews.tsv
├── tests/
├── docs/
│   └── ARCHITECTURE.md
└── .github/
    └── workflows/
        └── ci.yml
```

## Features

- Sentiment classification of restaurant reviews
- Streamlit-based interactive web interface
- Bag-of-Words feature extraction (`CountVectorizer`)
- Multinomial Naive Bayes machine learning model
- Prediction confidence score shown alongside each result
- Shared text preprocessing across training and inference
- Input validation and error handling
- Automated testing with pytest
- GitHub Actions CI workflow
- Reproducible local setup

## ⚙️ Installation

### Clone the repository

```
git clone https://github.com/apoorvrajdev/restaurant-sentiment-analysis.git
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run the application

```
streamlit run app.py
```

The app will open automatically in your browser.

### Retrain the model (optional)

The dataset lives in `data/Restaurant_Reviews.tsv`. To regenerate `model.pkl` and `vectorizer.pkl` from scratch using the same preprocessing as inference:

```
python train.py
```

### Run automated tests

```
python -m unittest discover -v
```

The inference layer validates empty, oversized, and unrecognized reviews before prediction. Model artifacts are loaded once per application process and the regression suite verifies known positive and negative predictions.

---
## Testing

Run the automated test suite:

```bash
pytest
```

The test suite validates:

- Prediction functionality
- Input validation
- Application stability

## Current Limitations

- Binary sentiment classification only
- Model trained on a limited review dataset
- English-language reviews only

---

## 🗺️ Roadmap

This project is being hardened from a working portfolio demo toward a more
production-oriented ML workflow. Progress is tracked below.

### ✅ Completed

| Area              | Improvement                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| Model correctness | Aligned inference with training by sharing a single `preprocess()` (clean → stem → stopword)       |
| Model quality     | Switched from Gaussian to **Multinomial Naive Bayes** (test accuracy 0.73 → 0.78)                  |
| Reproducibility   | Added a deterministic `train.py` (seeded) and versioned the training dataset                       |
| Feature           | Prediction **confidence score** surfaced in the UI, with prediction logging                        |
| Testing           | Expanded suite to 9 tests incl. regression cases for inflected words (93% coverage on `inference`) |
| Documentation     | Corrected feature-extraction description (Bag-of-Words, not TF-IDF) and refreshed metrics          |
| Tooling & CI      | Split runtime/dev dependencies, added `ruff` lint+format and coverage gates to CI                  |

### 🔜 Next Up

| Priority | Focus Area              | Planned Work                                                                                          |
| -------: | ----------------------- | ----------------------------------------------------------------------------------------------------- |
| **1**    | Reliability & Testing   | Test `app.py` and `load_artifacts()` failure paths (missing/corrupt artifacts)                        |
| **2**    | Model Integrity         | Add artifact checksums, validate model type on load, and publish a model card with version + metrics  |
| **3**    | Dependency Maintenance  | Bump `scikit-learn` off the pinned 1.2.2 and retrain artifacts against the new version                |

### 🔭 Future Direction

| Focus Area            | Ideas                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| Model Evaluation      | Benchmark alternative features/classifiers, hyperparameter tuning, metrics beyond accuracy           |
| Dataset Enhancement   | Larger and more diverse datasets, error analysis, class-imbalance handling                           |
| User-Focused Features | Batch sentiment analysis and visual summaries of sentiment trends                                    |
| Production Readiness  | Model monitoring/drift detection and a deployment-friendly prediction API                            |

---

## 👨‍💻 Author

**Apoorv Raj**

Machine Learning & NLP Engineer

---

## FAQ

### What machine learning model is used?

This project uses a Multinomial Naive Bayes classifier trained on restaurant review text transformed into Bag-of-Words features with `CountVectorizer`.

### Can I analyze my own restaurant reviews?

Yes. Enter any restaurant review into the Streamlit interface and the application will predict whether the sentiment is positive or negative.

### How is the text processed?

The review text is cleaned, lowercased, stemmed, and stripped of stopwords (the same `preprocess()` step used at training time), then transformed by the saved `CountVectorizer` before being passed to the trained model.

### Is the model suitable for production use?

The project includes input validation, automated testing, and CI workflows. It is primarily intended as a portfolio-quality machine learning project and educational reference.

### How do I run the application locally?

Install dependencies from `requirements.txt` and start the Streamlit app using:

```bash
streamlit run app.py
```

⭐ If you found this project useful, consider giving it a **star on GitHub**.
