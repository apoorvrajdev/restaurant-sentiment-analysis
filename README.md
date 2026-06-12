![CI](https://github.com/apoorvrajdev/restaurant-sentiment-analysis/actions/workflows/ci.yml/badge.svg)
# Restaurant Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

End-to-end **NLP classification pipeline** that predicts customer sentiment from restaurant reviews. Built the full workflow from raw text preprocessing to deployed web application using classical machine learning.

**Key results:** 73% accuracy on a 1,000-review dataset, with a live Streamlit app serving real-time predictions.

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

**Model Accuracy: 73%**

| Actual ↓ / Predicted → | Negative | Positive |
|------------------------|---------:|---------:|
| **Negative**           | 55       | 42       |
| **Positive**           | 12       | 91       |

The model performs reasonably well for a small dataset and demonstrates a full NLP classification workflow.

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
3. Feature extraction using **Bag of Words**
4. Model training using **Naive Bayes classifier**
5. Model evaluation using **confusion matrix and accuracy**
6. Saving the trained model
7. Deploying the model as a **web application using Streamlit**

---

## 🗂️ Project Structure

## Project Structure

```text
restaurant-sentiment-analysis/
├── app.py
├── inference.py
├── requirements.txt
├── model.pkl
├── vectorizer.pkl
├── tests/
├── docs/
│   └── ARCHITECTURE.md
└── .github/
    └── workflows/
        └── ci.yml
```

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

### Run automated tests

```
python -m unittest discover -v
```

The inference layer validates empty, oversized, and unrecognized reviews before prediction. Model artifacts are loaded once per application process and the regression suite verifies known positive and negative predictions.

---

## Future Work

The next development phase will focus on improving model performance, strengthening the inference pipeline, and making the application ready for broader real-world use.

### Future Development

| Phase | Focus Area            | Planned Improvements                                                                                                                                                      | Impact                                                    |
| ----: | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **1** | Model Evaluation      | Benchmark additional feature extraction techniques and classification algorithms, perform systematic hyperparameter tuning, and expand evaluation metrics beyond accuracy | Improved prediction quality and stronger model validation |
| **2** | Dataset Enhancement   | Incorporate larger and more diverse review datasets, analyze classification errors, and address class imbalance where applicable                                          | Better generalization across real-world customer feedback |
| **3** | Reliability & Testing | Expand automated test coverage, strengthen input validation, improve exception handling, and enhance reproducibility of the training pipeline                             | Increased application robustness and maintainability      |
| **4** | User-Focused Features | Add prediction confidence scores, support batch sentiment analysis, and provide visual summaries of sentiment trends                                                      | More actionable insights for end users                    |
| **5** | Production Readiness  | Introduce model versioning, CI/CD automation, monitoring, and deployment-friendly APIs                                                                                    | A more scalable and production-oriented ML workflow       |

---

## 👨‍💻 Author

**Apoorv Raj**

Machine Learning & NLP Engineer

---

⭐ If you found this project useful, consider giving it a **star on GitHub**.
