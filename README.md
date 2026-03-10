# Restaurant Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

This project performs **sentiment analysis on restaurant reviews** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The model classifies customer reviews as **positive** or **negative** to better understand customer feedback and automate sentiment detection.
---
# 🚀 Live Demo
[![Open App](https://img.shields.io/badge/Open%20Live%20App-Streamlit-red)](https://restaurant-sentiment-analyzer.streamlit.app)
OR Try the AI sentiment analyzer here:

https://restaurant-sentiment-analyzer.streamlit.app

Users can type a restaurant review and instantly receive a sentiment prediction.
---
# Demo

### Positive Review Prediction
![Positive Review](https://github.com/user-attachments/assets/18b117ef-e768-4394-b89c-dca3e7e4586a)

### Negative Review Prediction
![Negative Review](https://github.com/user-attachments/assets/f3042a7f-90c2-4b6f-b41e-0bc8bfe0bf0f)

### Confusion Matrix
![Confusion Matrix](https://github.com/user-attachments/assets/6e195b02-707b-47db-8c04-bed48ac126a4)

---
# Dataset

The dataset contains **1000 restaurant reviews** labeled as **positive or negative**.

The goal of the dataset is to train a machine learning model capable of understanding customer sentiment from text reviews.

---
# Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Jupyter Notebook
- Streamlit
---
# Project Workflow
The project follows a complete **machine learning pipeline**:
1. Data preprocessing and text cleaning
2. Tokenization and stopword removal
3. Feature extraction using **Bag of Words**
4. Model training using **Naive Bayes classifier**
5. Model evaluation using **confusion matrix and accuracy**
6. Saving the trained model
7. Deploying the model as a **web application using Streamlit**

---
## Model Performance

Model Accuracy: **73%**

### Confusion Matrix

| Actual / Predicted | Negative | Positive |
|--------------------|---------|---------|
| Negative           | 55      | 42      |
| Positive           | 12      | 91      |

The model performs reasonably well for a small dataset and demonstrates a full NLP classification workflow.
---
## Project Structure

```
restaurant-sentiment-analysis
│
├── Restaurant_sentiment_analysis.ipynb   # Main notebook with model training
├── app.py                                # Streamlit web application
├── model.pkl                             # Trained machine learning model
├── vectorizer.pkl                        # Text vectorizer
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```
## Installation

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
---
# Future Improvements
- Improve model accuracy using **deep learning models (LSTM / Transformers)**
- Train the model on a **larger dataset**
- Add **model explainability for word importance**
- Deploy using **Docker and cloud infrastructure**

---
# Author
Apoorv Raj  
Machine Learning & AI Enthusiast
