"""Train the restaurant sentiment model.

Reproducible replacement for the exploratory notebook. Uses the same
preprocess() pipeline as inference (see inference.py) so the deployed
vectorizer and the live request path share one vocabulary.

Usage:
    python train.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from inference import preprocess

RANDOM_STATE = 0
MAX_FEATURES = 1500
DATA_PATH = Path(__file__).resolve().parent / "data" / "Restaurant_Reviews.tsv"
ROOT = Path(__file__).resolve().parent


def main() -> None:
    dataset = pd.read_csv(DATA_PATH, delimiter="\t", quoting=3)
    corpus = [preprocess(review) for review in dataset["Review"]]

    vectorizer = CountVectorizer(max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

    joblib.dump(model, ROOT / "model.pkl")
    joblib.dump(vectorizer, ROOT / "vectorizer.pkl")
    print("Saved model.pkl and vectorizer.pkl")


if __name__ == "__main__":
    main()
