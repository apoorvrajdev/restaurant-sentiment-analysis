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

from inference import (
    CHECKSUM_FILENAME,
    MODEL_FILENAME,
    VECTORIZER_FILENAME,
    preprocess,
    sha256_of,
)

RANDOM_STATE = 0
MAX_FEATURES = 1500
DATA_PATH = Path(__file__).resolve().parent / "data" / "Restaurant_Reviews.tsv"
ROOT = Path(__file__).resolve().parent


def write_checksums(directory: Path, filenames: list[str]) -> None:
    """Write a sha256sum-compatible manifest for the given artifacts."""
    lines = [f"{sha256_of(directory / name)}  {name}" for name in filenames]
    (directory / CHECKSUM_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    joblib.dump(model, ROOT / MODEL_FILENAME)
    joblib.dump(vectorizer, ROOT / VECTORIZER_FILENAME)
    write_checksums(ROOT, [MODEL_FILENAME, VECTORIZER_FILENAME])
    print(f"Saved {MODEL_FILENAME}, {VECTORIZER_FILENAME}, and {CHECKSUM_FILENAME}")


if __name__ == "__main__":
    main()
