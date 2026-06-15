import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

MAX_REVIEW_LENGTH = 2_000

_STEMMER = PorterStemmer()


class ReviewValidationError(ValueError):
    """Raised when a review cannot produce a reliable prediction."""


@lru_cache(maxsize=1)
def _stopword_set() -> frozenset[str]:
    """English stopwords minus 'not', which carries sentiment."""
    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("english")
    return frozenset(w for w in words if w != "not")


def preprocess(text: str) -> str:
    """Clean, stem, and de-stopword a review.

    This MUST match the pipeline used at training time so the live
    vectorizer sees tokens drawn from the same vocabulary it was fit on.
    """
    cleaned = re.sub("[^a-zA-Z]", " ", text).lower().split()
    stops = _stopword_set()
    stemmed = [_STEMMER.stem(word) for word in cleaned if word not in stops]
    return " ".join(stemmed)


def load_artifacts(model_directory: Path | None = None) -> tuple[Any, Any]:
    """Load the trusted model artifacts bundled with the application."""
    artifact_directory = model_directory or Path(__file__).resolve().parent
    model = joblib.load(artifact_directory / "model.pkl")
    vectorizer = joblib.load(artifact_directory / "vectorizer.pkl")
    return model, vectorizer


def predict_sentiment(review: str, model: Any, vectorizer: Any) -> str:
    """Validate a review and return its predicted sentiment label."""
    if not isinstance(review, str):
        raise ReviewValidationError("Review must be text.")

    normalized_review = review.strip()
    if not normalized_review:
        raise ReviewValidationError("Enter a review before requesting a prediction.")

    if len(normalized_review) > MAX_REVIEW_LENGTH:
        raise ReviewValidationError(
            f"Review must be {MAX_REVIEW_LENGTH:,} characters or fewer."
        )

    review_vector = vectorizer.transform([preprocess(normalized_review)])
    if review_vector.nnz == 0:
        raise ReviewValidationError(
            "The review does not contain enough recognized words for a reliable prediction."
        )

    prediction = model.predict(review_vector.toarray())
    return "Positive" if prediction[0] == 1 else "Negative"
