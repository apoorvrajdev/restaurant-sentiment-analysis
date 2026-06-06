from pathlib import Path
from typing import Any

import joblib

MAX_REVIEW_LENGTH = 2_000


class ReviewValidationError(ValueError):
    """Raised when a review cannot produce a reliable prediction."""


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

    review_vector = vectorizer.transform([normalized_review])
    if review_vector.nnz == 0:
        raise ReviewValidationError(
            "The review does not contain enough recognized words for a reliable prediction."
        )

    prediction = model.predict(review_vector.toarray())
    return "Positive" if prediction[0] == 1 else "Negative"
