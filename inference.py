import hashlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

MAX_REVIEW_LENGTH = 2_000

MODEL_FILENAME = "model.pkl"
VECTORIZER_FILENAME = "vectorizer.pkl"
CHECKSUM_FILENAME = "artifacts.sha256"

_STEMMER = PorterStemmer()

logger = logging.getLogger(__name__)


class ReviewValidationError(ValueError):
    """Raised when a review cannot produce a reliable prediction."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when model artifacts fail integrity or type validation."""


@dataclass(frozen=True)
class Prediction:
    """A sentiment prediction and the model's confidence in it."""

    label: str
    confidence: float  # probability of the predicted label, in [0, 1]


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


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_checksums(directory: Path) -> None:
    """Verify artifacts against the checksum manifest, if one is present.

    The manifest uses the standard ``<hex-digest>  <filename>`` format so it
    is also checkable with ``sha256sum -c``. Missing manifest -> skip (older
    deployments stay compatible); present but mismatched -> hard failure.
    """
    manifest = directory / CHECKSUM_FILENAME
    if not manifest.exists():
        return
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected_digest, _, filename = line.partition("  ")
        actual_digest = sha256_of(directory / filename.strip())
        if actual_digest != expected_digest.strip():
            raise ArtifactIntegrityError(
                f"Checksum mismatch for {filename.strip()}; artifacts may be corrupt or tampered."
            )


def _validate_artifacts(model: Any, vectorizer: Any) -> None:
    """Confirm the loaded objects are a usable model/vectorizer pair."""
    if not callable(getattr(model, "predict_proba", None)):
        raise ArtifactIntegrityError("Loaded model does not support predict_proba.")
    if not callable(getattr(vectorizer, "transform", None)):
        raise ArtifactIntegrityError("Loaded vectorizer does not support transform.")


def load_artifacts(model_directory: Path | None = None) -> tuple[Any, Any]:
    """Load and integrity-check the trusted model artifacts."""
    artifact_directory = model_directory or Path(__file__).resolve().parent
    _verify_checksums(artifact_directory)
    model = joblib.load(artifact_directory / MODEL_FILENAME)
    vectorizer = joblib.load(artifact_directory / VECTORIZER_FILENAME)
    _validate_artifacts(model, vectorizer)
    return model, vectorizer


def predict_sentiment(review: str, model: Any, vectorizer: Any) -> Prediction:
    """Validate a review and return its predicted sentiment and confidence."""
    if not isinstance(review, str):
        raise ReviewValidationError("Review must be text.")

    normalized_review = review.strip()
    if not normalized_review:
        raise ReviewValidationError("Enter a review before requesting a prediction.")

    if len(normalized_review) > MAX_REVIEW_LENGTH:
        raise ReviewValidationError(f"Review must be {MAX_REVIEW_LENGTH:,} characters or fewer.")

    review_vector = vectorizer.transform([preprocess(normalized_review)])
    if review_vector.nnz == 0:
        raise ReviewValidationError(
            "The review does not contain enough recognized words for a reliable prediction."
        )

    probabilities = model.predict_proba(review_vector.toarray())[0]
    best_index = int(probabilities.argmax())
    predicted_class = model.classes_[best_index]
    label = "Positive" if predicted_class == 1 else "Negative"
    confidence = float(probabilities[best_index])

    logger.info(
        "Sentiment prediction: label=%s confidence=%.3f review_length=%d",
        label,
        confidence,
        len(normalized_review),
    )
    return Prediction(label=label, confidence=confidence)
