import logging

import streamlit as st

from inference import (
    MAX_REVIEW_LENGTH,
    ReviewValidationError,
    load_artifacts,
    predict_sentiment,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    model, vectorizer = get_artifacts()
except Exception:
    logger.exception("Failed to load sentiment model artifacts")
    st.error("The sentiment model is temporarily unavailable. Please try again later.")
    st.stop()

st.title("Restaurant Review Sentiment Analyzer")

review = st.text_area(
    "Enter a restaurant review",
    max_chars=MAX_REVIEW_LENGTH,
    placeholder="Example: The food was delicious and the service was excellent.",
)

if st.button("Predict Sentiment"):
    try:
        result = predict_sentiment(review, model, vectorizer)
    except ReviewValidationError as error:
        st.warning(str(error))
    else:
        confidence = f"{result.confidence:.0%} confidence"
        if result.label == "Positive":
            st.success(f"Positive Review 😊 — {confidence}")
        else:
            st.error(f"Negative Review 😠 — {confidence}")
