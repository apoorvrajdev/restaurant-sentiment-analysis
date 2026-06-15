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

st.set_page_config(
    page_title="Restaurant Sentiment Analyzer",
    page_icon="🍽️",
    layout="centered",
)

REPO_URL = "https://github.com/apoorvrajdev/restaurant-sentiment-analysis"
EXAMPLES = {
    "😋 Loved it": "The food was absolutely delicious and the service was excellent.",
    "🙂 Solid": "Fresh ingredients, generous portions, and a cozy atmosphere.",
    "😠 Disappointing": "Worst meal ever, cold food and rude service.",
}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stButton > button, textarea {
    font-family: 'Inter', -apple-system, sans-serif;
}
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 680px; padding-top: 3rem; padding-bottom: 4rem; }

.hero-kicker {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #4F46E5; margin-bottom: 0.4rem;
}
.hero-title {
    font-size: 2.4rem; font-weight: 700; line-height: 1.1;
    color: #0F172A; margin: 0 0 0.6rem 0;
}
.hero-sub { font-size: 1.05rem; color: #475569; margin-bottom: 1.75rem; }

.stButton > button {
    border-radius: 10px; border: 1px solid #E2E8F0; font-weight: 500;
    transition: all 0.15s ease;
}
.stButton > button:hover { border-color: #4F46E5; color: #4F46E5; }
.stTextArea textarea {
    border-radius: 12px; border: 1px solid #E2E8F0; font-size: 1rem;
}

.result {
    border-radius: 14px; padding: 1.25rem 1.4rem; margin-top: 0.5rem;
    border: 1px solid #E2E8F0; background: #FFFFFF;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}
.result-row { display: flex; justify-content: space-between; align-items: baseline; }
.result-label { font-size: 1.25rem; font-weight: 700; }
.result-conf { font-size: 0.95rem; color: #64748B; font-weight: 500; }
.result.positive .result-label { color: #047857; }
.result.negative .result-label { color: #B91C1C; }
.meter {
    height: 8px; border-radius: 999px; background: #F1F5F9;
    margin-top: 0.9rem; overflow: hidden;
}
.meter-fill { height: 100%; border-radius: 999px; }
.result.positive .meter-fill { background: #10B981; }
.result.negative .meter-fill { background: #EF4444; }

.footer {
    margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid #F1F5F9;
    font-size: 0.85rem; color: #94A3B8; text-align: center;
}
.footer a { color: #64748B; text-decoration: none; font-weight: 500; }
.footer a:hover { color: #4F46E5; }
</style>
"""


@st.cache_resource
def get_artifacts():
    return load_artifacts()


def render_result(label: str, confidence: float) -> None:
    sentiment = "positive" if label == "Positive" else "negative"
    emoji = "😊" if label == "Positive" else "😞"
    pct = round(confidence * 100)
    st.markdown(
        f"""
        <div class="result {sentiment}">
          <div class="result-row">
            <span class="result-label">{emoji} {label}</span>
            <span class="result-conf">{pct}% confidence</span>
          </div>
          <div class="meter"><div class="meter-fill" style="width:{pct}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(STYLE, unsafe_allow_html=True)

try:
    model, vectorizer = get_artifacts()
except Exception:
    logger.exception("Failed to load sentiment model artifacts")
    st.error("The sentiment model is temporarily unavailable. Please try again later.")
    st.stop()

st.markdown(
    """
    <div class="hero-kicker">Restaurant Review Sentiment Analyzer</div>
    <h1 class="hero-title">Is this review positive?</h1>
    <p class="hero-sub">Paste a restaurant review and classify its sentiment in real time,
    with a confidence score behind every prediction.</p>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("review_input", "")

st.caption("Try an example")
example_columns = st.columns(len(EXAMPLES))
for column, (chip_label, chip_text) in zip(example_columns, EXAMPLES.items(), strict=True):
    if column.button(chip_label, use_container_width=True):
        st.session_state["review_input"] = chip_text

review = st.text_area(
    "Your review",
    key="review_input",
    height=130,
    max_chars=MAX_REVIEW_LENGTH,
    placeholder="Example: The food was delicious and the service was excellent.",
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    try:
        result = predict_sentiment(review, model, vectorizer)
    except ReviewValidationError as error:
        st.warning(str(error))
    else:
        render_result(result.label, result.confidence)

st.markdown(
    f"""
    <div class="footer">
      Logistic Regression · TF-IDF · scikit-learn &nbsp;·&nbsp;
      <a href="{REPO_URL}" target="_blank">View source on GitHub ↗</a>
    </div>
    """,
    unsafe_allow_html=True,
)
