import logging
from string import Template

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
BANNER_IMG = "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=1600&q=80"
FOOD_THUMBS = [
    "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=600&q=80",
    "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=600&q=80",
    "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=600&q=80",
]
EXAMPLES = {
    "😋 Loved it": "The food was absolutely delicious and the service was excellent.",
    "🙂 Solid": "Fresh ingredients, generous portions, and a cozy atmosphere.",
    "😠 Disappointing": "Worst meal ever, cold food and rude service.",
}

LIGHT = {
    "bg": "#FFFFFF",
    "secondary": "#F8FAFC",
    "text": "#0F172A",
    "sub": "#475569",
    "border": "#E2E8F0",
    "card": "#FFFFFF",
    "accent": "#4F46E5",
    "meter_bg": "#F1F5F9",
}
DARK = {
    "bg": "#0E1117",
    "secondary": "#161B22",
    "text": "#E6EDF3",
    "sub": "#9AA7B4",
    "border": "#232A33",
    "card": "#161B22",
    "accent": "#818CF8",
    "meter_bg": "#232A33",
}

STYLE = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stButton > button, textarea {
    font-family: 'Inter', -apple-system, sans-serif;
}
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"] { background: $bg !important; }
.block-container { max-width: 720px; padding-top: 0.75rem; padding-bottom: 1rem; }
[data-testid="stVerticalBlock"] { gap: 0.5rem; }
.stApp, .stApp p, .stApp label, .stMarkdown,
[data-testid="stWidgetLabel"] p { color: $text !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: $sub !important; margin-bottom: 0;
}

.banner {
    position: relative; border-radius: 16px; overflow: hidden;
    padding: 1.5rem 1.9rem; margin-bottom: 0.5rem;
    background: linear-gradient(135deg, rgba(15,23,42,0.80), rgba(79,70,229,0.55)),
                url('$banner') center/cover;
}
.banner .kicker {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.09em;
    text-transform: uppercase; color: #C7D2FE !important; margin-bottom: 0.3rem;
}
.banner .title {
    font-size: 1.8rem; font-weight: 700; line-height: 1.1; color: #FFFFFF !important; margin: 0;
}
.banner .sub {
    font-size: 0.9rem; color: #E2E8F0 !important; margin-top: 0.35rem; max-width: 34rem;
}

.food-strip { display: flex; gap: 10px; margin-bottom: 0.5rem; }
.food-strip img { flex: 1; width: 100%; height: 54px; object-fit: cover; border-radius: 10px; }

.stTextArea textarea {
    border-radius: 12px; border: 1px solid $border; font-size: 1rem;
    background: $secondary !important; color: $text !important;
}
.stButton > button {
    border-radius: 10px; border: 1px solid $border; font-weight: 500;
    background: $card !important; color: $text !important; transition: all 0.15s ease;
}
.stButton > button:hover { border-color: $accent !important; color: $accent !important; }
.stButton > button[kind="primary"] {
    background: $accent !important; color: #FFFFFF !important; border: none;
}

.result {
    border-radius: 14px; padding: 1rem 1.3rem; margin-top: 0.25rem;
    border: 1px solid $border; background: $card;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}
.result-row { display: flex; justify-content: space-between; align-items: baseline; }
.result-label { font-size: 1.25rem; font-weight: 700; }
.result-conf { font-size: 0.95rem; color: $sub; font-weight: 500; }
.result.positive .result-label { color: #10B981; }
.result.negative .result-label { color: #EF4444; }
.meter {
    height: 8px; border-radius: 999px; background: $meter_bg;
    margin-top: 0.9rem; overflow: hidden;
}
.meter-fill { height: 100%; border-radius: 999px; }
.result.positive .meter-fill { background: #10B981; }
.result.negative .meter-fill { background: #EF4444; }

.footer {
    margin-top: 1.25rem; padding-top: 0.8rem; border-top: 1px solid $border;
    font-size: 0.82rem; color: $sub; text-align: center;
}
.footer a { color: $accent; text-decoration: none; font-weight: 500; }
</style>
""")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


def toggle_theme() -> None:
    st.session_state.dark = not st.session_state.get("dark", False)


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


st.session_state.setdefault("dark", False)
st.session_state.setdefault("review_input", "")
dark = st.session_state.dark

st.markdown(STYLE.substitute(DARK if dark else LIGHT, banner=BANNER_IMG), unsafe_allow_html=True)

try:
    model, vectorizer = get_artifacts()
except Exception:
    logger.exception("Failed to load sentiment model artifacts")
    st.error("The sentiment model is temporarily unavailable. Please try again later.")
    st.stop()

_, toggle_column = st.columns([6, 1])
toggle_column.button(
    "☀️ Day" if dark else "🌙 Night",
    on_click=toggle_theme,
    use_container_width=True,
    help="Switch between day and night theme",
)

st.markdown(
    f"""
    <div class="banner">
      <div class="kicker">Restaurant Review Sentiment Analyzer</div>
      <h1 class="title">Is this review positive?</h1>
      <p class="sub">Paste a restaurant review and classify its sentiment in real time,
      with a confidence score behind every prediction.</p>
    </div>
    <div class="food-strip">
      {"".join(f'<img src="{src}" alt="Food" />' for src in FOOD_THUMBS)}
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Try an example")
example_columns = st.columns(len(EXAMPLES))
for column, (chip_label, chip_text) in zip(example_columns, EXAMPLES.items(), strict=True):
    if column.button(chip_label, use_container_width=True):
        st.session_state["review_input"] = chip_text

review = st.text_area(
    "Your review",
    key="review_input",
    height=80,
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
