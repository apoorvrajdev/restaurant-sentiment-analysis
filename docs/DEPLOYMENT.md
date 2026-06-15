# Deployment Guide

## Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs at http://localhost:8501.

## Streamlit Community Cloud

The live demo is hosted on Streamlit Community Cloud.

1. Push the repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo, branch `main`, entrypoint `app.py`.
3. Streamlit installs `requirements.txt` automatically and serves the app.

The bundled `model.pkl`, `vectorizer.pkl`, and `artifacts.sha256` deploy with the
repository — no separate model hosting is required. NLTK stopwords are downloaded
on first use.

## Updating the model

Retrain locally and commit the regenerated artifacts together:

```bash
python train.py
git add model.pkl vectorizer.pkl artifacts.sha256
```

`load_artifacts()` verifies the checksum manifest on startup, so the artifacts and
their manifest must always be committed as a set.
