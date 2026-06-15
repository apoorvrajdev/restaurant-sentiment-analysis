# Contributing

Thank you for your interest in contributing to this project.

## Local Setup

```bash
git clone https://github.com/apoorvrajdev/restaurant-sentiment-analysis.git
cd restaurant-sentiment-analysis
pip install -r requirements-dev.txt
```

## Before submitting a change

Run the same checks CI runs — they must all pass:

```bash
ruff check .
ruff format --check .
pytest
```

## Conventions

- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.
- Keep training and inference preprocessing in sync — both use the shared
  `preprocess()` in `inference.py`. Divergence reintroduces train/serve skew.
- If model behavior changes, retrain via `python train.py` and commit the
  regenerated `model.pkl`, `vectorizer.pkl`, and `artifacts.sha256` together.
