# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Prediction confidence score shown in the UI (`predict_proba`), with prediction logging.
- Reproducible training script (`train.py`) producing the model, vectorizer, and a `sha256` checksum manifest.
- Artifact integrity checks in `load_artifacts()` — checksum verification and object-type validation.
- Model card (`docs/MODEL_CARD.md`).
- Expanded test suite (17 tests): inference, artifact-loading failure paths, and the Streamlit UI via `AppTest`.
- `ruff` lint/format and coverage gates in CI; split runtime/dev dependencies.

### Changed
- Inference now shares a single `preprocess()` with training, eliminating train/serve skew.
- Switched the classifier from Gaussian to Multinomial Naive Bayes (test accuracy 0.73 → 0.78).
- Adopted TF-IDF + Logistic Regression after benchmarking (test accuracy 0.78 → 0.82); see `docs/EVALUATION.md`.
- `train.py` now fits the vectorizer on the training split only (no test-set vocabulary leakage).
- Upgraded scikit-learn 1.2.2 → 1.7.x and retrained artifacts.
- Rewrote the README to portfolio standard and aligned the architecture docs.

### Added (analysis)
- `evaluate.py` benchmarking script and `docs/EVALUATION.md` report.

### Fixed
- Corrected documentation that misdescribed the feature extraction and model.

## [1.0.0]

### Added
- Streamlit UI
- Naive Bayes sentiment model
- Input validation
- Automated tests
- GitHub Actions CI
