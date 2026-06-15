# Project Conventions for Claude Code

## CRITICAL: Commit & Attribution Rules

**Claude Code MUST follow these rules without exception:**

1. **NEVER add `Co-Authored-By: Claude` or any AI co-author trailer to commit messages.**
2. **NEVER add `🤖 Generated with Claude Code` footers or any AI attribution.**
3. **NEVER mention Claude, Anthropic, AI, or LLMs in commit messages, code comments, file headers, or documentation.**
4. **All commits must be authored solely by:**
   - Name: `apoorvrajdev`
   - Email: `apoorvrajmgr@gmail.com`
5. **NEVER stage or commit changes on your own.** Only suggest commit messages — the user runs `git commit` themselves.
6. **NEVER push to remote.** Only the user pushes.

## Commit Message Format

Use Conventional Commits. Examples:
- `chore: pin dependencies and add lockfile`
- `feat(inference): expose prediction confidence score`
- `fix(inference): align preprocessing with training pipeline`
- `docs: correct feature-extraction description in README`
- `test(inference): add regression tests for inflected words`
- `refactor(train): extract shared preprocessing helper`

Keep subject under 72 characters. Body optional but explains *why*, not *what*.

## Commit Granularity

**Prefer many small, focused commits over a few large ones** — more commits means
more GitHub contribution activity, which is a goal for this project. Split a batch
of work so each logical change lands as its own Conventional Commit.

- When changes touch multiple files for **distinct** reasons, commit them
  separately (one `git add <file>` + `git commit` pair each) rather than together.
- Always suggest the commit sequence as separate per-file / per-change commits so
  the user can rack up individual commits.
- Each commit must still stand on its own as one coherent, meaningful change with
  a clear message. Do **not** split a single indivisible change across artificial
  commits purely to inflate the count — cohesive granularity, not padding.

## Project Stack

- **Language:** Python 3.10+ (CI runs 3.11)
- **Web app:** Streamlit
- **ML / NLP:** scikit-learn, NLTK, pandas, numpy, joblib
- **Quality:** pytest (ruff and mypy recommended, not yet configured)

## Architecture

- `app.py` — Streamlit UI only. No business logic; it imports from `inference.py`.
- `inference.py` — preprocessing, input validation, artifact loading, and prediction.
- `train.py` — reproducible training script that produces `model.pkl` and `vectorizer.pkl`.
- `tests/` — pytest/unittest suite.
- `docs/` — architecture and deployment notes.
- `data/` — training dataset (`Restaurant_Reviews.tsv`).

## Code Standards

- All Python functions have type hints.
- **Training and inference MUST share the same `preprocess()` function** (in `inference.py`).
  Any change to text cleaning, stemming, or stopword handling requires retraining the model
  so the vectorizer vocabulary stays consistent — never let the two pipelines diverge.
- Model artifacts (`model.pkl`, `vectorizer.pkl`) are regenerated **only** via `python train.py`,
  never hand-edited or produced from an ad-hoc notebook flow.
- User-facing validation errors raise `ReviewValidationError` with a clear, friendly message.
- Keep `app.py` thin — UI rendering and orchestration only.
- README and `docs/` must accurately reflect the actual model and feature extraction in use
  (TF-IDF via `TfidfVectorizer`, Logistic Regression) — keep docs in sync with code.

## Working Style

- Plan before implementing for any non-trivial change.
- One change at a time, with tests; run `pytest` before suggesting a commit.
- When model behavior changes, retrain and report the new accuracy/confusion matrix.
- After making changes, summarize what you did so the user can review and commit.
