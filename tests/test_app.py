import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _run_app():
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


def _markdown(app) -> str:
    """All rendered markdown joined — the hero and result cards are custom HTML."""
    return "\n".join(block.value for block in app.markdown)


def _analyze(app):
    """Click the primary Analyze button (not one of the example chips)."""
    button = next(b for b in app.button if "Analyze" in b.label)
    button.click()
    return app.run()


class AppTests(unittest.TestCase):
    def test_app_loads_without_error(self) -> None:
        app = _run_app()

        self.assertFalse(app.exception)
        self.assertIn("Sentiment Analyzer", _markdown(app))

    def test_positive_review_shows_result(self) -> None:
        app = _run_app()
        app.text_area[0].set_value("good food")
        app = _analyze(app)

        self.assertFalse(app.exception)
        markdown = _markdown(app)
        self.assertIn("Positive", markdown)
        self.assertIn("confidence", markdown)

    def test_negative_review_shows_result(self) -> None:
        app = _run_app()
        app.text_area[0].set_value("bad food")
        app = _analyze(app)

        self.assertFalse(app.exception)
        self.assertIn("Negative", _markdown(app))

    def test_empty_review_shows_warning(self) -> None:
        app = _run_app()
        app.text_area[0].set_value("   ")
        app = _analyze(app)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.warning), 1)
        self.assertIn("Enter a review", app.warning[0].value)

    def test_unrecognized_words_show_warning(self) -> None:
        app = _run_app()
        app.text_area[0].set_value("!!! 123456")
        app = _analyze(app)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.warning), 1)
        self.assertIn("recognized words", app.warning[0].value)


if __name__ == "__main__":
    unittest.main()
