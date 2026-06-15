import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _run_app():
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


class AppTests(unittest.TestCase):
    def test_app_loads_without_error(self) -> None:
        app = _run_app()

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Restaurant Review Sentiment Analyzer")

    def test_positive_review_shows_success(self) -> None:
        app = _run_app()
        app.text_area[0].set_value("good food")
        app.button[0].click()
        app.run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.success), 1)
        self.assertIn("Positive", app.success[0].value)
        self.assertIn("confidence", app.success[0].value)

    def test_empty_review_shows_warning(self) -> None:
        app = _run_app()
        app.text_area[0].set_value("   ")
        app.button[0].click()
        app.run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.warning), 1)
        self.assertIn("Enter a review", app.warning[0].value)

    def test_unrecognized_words_show_warning(self) -> None:
        app = _run_app()
        app.text_area[0].set_value("!!! 123456")
        app.button[0].click()
        app.run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.warning), 1)
        self.assertIn("recognized words", app.warning[0].value)


if __name__ == "__main__":
    unittest.main()
