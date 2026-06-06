import unittest

from inference import (
    MAX_REVIEW_LENGTH,
    ReviewValidationError,
    load_artifacts,
    predict_sentiment,
)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.vectorizer = load_artifacts()

    def test_predicts_known_positive_review(self) -> None:
        result = predict_sentiment("good food", self.model, self.vectorizer)

        self.assertEqual(result, "Positive")

    def test_predicts_known_negative_review(self) -> None:
        result = predict_sentiment("bad food", self.model, self.vectorizer)

        self.assertEqual(result, "Negative")

    def test_rejects_empty_review(self) -> None:
        with self.assertRaisesRegex(ReviewValidationError, "Enter a review"):
            predict_sentiment("   ", self.model, self.vectorizer)

    def test_rejects_oversized_review(self) -> None:
        with self.assertRaisesRegex(ReviewValidationError, "characters or fewer"):
            predict_sentiment("food " * MAX_REVIEW_LENGTH, self.model, self.vectorizer)

    def test_rejects_review_without_recognized_words(self) -> None:
        with self.assertRaisesRegex(ReviewValidationError, "recognized words"):
            predict_sentiment("!!! 😠 123456", self.model, self.vectorizer)


if __name__ == "__main__":
    unittest.main()
