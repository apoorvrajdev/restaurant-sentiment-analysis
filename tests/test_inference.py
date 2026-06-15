import unittest

from inference import (
    MAX_REVIEW_LENGTH,
    ReviewValidationError,
    load_artifacts,
    predict_sentiment,
    preprocess,
)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.vectorizer = load_artifacts()

    def test_predicts_known_positive_review(self) -> None:
        result = predict_sentiment("good food", self.model, self.vectorizer)

        self.assertEqual(result.label, "Positive")

    def test_predicts_known_negative_review(self) -> None:
        result = predict_sentiment("bad food", self.model, self.vectorizer)

        self.assertEqual(result.label, "Negative")

    def test_prediction_reports_confidence(self) -> None:
        result = predict_sentiment("good food", self.model, self.vectorizer)

        # The predicted label is the argmax probability, so for a binary
        # classifier its confidence is always in [0.5, 1.0].
        self.assertGreaterEqual(result.confidence, 0.5)
        self.assertLessEqual(result.confidence, 1.0)

    def test_rejects_empty_review(self) -> None:
        with self.assertRaisesRegex(ReviewValidationError, "Enter a review"):
            predict_sentiment("   ", self.model, self.vectorizer)

    def test_rejects_oversized_review(self) -> None:
        with self.assertRaisesRegex(ReviewValidationError, "characters or fewer"):
            predict_sentiment("food " * MAX_REVIEW_LENGTH, self.model, self.vectorizer)

    def test_rejects_review_without_recognized_words(self) -> None:
        with self.assertRaisesRegex(ReviewValidationError, "recognized words"):
            predict_sentiment("!!! 😠 123456", self.model, self.vectorizer)

    def test_predicts_inflected_positive_review(self) -> None:
        # Regression: inference must stem like training. "loved"/"tasty" only
        # match the model vocabulary ("love"/"tasti") after preprocessing.
        result = predict_sentiment(
            "I loved this place, the food was tasty.", self.model, self.vectorizer
        )

        self.assertEqual(result.label, "Positive")

    def test_predicts_inflected_negative_review(self) -> None:
        result = predict_sentiment(
            "Terrible experience, the service was disgusting.",
            self.model,
            self.vectorizer,
        )

        self.assertEqual(result.label, "Negative")

    def test_preprocess_matches_training_pipeline(self) -> None:
        # Lowercased, stemmed, stopwords removed; "not" deliberately kept.
        self.assertEqual(preprocess("I LOVED the tasty food!"), "love tasti food")
        self.assertEqual(preprocess("This is not good"), "not good")


if __name__ == "__main__":
    unittest.main()
