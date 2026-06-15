import shutil
import tempfile
import unittest
from pathlib import Path

import joblib

from inference import (
    CHECKSUM_FILENAME,
    MODEL_FILENAME,
    VECTORIZER_FILENAME,
    ArtifactIntegrityError,
    load_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]


class LoadArtifactsTests(unittest.TestCase):
    def test_bundled_artifacts_load_and_verify(self) -> None:
        # The repo's committed artifacts and checksum manifest must agree.
        model, vectorizer = load_artifacts()

        self.assertTrue(callable(getattr(model, "predict_proba", None)))
        self.assertTrue(callable(getattr(vectorizer, "transform", None)))

    def test_missing_artifacts_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_artifacts(Path(directory))

    def test_wrong_object_types_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            joblib.dump({"not": "a model"}, path / MODEL_FILENAME)
            joblib.dump({"not": "a vectorizer"}, path / VECTORIZER_FILENAME)

            with self.assertRaises(ArtifactIntegrityError):
                load_artifacts(path)

    def test_checksum_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            shutil.copy(ROOT / MODEL_FILENAME, path / MODEL_FILENAME)
            shutil.copy(ROOT / VECTORIZER_FILENAME, path / VECTORIZER_FILENAME)
            # Manifest claims a digest the (untampered) file will not match.
            (path / CHECKSUM_FILENAME).write_text(
                f"{'0' * 64}  {MODEL_FILENAME}\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ArtifactIntegrityError, "Checksum mismatch"):
                load_artifacts(path)


if __name__ == "__main__":
    unittest.main()
