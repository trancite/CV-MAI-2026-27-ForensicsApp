"""Model passports and classification must use each model's preprocessing."""

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from forensics_app.core import ImageDocument
from forensics_app.tools.set8.model_load import ClassifyWithPretrainedModel


class ModelLoadTests(unittest.TestCase):
    module = "forensics_app.tools.set8.model_load"

    def setUp(self):
        self.tool = ClassifyWithPretrainedModel()

    def _passport(self, size=(32, 32), normalization="zero_one"):
        return {
            "model": "Small classification CNN",
            "dataset": "Test dataset",
            "training_samples": 8,
            "validation_samples": 2,
            "classes": ["cat", "dog"],
            "preprocessing": {
                "color": "RGB",
                "size": list(size),
                "normalization": normalization,
                "instructions": "Resize to the specified size and normalize RGB pixels.",
            },
            "history": {"loss": [0.7, 0.3], "val_loss": [0.8, 0.4]},
        }

    def _model_files(self, root, passport, zipped=False):
        root = Path(root)
        model_path = root / "classifier.keras"
        model_path.write_bytes(b"model loading is mocked")
        if passport is not None:
            model_path.with_suffix(".json").write_text(
                json.dumps(passport), encoding="utf-8"
            )
        if not zipped:
            return model_path
        archive_path = root / "bundle.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(model_path, model_path.name)
            if passport is not None:
                archive.write(model_path.with_suffix(".json"), "classifier.json")
        return archive_path

    @contextmanager
    def _mock_model_loader(self, model):
        """Exercise package handling without importing or executing TensorFlow."""
        tensorflow = ModuleType("tensorflow")
        keras = ModuleType("tensorflow.keras")
        models = ModuleType("tensorflow.keras.models")
        loader = Mock(return_value=model)
        models.load_model = loader
        models.model_from_json = Mock(return_value=model)
        keras.models = models
        tensorflow.keras = keras
        with patch.dict(sys.modules, {
            "tensorflow": tensorflow,
            "tensorflow.keras": keras,
            "tensorflow.keras.models": models,
        }):
            yield loader

    def _model(self, input_shape=(None, 32, 32, 3), output_shape=(None, 2)):
        return SimpleNamespace(input_shape=input_shape, output_shape=output_shape)

    def test_reads_sibling_passport_without_tensorflow(self):
        passport = self._passport()
        with TemporaryDirectory() as root:
            path = self._model_files(root, passport)
            with patch.dict(sys.modules, {"tensorflow": None}):
                self.tool._load_passport(str(path))
        self.assertEqual(self.tool.passport, passport)

    def test_reads_zipped_passport_without_extracting_next_to_original(self):
        passport = self._passport()
        with TemporaryDirectory() as root:
            path = self._model_files(root, passport, zipped=True)
            before = set(Path(root).iterdir())
            with patch.dict(sys.modules, {"tensorflow": None}):
                self.tool._load_passport(str(path))
            self.assertEqual(set(Path(root).iterdir()), before)
        self.assertEqual(self.tool.passport, passport)

    def test_inspect_passport_works_without_selected_image_or_tensorflow(self):
        passport = self._passport()
        self.assertFalse(self.tool.requires_image)
        with TemporaryDirectory() as root:
            path = self._model_files(root, passport, zipped=True)
            with patch.dict(sys.modules, {"tensorflow": None}), \
                 patch(f"{self.module}.dialog_options", return_value=self.tool.options[0]), \
                 patch(f"{self.module}.filedialog.askopenfilename", return_value=str(path)), \
                 patch.object(self.tool, "_load_image") as load_model, \
                 patch.object(self.tool, "_show_passport") as show:
                result = self.tool.run(None, ImageDocument())
        load_model.assert_not_called()
        show.assert_called_once_with(passport, None)
        self.assertEqual(result.details, passport)
        self.assertIsNone(result.image)

    def test_classification_requires_an_image_before_loading_model(self):
        with patch(f"{self.module}.dialog_options", return_value=self.tool.options[1]), \
             patch(f"{self.module}.filedialog.askopenfilename", return_value="classifier.keras"), \
             patch.object(self.tool, "_load_image") as load_model, \
             self.assertRaisesRegex(ValueError, "(?i)image"):
            self.tool.run(None, ImageDocument())
        load_model.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
    def test_preprocessing_resizes_and_uses_explicit_normalization(self):
        image = np.zeros((9, 13, 3), dtype=np.uint8)
        image[..., 0] = 255
        expectations = {
            "zero_one": [1.0, 0.0, 0.0],
            "minus_one_one": [1.0, -1.0, -1.0],
            "none": [255.0, 0.0, 0.0],
        }
        for normalization, expected in expectations.items():
            with self.subTest(normalization=normalization):
                self.tool.passport = self._passport(normalization=normalization)
                actual = self.tool._preprocess_image(image)
                self.assertEqual(actual.shape, (32, 32, 3))
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape))

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
    def test_legacy_inception_passport_keeps_128_pixel_minus_one_one_preprocessing(self):
        passport = self._passport(size=(128, 128))
        del passport["preprocessing"]["normalization"]
        passport["model"] = "InceptionV3: frozen base, trained classification head"
        passport["preprocessing"]["instructions"] = (
            "Resize to 128x128; convert to float32; "
            "scale RGB pixels from [0,255] to [-1,1]."
        )
        with TemporaryDirectory() as root:
            path = self._model_files(root, passport)
            self.tool._load_passport(str(path))
            actual = self.tool._preprocess_image(np.full((11, 7, 3), 255, dtype=np.uint8))
        self.assertEqual(actual.shape, (128, 128, 3))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, 1.0)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
    def test_classification_uses_passport_classes_and_preprocessed_batch(self):
        self.tool.passport = self._passport()
        predict = Mock(return_value=np.array([[0.1, 0.9]], dtype=np.float32))
        self.tool.model = self._model()
        self.tool.model.predict = predict
        label, probability = self.tool._classify_image(
            np.full((11, 7, 3), 255, dtype=np.uint8)
        )
        self.assertEqual(label, "dog")
        self.assertAlmostEqual(probability, 0.9, places=6)
        batch = predict.call_args.args[0]
        self.assertEqual(batch.shape, (1, 32, 32, 3))
        self.assertEqual(batch.dtype, np.float32)
        np.testing.assert_allclose(batch, 1.0)

    def test_loads_direct_or_zipped_model_and_its_matching_passport(self):
        passport = self._passport()
        for zipped in (False, True):
            with self.subTest(zipped=zipped), TemporaryDirectory() as root:
                path = self._model_files(root, passport, zipped=zipped)
                before = set(Path(root).iterdir())
                model = self._model()
                with self._mock_model_loader(model) as load_model:
                    self.tool._load_image(str(path))
                self.assertIs(self.tool.model, model)
                self.assertEqual(self.tool.passport, passport)
                self.assertFalse(load_model.call_args.kwargs["compile"])
                self.assertEqual(set(Path(root).iterdir()), before)

    def test_missing_passport_has_an_informative_error_for_both_formats(self):
        for zipped in (False, True):
            with self.subTest(zipped=zipped), TemporaryDirectory() as root:
                path = self._model_files(root, None, zipped=zipped)
                with self.assertRaisesRegex(FileNotFoundError, "(?i)passport|json"):
                    self.tool._load_passport(str(path))

    def test_empty_passport_class_list_is_rejected(self):
        passport = self._passport()
        passport["classes"] = []
        with TemporaryDirectory() as root:
            path = self._model_files(root, passport)
            with self.assertRaisesRegex(ValueError, "(?i)class"):
                self.tool._load_passport(str(path))

    def test_rejects_passport_size_that_disagrees_with_model_input(self):
        passport = self._passport(size=(128, 128))
        with TemporaryDirectory() as root:
            path = self._model_files(root, passport)
            with self._mock_model_loader(self._model()), \
                 self.assertRaisesRegex(ValueError, "(?i)size|shape|input"):
                self.tool._load_image(str(path))

    def test_rejects_class_list_that_disagrees_with_model_output(self):
        passport = self._passport()
        with TemporaryDirectory() as root:
            path = self._model_files(root, passport)
            model = self._model(output_shape=(None, 3))
            with self._mock_model_loader(model), \
                 self.assertRaisesRegex(ValueError, "(?i)class|output"):
                self.tool._load_image(str(path))


if __name__ == "__main__":
    unittest.main()
