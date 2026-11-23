"""YOLO features, true labels and projection persistence without a detector install."""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.set9.YOLOfeatureprojection import YOLOFeatureProjection


class FakeTensor:
    """Only the tensor operations needed for global average pooling."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape

    def detach(self):
        return self

    def float(self):
        return self

    def mean(self, dim):
        return FakeTensor(self.values.mean(axis=dim))

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeHead:
    def __init__(self):
        self.callback = None
        self.remove = Mock()

    def register_forward_pre_hook(self, callback):
        self.callback = callback
        return SimpleNamespace(remove=self.remove)

    def emit(self, maps):
        self.callback(self, (maps,))


class YOLOFeatureProjectionTests(unittest.TestCase):
    module = "forensics_app.tools.set9.YOLOfeatureprojection"

    def setUp(self):
        self.tool = YOLOFeatureProjection()
        self.head = FakeHead()
        self.tool.model = SimpleNamespace(
            names={0: "healthy", 1: "infection"},
            model=SimpleNamespace(model=[self.head]),
        )

    def _make_dataset(self, directory):
        root = Path(directory)
        images, labels = root / "images", root / "labels"
        images.mkdir()
        labels.mkdir()
        return images, labels

    def _predict(self, image):
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image[0, 0], [255, 0, 0])
        self.head.emit([
            FakeTensor(np.array([[[[1, 3], [5, 7]], [[2, 4], [6, 8]]]])),
            FakeTensor(np.full((1, 1, 3, 3), 10)),
        ])
        # Predictions intentionally disagree with annotations: labels must come from txt files.
        return [SimpleNamespace(names={0: "predicted"})]

    def test_dataset_selection_accepts_test_or_images_and_rejects_missing_folders(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "images and labels"):
                self.tool._get_dataset_folders(directory)
            images, labels = self._make_dataset(directory)
            self.assertEqual(self.tool._get_dataset_folders(directory), (images, labels))
            self.assertEqual(self.tool._get_dataset_folders(images), (images, labels))

    def test_label_names_are_true_unique_classes_and_malformed_labels_are_rejected(self):
        with TemporaryDirectory() as directory:
            images, labels = self._make_dataset(directory)
            image = images / "sample.png"
            path = labels / "sample.txt"
            self.assertEqual(self.tool._get_image_label(image, images, labels), "Unlabelled")
            path.write_text("\n")
            self.assertEqual(self.tool._get_image_label(image, images, labels), "No objects")
            path.write_text("1 .5 .5 .2 .2\n0 .1 .1 .2 .2\n1 .3 .3 .1 .1\n")
            self.assertEqual(self.tool._get_image_label(image, images, labels), "healthy, infection")
            self.tool.model.names = ["healthy", "infection"]
            self.assertEqual(self.tool._get_image_label(image, images, labels), "healthy, infection")
            for token in ("oops", "-1", "2"):
                with self.subTest(token=token):
                    path.write_text(f"{token} .5 .5 .2 .2\n")
                    with self.assertRaisesRegex(ValueError, "class ID"):
                        self.tool._get_image_label(image, images, labels)

    def test_multiscale_pooling_uses_rgb_predict_and_skips_bad_rows_without_misalignment(self):
        with TemporaryDirectory() as directory:
            images, labels = self._make_dataset(directory)
            for name in ("a", "b_bad", "c"):
                Image.new("RGB", (6, 4), "red").save(images / f"{name}.png")
            (labels / "a.txt").write_text("1 .5 .5 .2 .2\n")
            (labels / "b_bad.txt").write_text("invalid .5 .5 .2 .2\n")
            (labels / "c.txt").write_text("0 .5 .5 .2 .2\n")
            (images / "d_broken.png").write_bytes(b"not an image")
            with patch.object(self.tool, "_predict", create=True, side_effect=self._predict) as predict, \
                 patch("builtins.print") as report:
                features, true_labels, paths = self.tool._extract_features(directory)
            self.assertEqual(predict.call_count, 2)
            self.assertGreaterEqual(report.call_count, 2)
        np.testing.assert_array_equal(features, [[4, 5, 10], [4, 5, 10]])
        self.assertEqual(true_labels.tolist(), ["infection", "healthy"])
        self.assertEqual(paths.tolist(), ["a.png", "c.png"])
        self.head.remove.assert_called_once_with()

    def test_capture_is_cleared_per_image_and_hook_is_removed_on_prediction_failure(self):
        with TemporaryDirectory() as directory:
            images, labels = self._make_dataset(directory)
            for name in ("a", "b"):
                Image.new("RGB", (6, 4), "red").save(images / f"{name}.png")
            calls = 0

            def first_prediction_only(image):
                nonlocal calls
                calls += 1
                if calls == 1:
                    self._predict(image)

            with patch.object(self.tool, "_predict", create=True, side_effect=first_prediction_only):
                # The second prediction must fail instead of reusing the first image's vector.
                with self.assertRaisesRegex(ValueError, "did not return feature maps"):
                    self.tool._extract_features(directory)
            self.assertEqual(calls, 2)
            self.head.remove.assert_called_once_with()

    def test_hook_removed_if_detector_raises_and_empty_input_is_clear(self):
        with TemporaryDirectory() as directory:
            images, labels = self._make_dataset(directory)
            with self.assertRaisesRegex(ValueError, "No valid image features"):
                self.tool._extract_features(directory)
            self.head.remove.assert_called_once_with()
            self.head.remove.reset_mock()
            Image.new("RGB", (6, 4), "red").save(images / "sample.png")
            with patch.object(self.tool, "_predict", create=True, side_effect=RuntimeError("detector failed")):
                with self.assertRaisesRegex(RuntimeError, "detector failed"):
                    self.tool._extract_features(directory)
            self.head.remove.assert_called_once_with()

    def test_save_load_preserves_features_labels_paths_without_loading_detector(self):
        features = np.arange(18, dtype=np.float32).reshape(6, 3)
        labels = np.array(["healthy"] * 3 + ["infection"] * 3)
        paths = np.array([f"test_{index}.png" for index in range(6)])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npz"
            with patch(f"{self.module}.dialog_options", return_value=self.tool.options[0]), \
                 patch(f"{self.module}.filedialog.askopenfilename", return_value="best.pt"), \
                 patch(f"{self.module}.filedialog.askdirectory", return_value=directory), \
                 patch(f"{self.module}.filedialog.asksaveasfilename", return_value=str(path)), \
                 patch.object(self.tool, "_load_model"), \
                 patch.object(self.tool, "_extract_features", return_value=(features, labels, paths)):
                result = self.tool.run(None, ImageDocument())
            self.assertEqual(result.details["Labels"], {"healthy": 3, "infection": 3})
            restored = YOLOFeatureProjection()
            with patch(f"{self.module}.dialog_options", return_value=restored.options[1]), \
                 patch(f"{self.module}.filedialog.askopenfilename", return_value=str(path)), \
                 patch.object(restored, "_load_model") as load_model, \
                 patch.object(restored, "_visualize_features") as show:
                restored.run(None, ImageDocument())
            load_model.assert_not_called()
            show.assert_called_once_with()
        np.testing.assert_array_equal(restored.features, features)
        np.testing.assert_array_equal(restored.labels, labels)
        np.testing.assert_array_equal(restored.paths, paths)

    def test_saved_feature_validation_rejects_misaligned_paths_and_nonfinite_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npz"
            for values in (
                {"features": np.ones((3, 2)), "labels": np.array(["a"] * 3), "paths": np.array(["one"])},
                {"features": np.full((3, 2), np.nan), "labels": np.array(["a"] * 3), "paths": np.array(["one"] * 3)},
            ):
                with self.subTest(values=list(values)):
                    np.savez_compressed(path, **values)
                    with self.assertRaises(ValueError):
                        self.tool._load_features(path)
                    self.assertIsNone(self.tool.features)
            path.write_bytes(b"not npz")
            with self.assertRaisesRegex(ValueError, "Could not load"):
                self.tool._load_features(path)

    def test_real_projections_are_2d_finite_and_keep_true_label_legends(self):
        rng = np.random.default_rng(4)
        self.tool.features = np.vstack((rng.normal(-2, .2, (15, 8)), rng.normal(2, .2, (15, 8))))
        self.tool.labels = np.array(["healthy"] * 15 + ["infection"] * 15)

        def inspect_plot():
            for axis in plt.gcf().axes:
                self.assertEqual([entry.get_text() for entry in axis.get_legend().get_texts()], ["healthy", "infection"])

        with patch("matplotlib.pyplot.show", side_effect=inspect_plot):
            self.tool._visualize_features()
        self.assertEqual(set(self.tool.projections), {"t-SNE", "UMAP"})
        for values in self.tool.projections.values():
            self.assertEqual(values.shape, (30, 2))
            self.assertTrue(np.isfinite(values).all())

    def test_tool_is_registered_without_an_open_image(self):
        registered = next(tool for tool in build_tool_registry().all() if tool.tool_id == self.tool.tool_id)
        self.assertIsInstance(registered, YOLOFeatureProjection)
        self.assertFalse(registered.requires_image)


if __name__ == "__main__":
    unittest.main()
