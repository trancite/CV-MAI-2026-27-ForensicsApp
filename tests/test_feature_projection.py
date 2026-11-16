"""Feature extraction, saved test vectors and real 2D projections."""

import importlib.util
import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.set8.feature_projection import FeatureProjection


class FeatureProjectionTests(unittest.TestCase):
    module = "forensics_app.tools.set8.feature_projection"

    def setUp(self):
        self.tool = FeatureProjection()

    def _make_model(self):
        from tensorflow import keras
        inputs = keras.Input(shape=(8, 8, 3))
        pooled = keras.layers.GlobalAveragePooling2D()(inputs)
        features = keras.layers.Dense(4, use_bias=False)(pooled)
        self.tool.model = keras.Model(inputs, keras.layers.Dense(2)(features))
        weights = np.column_stack((np.eye(3), np.ones(3))).astype(np.float32)
        self.tool.model.layers[-2].set_weights([weights])

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
    def test_extracts_latent_vectors_in_batches_with_aligned_labels_and_paths(self):
        self._make_model()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cats, dogs = root / "cats", root / "dogs"
            cats.mkdir()
            dogs.mkdir()
            for index in range(32):
                Image.new("RGB", (10, 12), "black").save(cats / f"black_{index:02}.png")
            Image.new("RGB", (13, 9), "red").save(cats / "red.png")
            Image.new("L", (11, 10), 255).save(dogs / "gray.png")
            Image.new("RGB", (12, 14), "white").save(dogs / "white.png")
            (cats / "broken.jpg").write_text("not an image")
            (root / "readme.txt").write_text("ignored")
            features, labels, paths = self.tool._extract_features(root)
        expected = np.array([[-1, -1, -1, -3]] * 32 + [[1, -1, -1, -1]] + [[1, 1, 1, 3]] * 2)
        np.testing.assert_allclose(features, expected, atol=1e-5)
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(labels.tolist(), ["cats"] * 33 + ["dogs"] * 2)
        self.assertEqual(paths[32:].tolist(), ["cats/red.png", "dogs/gray.png", "dogs/white.png"])
        self.assertEqual(len(paths), len(features))

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
    def test_flat_folder_is_supported_and_an_empty_folder_is_rejected(self):
        self._make_model()
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No valid images"):
                self.tool._extract_features(directory)
            Image.new("RGB", (10, 12), "white").save(Path(directory) / "test.png")
            features, labels, paths = self.tool._extract_features(directory)
        self.assertEqual(features.shape, (1, 4))
        self.assertEqual(labels.tolist(), ["Unlabelled"])
        self.assertEqual(paths.tolist(), ["test.png"])

    def test_save_and_load_preserve_vectors_labels_and_paths_without_reextracting(self):
        features = np.arange(24, dtype=np.float32).reshape(6, 4)
        labels = np.array(["gáto"] * 3 + ["dog"] * 3)
        paths = np.array([f"test/{index}.png" for index in range(6)])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            with patch(f"{self.module}.dialog_options", return_value=self.tool.options[0]), \
                 patch(f"{self.module}.filedialog.askopenfilename", return_value="model.zip"), \
                 patch(f"{self.module}.filedialog.askdirectory", return_value=directory), \
                 patch(f"{self.module}.filedialog.asksaveasfilename", return_value=str(path)), \
                 patch.object(self.tool, "_load_image") as load_model, \
                 patch.object(self.tool, "_extract_features", return_value=(features, labels, paths)):
                result = self.tool.run(None, ImageDocument())
            load_model.assert_called_once_with("model.zip")
            self.assertEqual(result.details["Feature dimensions"], 4)
            restored = FeatureProjection()
            with patch(f"{self.module}.dialog_options", return_value=restored.options[1]), \
                 patch(f"{self.module}.filedialog.askopenfilename", return_value=str(path)), \
                 patch.object(restored, "_load_image") as load_model, \
                 patch.object(restored, "_visualize_features") as show:
                result = restored.run(None, ImageDocument())
            load_model.assert_not_called()
            show.assert_called_once()
        np.testing.assert_array_equal(restored.features, features)
        np.testing.assert_array_equal(restored.labels, labels)
        np.testing.assert_array_equal(restored.paths, paths)
        self.assertEqual(result.details["Images"], 6)
        self.assertIsNone(result.image)

    def test_real_tsne_and_umap_are_finite_2d_and_display_true_class_labels(self):
        rng = np.random.default_rng(42)
        self.tool.features = np.vstack((rng.normal(-2, 0.2, (20, 8)), rng.normal(2, 0.2, (20, 8))))
        self.tool.labels = np.array(["cats"] * 20 + ["dogs"] * 20)
        before = set(plt.get_fignums())

        def inspect_plot():
            axes = plt.gcf().axes
            self.assertEqual([axis.get_title() for axis in axes], ["t-SNE", "UMAP"])
            for axis in axes:
                self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()], ["cats", "dogs"])
                self.assertEqual([len(points.get_offsets()) for points in axis.collections], [20, 20])

        with patch("matplotlib.pyplot.show", side_effect=inspect_plot):
            self.tool._visualize_features()
        for projection in self.tool.projections.values():
            self.assertEqual(projection.shape, (40, 2))
            self.assertTrue(np.isfinite(projection).all())
        self.assertEqual(set(plt.get_fignums()), before)

    def test_cancel_save_does_not_load_model_or_extract_images(self):
        with patch(f"{self.module}.dialog_options", return_value=self.tool.options[0]), \
             patch(f"{self.module}.filedialog.askopenfilename", return_value="model.zip"), \
             patch(f"{self.module}.filedialog.askdirectory", return_value="test"), \
             patch(f"{self.module}.filedialog.asksaveasfilename", return_value=""), \
             patch.object(self.tool, "_load_image") as load_model:
            self.assertIsNone(self.tool.run(None, ImageDocument()))
        load_model.assert_not_called()

    def test_is_registered_without_requiring_a_selected_image(self):
        registered = next(tool for tool in build_tool_registry().all() if tool.tool_id == self.tool.tool_id)
        self.assertIsInstance(registered, FeatureProjection)
        self.assertEqual(registered.category, "Set8")
        self.assertFalse(registered.requires_image)


if __name__ == "__main__":
    unittest.main()
