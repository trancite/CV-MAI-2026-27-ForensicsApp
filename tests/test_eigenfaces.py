"""PCA variance selection, flat training folders, and eigenface visualization."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.set6.eigenfaces import EigenFaces


class EigenFacesTests(unittest.TestCase):
    module = "forensics_app.tools.set6.eigenfaces"

    def setUp(self):
        self.tool = EigenFaces()
        self.rng = np.random.default_rng(42)

    def test_pov_matches_covariance_and_uses_minimum_component_count(self):
        for shape in ((8, 15), (15, 8)):
            X = self.rng.normal(size=shape)
            original = X.copy()
            full_values = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
            ratios = full_values / full_values.sum()
            for pov in (0.1, 0.9, 0.99):
                with self.subTest(shape=shape, pov=pov):
                    scores, values, vectors = self.tool._perform_PCA(X, pov=pov)
                    count = int(np.searchsorted(np.cumsum(ratios), pov)) + 1
                    self.assertEqual(scores.shape, (shape[0], count))
                    np.testing.assert_allclose(values, full_values[:count], atol=1e-12)
                    np.testing.assert_allclose(vectors.T @ vectors, np.eye(count), atol=1e-12)
                    np.testing.assert_allclose(scores, (X - X.mean(axis=0)) @ vectors)
                    np.testing.assert_array_equal(X, original)
                    self.assertGreaterEqual(self.tool.explained_variance_ratio_.sum(), pov)

    def test_full_pov_reconstructs_training_images_and_fixed_count_still_works(self):
        X = self.rng.normal(size=(6, 20))
        scores, _, vectors = self.tool._perform_PCA(X, pov=1.0)
        np.testing.assert_allclose(scores @ vectors.T + self.tool.mean_, X, atol=1e-12)
        self.assertAlmostEqual(self.tool.explained_variance_ratio_.sum(), 1)
        self.assertEqual(self.tool._perform_PCA(X, n_components=2)[0].shape, (6, 2))

    def test_invalid_inputs_and_options_are_rejected(self):
        for X in (np.zeros((1, 4)), np.zeros((3, 0)), np.zeros(5), np.full((3, 4), np.nan), np.ones((3, 4))):
            with self.subTest(X=X), self.assertRaises(ValueError):
                self.tool._perform_PCA(X)
        X = self.rng.normal(size=(5, 6))
        for options in ({"pov": 0}, {"pov": 95}, {"pov": np.nan}, {"n_components": 0},
                        {"n_components": 5}, {"n_components": 1.5}, {"n_components": True},
                        {"n_components": 2, "pov": 0.9}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.tool._perform_PCA(X, **options)

    def _write_images(self, folder, values):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, value in enumerate(values):
            path = folder / f"{index}.png"
            Image.new("RGB", (4, 3), (value, value // 2, 255 - value)).save(path)
            paths.append(path.resolve())
        return paths

    def test_flat_training_folder_runs_pca_and_displays_k_without_extra_steps(self):
        with TemporaryDirectory() as folder:
            paths = self._write_images(folder, [10, 30, 70, 90])
            document = ImageDocument()
            document.current = Image.new("RGB", (5, 5), "red")
            original = document.current.tobytes()
            with patch(f"{self.module}.filedialog.askdirectory", return_value=folder) as choose_folder, \
                 patch(f"{self.module}.simpledialog.askfloat", return_value=95.0) as choose_pov, \
                 patch(f"{self.module}.simpledialog.askinteger", return_value=1) as choose_k, \
                 patch(f"{self.module}.filedialog.asksaveasfilename") as save, \
                 patch.object(self.tool, "_visualize_eigenfaces") as visualize:
                result = self.tool.run(None, document)
            choose_folder.assert_called_once()
            choose_pov.assert_called_once()
            choose_k.assert_called_once()
            save.assert_not_called()
            visualize.assert_called_once_with(1)
            self.assertEqual(self.tool.loaded_paths_, paths)
            self.assertEqual(result.details["Training images"], 4)
            self.assertEqual(self.tool.coefficients_.shape[0], 4)
            self.assertGreaterEqual(self.tool.explained_variance_ratio_.sum(), 0.95)
            self.assertIsNone(result.image)
            self.assertEqual(document.current.tobytes(), original)
            self.assertFalse(document.can_undo)

    def test_training_folder_also_accepts_subfolders_without_labels(self):
        with TemporaryDirectory() as folder:
            paths = self._write_images(folder, [10, 30])
            paths += self._write_images(Path(folder) / "more_images", [70, 90])
            with patch.object(self.tool, "_choose_set", return_value=folder), \
                 patch.object(self.tool, "_choose_pov", return_value=0.95), \
                 patch(f"{self.module}.simpledialog.askinteger", return_value=1), \
                 patch.object(self.tool, "_visualize_eigenfaces"):
                result = self.tool.run(None, ImageDocument())
            self.assertEqual(result.details["Training images"], 4)
            self.assertEqual(set(self.tool.loaded_paths_), set(paths))

    def test_loading_skips_unreadable_files_and_preserves_row_correspondence(self):
        with TemporaryDirectory() as folder:
            paths = self._write_images(folder, [10, 20, 40])
            broken = Path(folder) / "broken.png"
            broken.write_text("broken")
            X = self.tool._load_images([paths[0], broken, *paths[1:]])
            self.assertEqual(X.shape, (3, 36))
            self.assertEqual(self.tool.image_shape_, (3, 4, 3))
            self.assertEqual(self.tool.loaded_paths_, paths)
            for row, path in zip(X, paths):
                with Image.open(path) as image:
                    np.testing.assert_array_equal(row, np.asarray(image.convert("RGB")).ravel())

    def test_cancellation_at_each_dialog_stops_before_the_next_step(self):
        with TemporaryDirectory() as folder:
            self._write_images(folder, [10, 30])
            for selected, pov, k in (("", 0.95, 1), (folder, None, 1), (folder, 0.95, None)):
                with self.subTest(folder=selected, pov=pov, k=k), \
                     patch.object(self.tool, "_choose_set", return_value=selected), \
                     patch.object(self.tool, "_choose_pov", return_value=pov) as choose_pov, \
                     patch(f"{self.module}.simpledialog.askinteger", return_value=k) as choose_k, \
                     patch.object(self.tool, "_visualize_eigenfaces") as visualize:
                    self.assertIsNone(self.tool.run(None, ImageDocument()))
                    visualize.assert_not_called()
                    if not selected:
                        choose_pov.assert_not_called()
                    if not selected or pov is None:
                        choose_k.assert_not_called()

    def test_visualization_renders_first_k_components(self):
        self.tool._perform_PCA(self.rng.normal(size=(8, 36)), pov=1.0)
        self.tool.image_shape_ = (3, 4, 3)
        captured = []

        def inspect_gallery():
            figure = plt.gcf()
            shown = [axis.images[0].get_array() for axis in figure.axes if axis.images]
            self.assertEqual(len(shown), 5)
            for index, image in enumerate(shown):
                component = self.tool.components_[index].reshape(3, 4, 3)
                np.testing.assert_allclose(image, (component - component.min()) / np.ptp(component))
            captured.append(figure.number)

        with patch(f"{self.module}.plt.show", side_effect=inspect_gallery):
            self.tool._visualize_eigenfaces(5)
        self.assertFalse(plt.fignum_exists(captured[0]))

    def test_eigenfaces_is_registered_and_does_not_require_a_working_image(self):
        tool = next(tool for tool in build_tool_registry().all() if tool.tool_id == "eigenfaces")
        self.assertIsInstance(tool, EigenFaces)
        self.assertEqual(tool.category, "Set6")
        self.assertFalse(tool.requires_image)


if __name__ == "__main__":
    unittest.main()
