"""Per-class codebooks, descriptor provenance, and real visual-word patches."""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.set7.BoW import BagOfWords


class BagOfWordsTests(unittest.TestCase):
    module = "forensics_app.tools.set7.BoW"

    def setUp(self):
        self.tool = BagOfWords()
        self.rng = np.random.default_rng(42)

    def test_descriptor_rows_keep_their_image_and_keypoint_correspondence(self):
        images = {"cats": [np.zeros((40, 60, 3)), np.zeros((70, 50, 3))], "empty": []}
        first = np.full((2, 128), 10, dtype=np.uint8)
        second = np.full((1, 128), 20, dtype=np.uint8)
        extracted = [(np.array([[10, 20], [15, 30]]), first), (np.array([[25, 35]]), second)]
        with patch.object(self.tool, "_generate_keypoints_and_descriptors", side_effect=extracted) as extract:
            descriptors = self.tool._classes_and_descriptors(images, "SIFT", {})
        np.testing.assert_array_equal(descriptors["cats"], np.vstack((first, second)))
        np.testing.assert_array_equal(self.tool.descriptor_sources_["cats"], [[0, 10, 20], [0, 15, 30], [1, 25, 35]])
        self.assertEqual(descriptors["empty"].shape, (0, 128))
        for call in extract.call_args_list:
            self.assertEqual(call.args[0].ndim, 2)
            self.assertEqual(call.args[0].dtype, np.uint8)

    def test_real_orb_and_sift_extract_from_rgb_images_of_different_sizes(self):
        images = [self.rng.integers(0, 256, shape, dtype=np.uint8) for shape in ((128, 192, 3), (96, 128, 3))]
        before = [image.copy() for image in images]
        for algorithm, parameters, width in (
            ("ORB", {"n_keypoints": 30}, 256),
            ("SIFT", {"n_octaves": 2, "n_scales": 3, "sigma_min": 1.6}, 128),
        ):
            with self.subTest(algorithm=algorithm):
                descriptors = self.tool._classes_and_descriptors({"cats": images}, algorithm, parameters)["cats"]
                self.assertEqual(descriptors.shape[1], width)
                self.assertGreater(len(descriptors), 0)
                self.assertEqual(len(self.tool.descriptor_sources_["cats"]), len(descriptors))
                for original, current in zip(before, images):
                    np.testing.assert_array_equal(current, original)

    def test_classes_keep_original_image_sizes_and_skip_unreadable_files(self):
        with TemporaryDirectory() as folder:
            first = Path(folder) / "first"
            first.mkdir()
            Image.new("RGB", (40, 60), "red").save(first / "a.png")
            Image.new("L", (70, 50), 120).save(first / "b.png")
            (first / "broken.png").write_text("not an image")
            (Path(folder) / "empty").mkdir()
            (Path(folder) / "__MACOSX").mkdir()
            classes = self.tool._return_classes_images(None, folder)
            self.assertEqual(set(classes), {"first", "empty"})
            self.assertEqual([image.shape for image in classes["first"]], [(60, 40, 3), (50, 70, 3)])
            self.assertEqual(classes["empty"], [])

    def test_kmeans_uses_descriptor_rows_and_caps_words_at_distinct_samples(self):
        descriptors = np.array([[0, 0], [0, 0], [10, 10], [10, 10]], dtype=np.float32)
        model = self.tool.perform_k_means(descriptors, ncl=30)
        self.assertEqual(model.n_clusters, 2)
        self.assertEqual(model.cluster_centers_.shape, (2, 2))
        np.testing.assert_allclose(np.sort(model.cluster_centers_, axis=0), [[0, 0], [10, 10]], atol=1e-5)

    def test_invalid_kmeans_input_has_a_clear_error(self):
        for values, count in ((np.empty((0, 128)), 3), (np.ones(128), 3),
                              (np.ones((3, 2)), 0),
                              (np.full((3, 2), np.nan), 1)):
            with self.subTest(values=values, count=count), self.assertRaises(ValueError):
                self.tool.perform_k_means(values, count)

    def test_each_centroid_is_visualized_using_its_nearest_observed_patch(self):
        images = [self.rng.integers(1, 256, (50, 50, 3), dtype=np.uint8) for _ in range(2)]
        descriptors = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float32)
        sources = np.array([[0, 7, 3], [1, 20, 24], [1, 40, 8]])
        model = SimpleNamespace(cluster_centers_=np.array([[9, 10], [0, 0]], dtype=np.float32))
        patches = self.tool._representative_patches(images, descriptors, sources, model, patch_size=10)
        self.assertEqual([image.size for image in patches], [(10, 10), (10, 10)])
        self.assertEqual(patches[0].getpixel((5, 5)), tuple(images[1][20, 24]))
        self.assertEqual(patches[1].getpixel((5, 5)), tuple(images[0][7, 3]))
        self.assertEqual(patches[1].getpixel((0, 5)), (0, 0, 0))

    def test_option_zero_builds_and_displays_codebooks_without_a_required_image(self):
        with TemporaryDirectory() as folder:
            for name, shapes in {"cats": [(128, 192, 3), (96, 128, 3)], "dogs": [(128, 128, 3)]}.items():
                class_folder = Path(folder) / name
                class_folder.mkdir()
                for index, shape in enumerate(shapes):
                    Image.fromarray(self.rng.integers(0, 256, shape, dtype=np.uint8)).save(class_folder / f"{index}.png")
            (Path(folder) / "empty").mkdir()
            document = ImageDocument()
            before = set(plt.get_fignums())
            displayed = []

            def inspect_galleries():
                for number in sorted(set(plt.get_fignums()) - before):
                    figure = plt.figure(number)
                    displayed.append(figure._suptitle.get_text())
                    patches = [axis.images[0].get_array() for axis in figure.axes if axis.images]
                    self.assertEqual(len(patches), 2)
                    self.assertTrue(all(image.shape == (32, 32, 3) for image in patches))

            with patch.object(self.tool, "_choose_option", return_value=self.tool.options[0]), \
                 patch.object(self.tool, "_choose_algorithm", return_value="ORB"), \
                 patch.object(self.tool, "_choose_parameters", return_value={"n_keypoints": 30}), \
                 patch.object(self.tool, "_choose_set", return_value=folder), \
                 patch(f"{self.module}.simpledialog.askinteger", return_value=2), \
                 patch(f"{self.module}.plt.show", side_effect=inspect_galleries):
                result = self.tool.run(None, document)
            self.assertEqual(set(self.tool.codebooks_), {"cats", "dogs"})
            self.assertEqual(set(displayed), {"cats: 2 visual words", "dogs: 2 visual words"})
            self.assertEqual(result.details["cats"], "2 images, 60 descriptors, 2 words")
            self.assertEqual(result.details["dogs"], "1 images, 30 descriptors, 2 words")
            self.assertEqual(result.details["Classes without descriptors"], "empty")
            self.assertEqual(set(plt.get_fignums()), before)
            self.assertIsNone(result.image)
            self.assertIsNone(document.current)
            self.assertFalse(document.can_undo)
            self.assertEqual(self.tool.algorithm_, "ORB")
            self.assertEqual(self.tool.parameters_, {"n_keypoints": 30})

    def test_featureless_classes_do_not_crash_or_open_an_empty_gallery(self):
        classes = {"blank": [np.zeros((128, 128, 3), dtype=np.uint8)], "empty": []}
        with patch.object(self.tool, "_choose_option", return_value=self.tool.options[0]), \
             patch.object(self.tool, "_choose_algorithm", return_value="ORB"), \
             patch.object(self.tool, "_choose_parameters", return_value={"n_keypoints": 30}), \
             patch.object(self.tool, "_choose_set", return_value="dataset"), \
             patch(f"{self.module}.simpledialog.askinteger", return_value=2), \
             patch.object(self.tool, "_return_classes_images", return_value=classes), \
             patch.object(self.tool, "_visualize_visual_words") as visualize:
            result = self.tool.run(None, ImageDocument())
        self.assertEqual(self.tool.codebooks_, {})
        self.assertEqual(result.details["Classes without descriptors"], "blank, empty")
        visualize.assert_not_called()

    def test_cancellation_at_each_selection_stops_before_loading(self):
        cases = [(None, "ORB", {}, "dataset", 2), (self.tool.options[0], None, {}, "dataset", 2),
                 (self.tool.options[0], "ORB", None, "dataset", 2),
                 (self.tool.options[0], "ORB", {}, "", 2), (self.tool.options[0], "ORB", {}, "dataset", None)]
        for option, algorithm, parameters, folder, words in cases:
            with self.subTest(option=option, algorithm=algorithm, parameters=parameters, folder=folder, words=words), \
                 patch.object(self.tool, "_choose_option", return_value=option), \
                 patch.object(self.tool, "_choose_algorithm", return_value=algorithm), \
                 patch.object(self.tool, "_choose_parameters", return_value=parameters), \
                 patch.object(self.tool, "_choose_set", return_value=folder), \
                 patch(f"{self.module}.simpledialog.askinteger", return_value=words), \
                 patch.object(self.tool, "_return_classes_images") as load:
                self.assertIsNone(self.tool.run(None, ImageDocument()))
                load.assert_not_called()

    def test_other_options_do_not_run_codebook_construction(self):
        for option in self.tool.options[2:]:
            with self.subTest(option=option), \
                 patch.object(self.tool, "_choose_option", return_value=option), \
                 patch.object(self.tool, "_choose_algorithm") as choose:
                result = self.tool.run(None, ImageDocument())
                self.assertIn("not been implemented", result.message)
                choose.assert_not_called()

    def test_histogram_counts_nearest_centroids_including_unused_words(self):
        centroids = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float32)
        descriptors = np.array([[0, 1], [9, 10], [10, 11], [11, 10]], dtype=np.float32)
        histogram = self.tool._frequency_histogram(descriptors, centroids)
        np.testing.assert_array_equal(histogram, [1, 3, 0])
        self.assertEqual(histogram.sum(), len(descriptors))
        np.testing.assert_array_equal(self.tool._frequency_histogram(np.empty((0, 2)), centroids), [0, 0, 0])

    def test_histogram_requires_an_open_image_before_asking_for_training(self):
        with patch.object(self.tool, "_choose_option", return_value=self.tool.options[1]), \
             patch.object(self.tool, "_choose_algorithm") as choose_algorithm, \
             patch.object(self.tool, "_choose_set") as choose_training:
            result = self.tool.run(None, ImageDocument())
        self.assertIn("Open a test image first", result.message)
        choose_algorithm.assert_not_called()
        choose_training.assert_not_called()

    def test_histogram_builds_kmeans_and_plots_the_current_image_from_a_fresh_tool(self):
        document = ImageDocument()
        document.path = Path("selected-test.png")
        document.current = Image.new("RGB", (40, 30), (10, 30, 50))
        original = document.current.tobytes()
        training = {"cats": [np.zeros((40, 50, 3), dtype=np.uint8)],
                    "dogs": [np.zeros((50, 60, 3), dtype=np.uint8)]}
        parameters = {"n_octaves": 2, "n_scales": 3, "sigma_min": 1.6}
        train_a = np.array([np.full(128, value) for value in (0, 10)], dtype=np.uint8)
        train_b = np.array([np.full(128, value) for value in (20, 30)], dtype=np.uint8)
        test = np.array([np.full(128, value) for value in (0, 9, 11, 30)], dtype=np.uint8)
        features = [(np.zeros((len(values), 2)), values) for values in (train_a, train_b, test)]
        before = set(plt.get_fignums())
        shown = []

        def inspect_histogram():
            numbers = set(plt.get_fignums()) - before
            self.assertEqual(len(numbers), 1)
            axis = plt.figure(numbers.pop()).axes[0]
            self.assertEqual(axis.get_title(), "selected-test.png")
            self.assertEqual(axis.get_xlabel(), "Visual word")
            self.assertEqual(axis.get_ylabel(), "Frequency")
            shown.extend(bar.get_height() for bar in axis.patches)

        self.assertFalse(hasattr(self.tool, "codebooks_"))
        with patch.object(self.tool, "_choose_option", return_value=self.tool.options[1]), \
             patch.object(self.tool, "_choose_algorithm", return_value="SIFT"), \
             patch.object(self.tool, "_choose_parameters", return_value=parameters), \
             patch.object(self.tool, "_choose_set", return_value="training"), \
             patch(f"{self.module}.simpledialog.askinteger", return_value=2), \
             patch.object(self.tool, "_return_classes_images", return_value=training), \
             patch.object(self.tool, "_generate_keypoints_and_descriptors", side_effect=features) as extract, \
             patch.object(self.tool, "perform_k_means", wraps=self.tool.perform_k_means) as fit, \
             patch.object(self.tool, "_representative_patches") as patches, \
             patch.object(self.tool, "_visualize_visual_words") as show_words, \
             patch(f"{self.module}.filedialog.askopenfilenames") as choose_files, \
             patch(f"{self.module}.plt.show", side_effect=inspect_histogram):
            result = self.tool.run(None, document)
        centroids = np.vstack([model.cluster_centers_ for model in self.tool.codebooks_.values()])
        order = np.argsort(centroids.mean(axis=1))
        np.testing.assert_array_equal(self.tool.histogram_[order], [1, 2, 0, 1])
        np.testing.assert_array_equal(shown, self.tool.histogram_)
        self.assertEqual(result.details, {"Histogram dimensions": 4, "Descriptors": 4})
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(extract.call_count, 3)
        np.testing.assert_array_equal(extract.call_args.args[0], np.asarray(document.current.convert("L")))
        self.assertEqual(extract.call_args.args[1:], ("SIFT", parameters))
        patches.assert_not_called()
        show_words.assert_not_called()
        choose_files.assert_not_called()
        self.assertEqual(set(plt.get_fignums()), before)
        self.assertEqual(document.current.tobytes(), original)
        self.assertFalse(document.can_undo)
        self.assertIsNone(result.image)

    def test_cancelling_histogram_training_selection_stops_before_loading_or_fitting(self):
        document = ImageDocument()
        document.current = Image.new("RGB", (40, 30), "red")
        with patch.object(self.tool, "_choose_option", return_value=self.tool.options[1]), \
             patch.object(self.tool, "_choose_algorithm", return_value="ORB"), \
             patch.object(self.tool, "_choose_parameters", return_value={"n_keypoints": 30}), \
             patch.object(self.tool, "_choose_set", return_value=""), \
             patch.object(self.tool, "_return_classes_images") as load, \
             patch.object(self.tool, "perform_k_means") as fit:
            self.assertIsNone(self.tool.run(None, document))
        load.assert_not_called()
        fit.assert_not_called()

    def test_bow_is_available_under_set7(self):
        tool = next(tool for tool in build_tool_registry().all() if tool.tool_id == "bagofwords")
        self.assertIsInstance(tool, BagOfWords)
        self.assertEqual(tool.category, "Set7")
        self.assertFalse(tool.requires_image)


if __name__ == "__main__":
    unittest.main()
