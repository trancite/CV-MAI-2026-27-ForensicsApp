"""Global BoW vocabulary, histograms, and chi-squared SVM classification."""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import chi2_kernel

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.set7.BoW import BagOfWords
from forensics_app.tools.set7.histogram_construction import HistogramConstruction
from forensics_app.tools.set7.classification import Classification


class BagOfWordsTests(unittest.TestCase):
    module = "forensics_app.tools.set7.bow_base"

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
        np.testing.assert_array_equal(self.tool.features_["cats"][0][1], extracted[0][0])
        np.testing.assert_array_equal(self.tool.features_["cats"][1][2], second)
        self.assertEqual(self.tool.features_["empty"], [])
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
                self.assertEqual(sum(len(points) for _, points, _ in self.tool.features_["cats"]), len(descriptors))
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

    @contextmanager
    def _workflow(self, images, algorithm="SIFT", parameters=None, n_words=2):
        parameters = parameters or {"n_octaves": 2, "n_scales": 3, "sigma_min": 1.6}
        with patch.object(self.tool, "_choose_algorithm", return_value=algorithm), \
             patch.object(self.tool, "_choose_parameters", return_value=parameters), \
             patch(f"{self.module}.filedialog.askdirectory", return_value="training"), \
             patch("forensics_app.tools.set7.bow_base.simpledialog.askinteger", return_value=n_words), \
             patch.object(self.tool, "_return_classes_images", return_value=images):
            yield

    def _test_document(self):
        document = ImageDocument()
        document.path = Path("selected-test.png")
        document.current = Image.new("RGB", (40, 30), (10, 30, 50))
        return document

    def test_test_file_is_excluded_from_class_training_images(self):
        with TemporaryDirectory() as folder:
            person = Path(folder) / "cats"
            person.mkdir()
            selected = person / "test.png"
            other = person / "train.png"
            Image.new("RGB", (30, 30), "red").save(selected)
            Image.new("RGB", (30, 30), "blue").save(other)
            images = self.tool._return_classes_images(None, folder, exclude=selected.resolve())
            self.assertEqual(len(images["cats"]), 1)
            np.testing.assert_array_equal(images["cats"][0][0, 0], [0, 0, 255])

    def test_global_words_use_the_closest_patch_across_all_classes(self):
        cat = self.rng.integers(1, 256, (50, 50, 3), dtype=np.uint8)
        dog = self.rng.integers(1, 256, (50, 50, 3), dtype=np.uint8)
        classes = {"cats": np.array([[0, 0], [8, 8]], dtype=np.float32),
                   "dogs": np.array([[10, 10], [20, 20]], dtype=np.float32)}
        self.tool.features_ = {"cats": [(cat, np.array([[7, 3], [10, 20]]), classes["cats"])],
                               "dogs": [(dog, np.array([[20, 24], [40, 8]]), classes["dogs"])]}
        self.tool.centroids_ = np.array([[20, 19], [0, 0], [7, 8], [11, 10]], dtype=np.float32)
        patches = self.tool._representative_patches(classes, patch_size=10)
        self.assertEqual(len(patches), 4)
        self.assertEqual(patches[0].getpixel((5, 5)), tuple(dog[40, 8]))
        self.assertEqual(patches[1].getpixel((5, 5)), tuple(cat[7, 3]))
        self.assertEqual(patches[2].getpixel((5, 5)), tuple(cat[10, 20]))
        self.assertEqual(patches[3].getpixel((5, 5)), tuple(dog[20, 24]))
        self.assertEqual(patches[1].getpixel((0, 5)), (0, 0, 0))

    def test_one_global_codebook_and_exact_requested_total_words(self):
        images = {"cats": [self.rng.integers(0, 256, (128, 192, 3), dtype=np.uint8)],
                  "dogs": [self.rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)]}
        displayed = []
        histograms = {}
        before = set(plt.get_fignums())

        def inspect_galleries():
            for number in set(plt.get_fignums()) - before:
                figure = plt.figure(number)
                for axis in figure.axes:
                    if axis.images:
                        self.assertEqual(figure._suptitle.get_text(), "10 visual words")
                        displayed.append(int(axis.get_title()))
                        self.assertEqual(axis.images[0].get_array().shape, (32, 32, 3))
                    elif axis.patches:
                        histograms[axis.get_ylabel()] = [bar.get_height() for bar in axis.patches]

        with self._workflow(images, "ORB", {"n_keypoints": 30}, n_words=10), \
             patch.object(self.tool, "perform_k_means", wraps=self.tool.perform_k_means) as fit, \
             patch("matplotlib.pyplot.show", side_effect=inspect_galleries):
            result = self.tool.run(None, ImageDocument())
        fit.assert_called_once()
        pooled = np.vstack([d for features in self.tool.features_.values() for _, _, d in features])
        np.testing.assert_array_equal(fit.call_args.args[0], pooled)
        self.assertEqual(pooled.shape, (60, 256))
        self.assertEqual(fit.call_args.args[1], 10)
        self.assertEqual(self.tool.n_words_, 10)
        self.assertEqual(self.tool.centroids_.shape, (10, 256))
        np.testing.assert_array_equal(self.tool.centroids_, self.tool.codebook_.cluster_centers_)
        self.assertEqual(displayed, list(range(1, 11)))
        self.assertEqual(set(histograms), {"cats", "dogs"})
        for name, histogram in histograms.items():
            np.testing.assert_allclose(histogram, self.tool.class_histograms_[name])
            self.assertEqual(len(histogram), 10)
            self.assertAlmostEqual(sum(histogram), 1.0)
        self.assertEqual(result.details, {"Classes": 2, "Visual words": 10})
        self.assertEqual(set(plt.get_fignums()), before)
        self.assertIsNone(result.image)

    def test_featureless_training_is_rejected_without_opening_a_gallery(self):
        images = {"blank": [np.zeros((128, 128, 3), dtype=np.uint8)], "empty": []}
        with self._workflow(images, "ORB", {"n_keypoints": 30}), \
             patch.object(self.tool, "_visualize_visual_words") as show:
            with self.assertRaises(ValueError):
                self.tool.run(None, ImageDocument())
        show.assert_not_called()

    def test_each_tool_cancels_before_loading_training_images(self):
        for tool_class in (BagOfWords, HistogramConstruction, Classification):
            for algorithm, parameters, folder in ((None, {}, "training"),
                                                   ("ORB", None, "training"), ("ORB", {}, "")):
                tool = tool_class()
                document = self._test_document() if tool.requires_image else ImageDocument()
                with self.subTest(tool=tool.title, algorithm=algorithm, parameters=parameters, folder=folder), \
                     patch.object(tool, "_choose_algorithm", return_value=algorithm), \
                     patch.object(tool, "_choose_parameters", return_value=parameters), \
                     patch(f"{self.module}.filedialog.askdirectory", return_value=folder), \
                     patch("forensics_app.tools.set7.bow_base.simpledialog.askinteger", return_value=10), \
                     patch.object(tool, "_return_classes_images") as load:
                    self.assertIsNone(tool.run(None, document))
                    load.assert_not_called()

    def test_total_word_count_question_defaults_to_one_hundred_and_supports_cancel(self):
        for tool_class in (BagOfWords, HistogramConstruction, Classification):
            tool = tool_class()
            document = self._test_document() if tool.requires_image else ImageDocument()
            with self.subTest(tool=tool.title), \
                 patch.object(tool, "_choose_algorithm", return_value="ORB"), \
                 patch.object(tool, "_choose_parameters", return_value={"n_keypoints": 30}), \
                 patch(f"{self.module}.filedialog.askdirectory", return_value="training"), \
                 patch(f"{self.module}.simpledialog.askinteger", return_value=None) as choose_count, \
                 patch.object(tool, "_return_classes_images") as load:
                self.assertIsNone(tool.run(None, document))
            choose_count.assert_called_once()
            self.assertEqual(choose_count.call_args.kwargs["initialvalue"], 100)
            self.assertEqual(choose_count.call_args.kwargs["minvalue"], 1)
            load.assert_not_called()

    def test_too_few_distinct_features_does_not_silently_show_fewer_words(self):
        images = {"cats": [np.zeros((40, 50, 3), dtype=np.uint8)]}
        descriptors = np.array([np.full(128, value) for value in (0, 0, 10)], dtype=np.uint8)
        with self._workflow(images, n_words=10), \
             patch.object(self.tool, "_generate_keypoints_and_descriptors", return_value=(np.zeros((3, 2)), descriptors)), \
             patch.object(self.tool, "_visualize_visual_words") as show:
            result = self.tool.run(None, ImageDocument())
        self.assertIn("2", result.message)
        self.assertIn("smaller", result.message)
        show.assert_not_called()
        self.assertIsNone(result.image)

    def test_histogram_counts_closest_centroids_and_retains_zero_bins(self):
        centroids = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float32)
        descriptors = np.array([[0, 1], [9, 10], [10, 11], [11, 10]], dtype=np.float32)
        histogram = self.tool._frequency_histogram(descriptors, centroids)
        np.testing.assert_array_equal(histogram, [1, 3, 0])
        self.assertEqual(histogram.sum(), len(descriptors))
        np.testing.assert_array_equal(self.tool._frequency_histogram(np.empty((0, 2)), centroids), [0, 0, 0])

    def test_class_histograms_average_normalized_images_in_the_global_vocabulary(self):
        self.tool.centroids_ = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float32)
        self.tool.features_ = {
            "cats": [(None, None, np.array([[0, 0]])),
                     (None, None, np.array([[10, 10]] * 3))],
            "dogs": [(None, None, np.array([[20, 20]]))],
            "blank": [(None, None, np.empty((0, 2)))],
            "empty": [],
        }
        histograms = self.tool._class_histograms()
        self.assertEqual(set(histograms), {"cats", "dogs", "blank"})
        np.testing.assert_allclose(histograms["cats"], [0.5, 0.5, 0])
        np.testing.assert_allclose(histograms["dogs"], [0, 0, 1])
        np.testing.assert_array_equal(histograms["blank"], [0, 0, 0])

    def test_histogram_and_classification_require_a_test_image(self):
        for tool_class in (HistogramConstruction, Classification):
            tool = tool_class()
            with self.subTest(tool=tool.title), patch.object(tool, "_choose_algorithm") as choose:
                result = tool.run(None, ImageDocument())
                self.assertIn("Open a test image first", result.message)
                choose.assert_not_called()

    def test_histogram_uses_global_codebook_and_current_grayscale_image(self):
        self.tool = HistogramConstruction()
        images = {"cats": [np.zeros((40, 50, 3), dtype=np.uint8)], "dogs": [np.zeros((50, 60, 3), dtype=np.uint8)]}
        batches = [np.array([np.full(128, x) for x in values], dtype=np.uint8)
                   for values in ((0, 2), (20, 22), (0, 1, 2, 22))]
        features = [(np.zeros((len(batch), 2)), batch) for batch in batches]
        document = self._test_document()
        original = document.current.tobytes()
        shown = []

        def inspect_histogram():
            axis = plt.gcf().axes[0]
            self.assertEqual(axis.get_title(), "selected-test.png")
            shown.extend(bar.get_height() for bar in axis.patches)

        with self._workflow(images), \
             patch.object(self.tool, "_generate_keypoints_and_descriptors", side_effect=features) as extract, \
             patch.object(self.tool, "perform_k_means", wraps=self.tool.perform_k_means) as fit, \
             patch("matplotlib.pyplot.show", side_effect=inspect_histogram):
            result = self.tool.run(None, document)
        fit.assert_called_once()
        np.testing.assert_array_equal(fit.call_args.args[0], np.vstack(batches[:2]))
        self.assertEqual(fit.call_args.args[1], 2)
        np.testing.assert_array_equal(self.tool.histogram_[np.argsort(self.tool.centroids_.mean(axis=1))], [3, 1])
        np.testing.assert_array_equal(shown, self.tool.histogram_)
        np.testing.assert_array_equal(extract.call_args.args[0], np.asarray(document.current.convert("L")))
        self.assertEqual(result.details, {"Histogram dimensions": 2, "Descriptors": 4})
        self.assertEqual(document.current.tobytes(), original)
        self.assertIsNone(result.image)

    def test_classify_button_uses_l1_normalized_chi_squared_svm_and_displays_top_probability(self):
        self.tool = Classification()
        images = {"cats": [np.zeros((40, 50, 3), dtype=np.uint8) for _ in range(5)],
                  "dogs": [np.zeros((50, 60, 3), dtype=np.uint8) for _ in range(5)]}
        batches = [np.array([np.full(128, x) for x in (0, 1)], dtype=np.uint8) for _ in range(5)]
        batches += [np.array([np.full(128, x) for x in (20, 21)], dtype=np.uint8) for _ in range(5)]
        batches += [np.array([np.full(128, x) for x in (0, 0, 0, 1)], dtype=np.uint8)]
        features = [(np.zeros((len(batch), 2)), batch) for batch in batches]
        document = self._test_document()

        def click_classify():
            figure = plt.gcf()
            self.assertEqual(figure.axes[0].get_title(), "selected-test.png")
            figure.canvas.draw()
            x, y = figure.axes[1].transAxes.transform((0.5, 0.5))
            for name in ("button_press_event", "button_release_event"):
                event = MouseEvent(name, figure.canvas, x, y, button=1)
                figure.canvas.callbacks.process(name, event)
            self.assertIn("cats:", figure.axes[0].get_title())

        with self._workflow(images), \
             patch.object(self.tool, "_generate_keypoints_and_descriptors", side_effect=features), \
             patch("matplotlib.pyplot.show", side_effect=click_classify):
            result = self.tool.run(None, document)
        classifier = self.tool.classifier_
        self.assertIs(classifier.named_steps["svc"].kernel, chi2_kernel)
        self.assertTrue(classifier.named_steps["svc"].probability)
        self.assertEqual(classifier.named_steps["normalizer"].norm, "l1")
        self.assertEqual(classifier.named_steps["normalizer"].n_features_in_, len(self.tool.centroids_))
        self.assertEqual(result.details["Prediction"], "cats")
        self.assertGreater(result.details["Probability"], 0.5)
        probabilities = classifier.predict_proba(self.tool.histogram_[None, :])[0]
        self.assertAlmostEqual(result.details["Probability"], probabilities.max())
        self.assertEqual(self.tool.histogram_.sum(), 4)
        self.assertIsNone(result.image)

    def test_closing_classification_without_clicking_does_not_predict(self):
        self.tool = Classification()
        images = {"cats": [np.zeros((40, 50, 3), dtype=np.uint8) for _ in range(5)],
                  "dogs": [np.zeros((50, 60, 3), dtype=np.uint8) for _ in range(5)]}
        batches = [np.array([np.full(128, x) for x in (0, 1)], dtype=np.uint8) for _ in range(5)]
        batches += [np.array([np.full(128, x) for x in (20, 21)], dtype=np.uint8) for _ in range(5)]
        batches += [np.zeros((1, 128), dtype=np.uint8)]
        features = [(np.zeros((len(batch), 2)), batch) for batch in batches]
        with self._workflow(images), \
             patch.object(self.tool, "_generate_keypoints_and_descriptors", side_effect=features), \
             patch("forensics_app.tools.set7.classification.SVC.predict_proba") as predict, \
             patch("matplotlib.pyplot.show"):
            result = self.tool.run(None, self._test_document())
        predict.assert_not_called()
        self.assertNotIn("Prediction", result.details)

    def test_registry_has_three_distinct_set7_tools_without_an_options_menu(self):
        tools = {tool.tool_id: tool for tool in build_tool_registry().all()}
        expected = {
            "bagofwords": (BagOfWords, "Codebook Construction", False),
            "histogramconstruction": (HistogramConstruction, "Histogram Construction", True),
            "bowclassification": (Classification, "Classification", True),
        }
        self.assertEqual({tool.tool_id for tool in tools.values() if tool.category == "Set7"}, set(expected))
        for tool_id, (tool_class, title, requires_image) in expected.items():
            tool = tools[tool_id]
            with self.subTest(tool=title):
                self.assertIsInstance(tool, tool_class)
                self.assertEqual(tool.title, title)
                self.assertEqual(tool.requires_image, requires_image)
                self.assertFalse(hasattr(tool, "options"))
                self.assertFalse(hasattr(tool, "_choose_option"))
        self.assertIn("face_recognition", tools)
        from forensics_app.app import main
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
