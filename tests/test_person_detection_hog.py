"""Sliding-window HOG matching and selection workflow without GUI windows."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage.transform import resize

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.set4.person_detection_hog import PersonDetectionHog


class PersonDetectionHogTests(unittest.TestCase):
    module = "forensics_app.tools.set4.person_detection_hog"

    def setUp(self):
        self.tool = PersonDetectionHog()
        self.generator = np.random.default_rng(42)
        self.image = self.generator.integers(0, 256, (12, 14, 3), dtype=np.uint8)
        self.person = self.image[6:, 8:].copy()
        self.parameters = ((2, 2), (1, 1), 9)

    def test_all_metrics_find_the_last_valid_window_using_hog_descriptors(self):
        for name, metric in self.tool.metric_options.items():
            with self.subTest(metric=name):
                maximize = name in ("Dot product", "Minimum")
                score, position = self.tool._compute_differences(
                    self.image, self.person, self.parameters, metric, maximize=maximize
                )
                self.assertEqual(position, (6, 8))
                descriptor = self.tool._compute_template_hog(self.person, self.parameters)
                self.assertAlmostEqual(score, metric(descriptor, descriptor))

    def test_template_equal_to_image_produces_one_window(self):
        score, position = self.tool._compute_differences(
            self.person, self.person, self.parameters, self.tool.metric_options["Euclidean"]
        )
        self.assertEqual((score, position), (0.0, (0, 0)))

    def test_one_pixel_cells_are_safe_without_hog_visualization(self):
        score, position = self.tool._compute_differences(
            self.person, self.person, ((1, 1), (1, 1), 9), self.tool.metric_options["Euclidean"]
        )
        self.assertEqual((score, position), (0.0, (0, 0)))

    def test_oversized_template_and_hog_blocks_are_rejected(self):
        metric = self.tool.metric_options["Euclidean"]
        for shape in ((13, 6, 3), (6, 15, 3)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "larger"):
                self.tool._compute_differences(self.image, np.zeros(shape), self.parameters, metric)
        with self.assertRaisesRegex(ValueError, "at least 8 x 8"):
            self.tool._compute_differences(self.image, self.person, ((4, 4), (2, 2), 9), metric)

    def test_default_workflow_marks_gray_and_color_images_without_mutation(self):
        for mode in ("L", "RGB"):
            with self.subTest(mode=mode), TemporaryDirectory() as folder:
                shape = (26, 28) if mode == "L" else (26, 28, 3)
                source = Image.fromarray(self.generator.integers(0, 256, shape, dtype=np.uint8))
                document = ImageDocument()
                document.current = source.copy()
                filename = str(Path(folder) / "person.png")
                source.crop((4, 2, 28, 26)).save(filename)
                with patch(f"{self.module}.filedialog.askopenfilename", return_value=filename), \
                     patch(f"{self.module}.dialog_options", side_effect=["No", "Euclidean"]), \
                     patch(f"{self.module}.simpledialog.askinteger") as ask_parameter, \
                     patch(f"{self.module}.simpledialog.askfloat", return_value=1.0):
                    result = self.tool.run(None, document)
                ask_parameter.assert_not_called()
                self.assertEqual(result.details["Top-left (x, y)"], (4, 2))
                self.assertEqual(result.details["Windows compared"], 15)
                self.assertEqual(result.details["Best score"], 0)
                self.assertEqual(result.details["Pixels per cell"], (8, 8))
                self.assertEqual(result.details["Background scale"], 1.0)
                self.assertEqual(result.image.mode, "RGB")
                self.assertEqual(result.image.size, source.size)
                self.assertEqual(result.image.getpixel((4, 2)), (255, 0, 0))
                self.assertEqual(result.image.getpixel((27, 25)), (255, 0, 0))
                self.assertEqual(result.image.getpixel((0, 0)), source.convert("RGB").getpixel((0, 0)))
                self.assertEqual(document.current.mode, mode)
                self.assertEqual(document.current.tobytes(), source.tobytes())

    def test_custom_workflow_selects_minimum_distance_or_maximum_similarity(self):
        with TemporaryDirectory() as folder:
            filename = str(Path(folder) / "person.png")
            Image.fromarray(self.person).save(filename)
            for name in ("Euclidean", "Dot product", "Minimum"):
                with self.subTest(metric=name):
                    document = ImageDocument()
                    document.current = Image.fromarray(self.image)
                    with patch(f"{self.module}.filedialog.askopenfilename", return_value=filename), \
                         patch(f"{self.module}.dialog_options", side_effect=["Yes", name]), \
                         patch(f"{self.module}.simpledialog.askinteger", side_effect=[2, 1, 9]), \
                         patch(f"{self.module}.simpledialog.askfloat", return_value=1.0):
                        result = self.tool.run(None, document)
                    self.assertEqual(result.details["Top-left (x, y)"], (8, 6))
                    selection = "Minimum distance" if name == "Euclidean" else "Maximum similarity"
                    self.assertEqual(result.details["Selection"], selection)
                    self.assertEqual(result.image.getpixel((8, 6)), (255, 0, 0))

    def test_cancellation_at_each_selection_keeps_the_document_unchanged(self):
        with TemporaryDirectory() as folder:
            filename = str(Path(folder) / "person.png")
            Image.fromarray(self.person).save(filename)
            cases = [
                ("", [], [], 1.0),
                (filename, [None], [], 1.0),
                (filename, ["Yes"], [None], 1.0),
                (filename, ["Yes"], [2, None], 1.0),
                (filename, ["Yes"], [2, 1, None], 1.0),
                (filename, ["No", None], [], 1.0),
                (filename, ["No", "Euclidean"], [], None),
            ]
            for path, choices, parameters, scale in cases:
                with self.subTest(path=path, choices=choices, parameters=parameters):
                    document = ImageDocument()
                    document.current = Image.fromarray(self.image)
                    with patch(f"{self.module}.filedialog.askopenfilename", return_value=path), \
                         patch(f"{self.module}.dialog_options", side_effect=choices), \
                         patch(f"{self.module}.simpledialog.askinteger", side_effect=parameters), \
                         patch(f"{self.module}.simpledialog.askfloat", return_value=scale):
                        self.assertIsNone(self.tool.run(None, document))
                    np.testing.assert_array_equal(np.asarray(document.current), self.image)

    def test_detection_tool_is_available_in_the_application_registry(self):
        self.assertTrue(any(isinstance(tool, PersonDetectionHog) for tool in build_tool_registry().all()))

    def test_scaled_match_forwards_minimum_or_maximum_selection(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        person = np.zeros((4, 4, 3), dtype=np.uint8)
        for maximize in (False, True):
            with self.subTest(maximize=maximize):
                with patch.object(self.tool, "_compute_differences", return_value=(0.3, (2, 3))) as compute:
                    match = self.tool._compute_scaled_match(
                        image, person, 0.5, self.parameters,
                        self.tool.metric_options["Euclidean"], maximize,
                    )
                compute.assert_called_once()
                self.assertEqual(compute.call_args.kwargs["maximize"], maximize)
                self.assertEqual(match, (0.3, (4, 6), (8, 8), 49))

    def test_scaled_match_preserves_range_and_uses_actual_dimensions_for_box(self):
        image = np.full((7, 11, 3), 200, dtype=np.uint8)
        person = np.zeros((3, 4, 3), dtype=np.uint8)
        with patch.object(self.tool, "_compute_differences", return_value=(0.25, (4, 8))) as compute:
            match = self.tool._compute_scaled_match(
                image, person, 1.5, self.parameters, self.tool.metric_options["Euclidean"], False,
            )
        scaled = compute.call_args.args[0]
        self.assertEqual(scaled.shape, (10, 16, 3))
        np.testing.assert_allclose(scaled, 200)
        self.assertEqual(match, (0.25, (2, 5), (3, 4), 104))

    def test_scaled_match_can_upscale_an_undersized_source(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        person = np.zeros((6, 6, 3), dtype=np.uint8)
        with patch.object(self.tool, "_compute_differences", return_value=(0, (2, 2))) as compute:
            match = self.tool._compute_scaled_match(
                image, person, 2.0, self.parameters, self.tool.metric_options["Euclidean"], False,
            )
        compute.assert_called_once()
        self.assertEqual(compute.call_args.args[0].shape, (8, 8, 3))
        self.assertEqual(match, (0, (1, 1), (3, 3), 9))

    def test_scaled_match_rejects_too_small_background_and_invalid_factors(self):
        with self.assertRaisesRegex(ValueError, "scaled background"):
            self.tool._compute_scaled_match(
                self.image, self.person, 0.25, self.parameters, self.tool.metric_options["Euclidean"], False,
            )
        for scale in (0, -1, np.nan, np.inf):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "positive and finite"):
                self.tool._compute_scaled_match(
                    self.image, self.person, scale, self.parameters, self.tool.metric_options["Euclidean"], False,
                )

    def test_run_searches_one_selected_scale_and_draws_on_original(self):
        document = ImageDocument()
        document.current = Image.fromarray(np.full((7, 11, 3), 200, dtype=np.uint8))
        before = document.current.tobytes()
        person = np.zeros((3, 4, 3), dtype=np.uint8)
        with patch.object(self.tool, "_load_image_person", return_value=person), \
             patch.object(self.tool, "_choose_hog_parameters", return_value=((1, 1), (1, 1), 9)), \
             patch.object(self.tool, "_choose_metric", return_value=self.tool.metric_options["Euclidean"]), \
             patch(f"{self.module}.simpledialog.askfloat", return_value=1.5), \
             patch.object(self.tool, "_compute_differences", return_value=(0.1, (4, 8))) as compute:
            result = self.tool.run(None, document)
        compute.assert_called_once()
        self.assertEqual(result.details["Background scale"], 1.5)
        self.assertEqual(result.details["Top-left (x, y)"], (5, 2))
        self.assertEqual(result.details["Window size"], (4, 3))
        self.assertEqual(result.image.size, (11, 7))
        self.assertEqual(result.image.getpixel((5, 2)), (255, 0, 0))
        self.assertEqual(result.image.getpixel((8, 4)), (255, 0, 0))
        self.assertEqual(result.image.getpixel((0, 0)), (200, 200, 200))
        self.assertEqual(document.current.tobytes(), before)

    def test_real_fractional_scale_workflow_finds_the_resized_template(self):
        image = np.random.default_rng(2026).integers(0, 256, (8, 10, 3), dtype=np.uint8)
        enlarged = resize(image, (12, 15, 3), preserve_range=True, anti_aliasing=False)
        person = np.rint(enlarged[6:, 9:]).astype(np.uint8)
        document = ImageDocument()
        document.current = Image.fromarray(image)
        with TemporaryDirectory() as folder:
            filename = str(Path(folder) / "person.png")
            Image.fromarray(person).save(filename)
            with patch(f"{self.module}.filedialog.askopenfilename", return_value=filename), \
                 patch(f"{self.module}.dialog_options", side_effect=["Yes", "Euclidean"]), \
                 patch(f"{self.module}.simpledialog.askinteger", side_effect=[2, 1, 9]), \
                 patch(f"{self.module}.simpledialog.askfloat", return_value=1.5):
                result = self.tool.run(None, document)
        self.assertEqual(result.details["Background scale"], 1.5)
        self.assertEqual(result.details["Top-left (x, y)"], (6, 4))
        self.assertEqual(result.details["Window size"], (4, 4))
        self.assertEqual(result.details["Windows compared"], 70)
        self.assertLess(result.details["Best score"], 0.02)
        self.assertEqual(result.image.getpixel((6, 4)), (255, 0, 0))
        self.assertEqual(result.image.getpixel((9, 7)), (255, 0, 0))
        np.testing.assert_array_equal(np.asarray(document.current), image)


if __name__ == "__main__":
    unittest.main()
