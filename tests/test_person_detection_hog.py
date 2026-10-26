"""Sliding-window HOG matching and selection workflow without GUI windows."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

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
                descriptor = self.tool._compute_hog(self.person, self.parameters)
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
                     patch(f"{self.module}.simpledialog.askinteger", return_value=1) as ask_step:
                    result = self.tool.run(None, document)
                ask_step.assert_called_once()
                self.assertEqual(ask_step.call_args.kwargs["minvalue"], 1)
                self.assertEqual(ask_step.call_args.kwargs["initialvalue"], 8)
                self.assertEqual(result.details["Top-left (x, y)"], (4, 2))
                self.assertEqual(result.details["Windows compared"], 15)
                self.assertEqual(result.details["Best score"], 0)
                self.assertEqual(result.details["Pixels per cell"], (8, 8))
                self.assertEqual(result.details["Step size"], 1)
                self.assertNotIn("Background scale", result.details)
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
            for name in self.tool.metric_options:
                with self.subTest(metric=name):
                    document = ImageDocument()
                    document.current = Image.fromarray(self.image)
                    with patch(f"{self.module}.filedialog.askopenfilename", return_value=filename), \
                         patch(f"{self.module}.dialog_options", side_effect=["Yes", name]), \
                         patch(f"{self.module}.simpledialog.askinteger", side_effect=[2, 1, 9, 2]):
                        result = self.tool.run(None, document)
                    self.assertEqual(result.details["Top-left (x, y)"], (8, 6))
                    selection = "Maximum similarity" if name in ("Dot product", "Minimum") else "Minimum distance"
                    self.assertEqual(result.details["Selection"], selection)
                    self.assertEqual(result.details["Step size"], 2)
                    self.assertEqual(result.details["Windows compared"], 20)
                    self.assertEqual(result.image.getpixel((8, 6)), (255, 0, 0))

    def test_cancellation_at_each_selection_keeps_the_document_unchanged(self):
        with TemporaryDirectory() as folder:
            filename = str(Path(folder) / "person.png")
            Image.fromarray(self.person).save(filename)
            cases = [
                ("", [], []),
                (filename, [None], []),
                (filename, ["Yes"], [None]),
                (filename, ["Yes"], [2, None]),
                (filename, ["Yes"], [2, 1, None]),
                (filename, ["No", None], []),
                (filename, ["No", "Euclidean"], [None]),
            ]
            for path, choices, parameters in cases:
                with self.subTest(path=path, choices=choices, parameters=parameters):
                    document = ImageDocument()
                    document.current = Image.fromarray(self.image)
                    with patch(f"{self.module}.filedialog.askopenfilename", return_value=path), \
                         patch(f"{self.module}.dialog_options", side_effect=choices), \
                         patch(f"{self.module}.simpledialog.askinteger", side_effect=parameters):
                        self.assertIsNone(self.tool.run(None, document))
                    np.testing.assert_array_equal(np.asarray(document.current), self.image)

    def test_detection_tool_is_available_in_the_application_registry(self):
        self.assertTrue(any(isinstance(tool, PersonDetectionHog) for tool in build_tool_registry().all()))

    def test_step_size_searches_the_grid_without_adding_unaligned_edges(self):
        cases = [(3, (6, 6), 9), (4, (4, 8), 6)]
        for step_size, position, windows in cases:
            with self.subTest(step_size=step_size):
                row, column = position
                person = self.image[row : row + 6, column : column + 6].copy()
                with patch.object(self.tool, "_compute_hog", wraps=self.tool._compute_hog) as compute:
                    score, match = self.tool._compute_differences(
                        self.image, person, self.parameters, self.tool.metric_options["Euclidean"],
                        step_size=step_size,
                    )
                self.assertEqual((score, match), (0.0, position))
                self.assertEqual(compute.call_count, windows + 1)

    def test_step_larger_than_the_search_space_still_compares_the_origin(self):
        metric = self.tool.metric_options["Euclidean"]
        with patch.object(self.tool, "_compute_hog", wraps=self.tool._compute_hog) as compute:
            score, position = self.tool._compute_differences(
                self.image, self.person, self.parameters, metric, step_size=100,
            )
        self.assertEqual(position, (0, 0))
        self.assertEqual(compute.call_count, 2)
        expected_score = metric(
            self.tool._compute_hog(self.person, self.parameters),
            self.tool._compute_hog(self.image[:6, :6], self.parameters),
        )
        self.assertEqual(score, expected_score)

    def test_nonpositive_step_sizes_are_rejected(self):
        for step_size in (0, -1):
            with self.subTest(step_size=step_size), self.assertRaisesRegex(ValueError, "[Ss]tep"):
                self.tool._compute_differences(
                    self.image, self.person, self.parameters, self.tool.metric_options["Euclidean"],
                    step_size=step_size,
                )

    def test_run_uses_original_coordinates_and_reports_the_stepped_window_count(self):
        document = ImageDocument()
        document.current = Image.fromarray(np.full((20, 25, 3), 200, dtype=np.uint8))
        before = document.current.tobytes()
        person = np.zeros((6, 8, 3), dtype=np.uint8)
        with patch.object(self.tool, "_load_image_person", return_value=person), \
             patch.object(self.tool, "_choose_hog_parameters", return_value=((1, 1), (1, 1), 9)), \
             patch.object(self.tool, "_choose_metric", return_value=self.tool.metric_options["Euclidean"]), \
             patch.object(self.tool, "_choose_step_size", return_value=4), \
             patch.object(self.tool, "_compute_differences", return_value=(0.1, (4, 12))) as compute:
            result = self.tool.run(None, document)
        compute.assert_called_once()
        np.testing.assert_array_equal(compute.call_args.args[0], np.asarray(document.current))
        self.assertEqual(compute.call_args.kwargs["step_size"], 4)
        self.assertFalse(compute.call_args.kwargs["maximize"])
        self.assertEqual(result.details["Step size"], 4)
        self.assertEqual(result.details["Windows compared"], 20)
        self.assertEqual(result.details["Top-left (x, y)"], (12, 4))
        self.assertEqual(result.details["Window size"], (8, 6))
        self.assertEqual(result.details["Template size"], (8, 6))
        self.assertNotIn("Background scale", result.details)
        self.assertEqual(result.image.size, (25, 20))
        self.assertEqual(result.image.getpixel((12, 4)), (255, 0, 0))
        self.assertEqual(result.image.getpixel((19, 9)), (255, 0, 0))
        self.assertEqual(result.image.getpixel((0, 0)), (200, 200, 200))
        self.assertEqual(document.current.tobytes(), before)


if __name__ == "__main__":
    unittest.main()
