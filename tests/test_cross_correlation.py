"""Numerical and image-mode checks without opening dialogs."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.set4.cross_correlation import CrossCorrelationTool


class CrossCorrelationNumericalTests(unittest.TestCase):
    def setUp(self):
        self.tool = CrossCorrelationTool()

    def test_ssd_matches_direct_squared_differences_without_uint8_overflow(self):
        image = np.array(
            [
                [255, 180, 230, 140, 220, 160],
                [130, 250, 128, 200, 150, 240],
                [170, 210, 190, 255, 135, 225],
                [245, 155, 205, 175, 235, 185],
            ],
            dtype=np.uint8,
        )
        template = image[:2, :3].copy()
        expected = np.empty((3, 4), dtype=np.float64)
        for row, column in np.ndindex(expected.shape):
            difference = (
                image[row : row + 2, column : column + 3].astype(np.float64)
                - template.astype(np.float64)
            )
            expected[row, column] = np.sum(difference * difference)

        result = self.tool._sum_square_differences(image, template)

        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-8)
        self.assertEqual(np.unravel_index(np.argmin(result), result.shape), (0, 0))
        self.assertGreaterEqual(float(result.min()), 0)

    def test_mean_subtracted_image_correlation_matches_direct_window_products(self):
        image = np.array(
            [
                [230, 140, 250, 190],
                [160, 220, 130, 240],
                [210, 150, 200, 180],
            ],
            dtype=np.uint8,
        )
        template = np.array([[180, 255], [130, 220]], dtype=np.uint8)
        expected = np.empty((2, 3), dtype=np.float64)
        image_mean = float(image.mean())
        for row, column in np.ndindex(expected.shape):
            window = image[row : row + 2, column : column + 2].astype(np.float64)
            expected[row, column] = np.sum((window - image_mean) * template)

        result = self.tool._cross_correlation_normalized_image(image, template)

        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-8)

    def test_normalized_correlation_finds_rectangular_templates_at_image_borders(self):
        generator = np.random.default_rng(2026)
        for template_shape, location in [((3, 4), (0, 0)), ((2, 3), (7, 8))]:
            with self.subTest(template_shape=template_shape, location=location):
                image = generator.integers(0, 256, size=(9, 11), dtype=np.uint8)
                height, width = template_shape
                row, column = location
                template = image[row : row + height, column : column + width].copy()

                result = self.tool._normalized_cross_correlation(image, template)

                self.assertEqual(result.shape, (10 - height, 12 - width))
                self.assertEqual(np.unravel_index(np.argmax(result), result.shape), location)
                self.assertAlmostEqual(float(result[location]), 1.0, places=10)

    def test_normalized_correlation_rejects_constant_template(self):
        image = np.arange(48, dtype=np.uint8).reshape(6, 8)
        template = np.full((2, 3), 200, dtype=np.uint8)

        with self.assertRaises(ValueError):
            self.tool._normalized_cross_correlation(image, template)


class CrossCorrelationImageTests(unittest.TestCase):
    module = "forensics_app.tools.set4.cross_correlation"

    def setUp(self):
        self.tool = CrossCorrelationTool()
        self.template = Image.fromarray(np.array([
            [12, 240, 80, 170],
            [220, 32, 150, 2],
            [45, 200, 96, 255],
        ], dtype=np.uint8))

    def run_match(self, source, template, choice=None, threshold=1.0,
                  transformation="No transformation", angle=90,
                  visualization="Thresholded image"):
        document = ImageDocument()
        document.current = source.copy()
        before = document.current.tobytes()
        choice = choice or self.tool.options[2]
        selections = [choice, transformation]
        if choice == self.tool.options[2]:
            selections.append(visualization)
        with TemporaryDirectory() as folder:
            filename = str(Path(folder) / "template.png")
            template.save(filename)
            with patch(f"{self.module}.dialog_options", side_effect=selections), \
                 patch(f"{self.module}.filedialog.askopenfilename", return_value=filename), \
                 patch(f"{self.module}.simpledialog.askfloat", return_value=threshold), \
                 patch.object(self.tool, "_choose_angle", return_value=angle):
                result = self.tool.run(None, document)
        self.assertEqual(document.current.tobytes(), before)
        self.assertEqual(document.current.mode, source.mode)
        return result

    def test_all_methods_mark_multiple_border_match_centers_in_red(self):
        source = Image.new("L", (30, 20))
        source.paste(self.template, (0, 0))
        source.paste(self.template, (26, 17))
        for choice in self.tool.options:
            with self.subTest(choice=choice):
                result = self.run_match(source, self.template, choice)
                self.assertEqual(set(result.details["Peak centers (x, y)"]), {(2, 1), (28, 18)})
                self.assertEqual(result.details["Detected peaks"], 2)
                self.assertEqual(result.image.mode, "RGB")
                self.assertEqual(result.image.size, source.size)
                self.assertEqual(result.image.getpixel((2, 1)), (255, 0, 0))
                self.assertEqual(result.image.getpixel((28, 18)), (255, 0, 0))

    def test_mean_subtracted_correlation_uses_an_inclusive_relative_threshold(self):
        source = Image.new("L", (24, 17), 100)
        positions = [(1, 1), (7, 10), (13, 19)]
        cases = [
            ((0.6, 0.54, 0.3), 0.8),
            ((0.6, 0.48, 0.3), 0.8),
            ((0.6, 0.6, 0.3), 1.0),
        ]
        for values, threshold in cases:
            with self.subTest(values=values, threshold=threshold):
                response = np.zeros((15, 21))
                for position, value in zip(positions, values):
                    response[position] = value
                with patch.object(self.tool, "_cross_correlation_normalized_image", return_value=response):
                    result = self.run_match(source, self.template, self.tool.options[1], threshold)
                self.assertEqual(set(result.details["Peak centers (x, y)"]), {(3, 2), (12, 8)})
                self.assertEqual(result.details["Detected peaks"], 2)
                self.assertEqual(result.image.getpixel((3, 2)), (255, 0, 0))
                self.assertEqual(result.image.getpixel((12, 8)), (255, 0, 0))

        with self.subTest(method="Normalized cross-correlation keeps its absolute threshold"):
            with patch.object(self.tool, "_normalized_cross_correlation", return_value=response):
                result = self.run_match(source, self.template, self.tool.options[2], threshold=0.8)
            self.assertEqual(result.details["Detected peaks"], 0)

    def test_relative_correlation_with_no_positive_response_has_no_matches(self):
        source = Image.new("L", (24, 17), 100)
        negative_response = np.full((15, 21), -1.0)
        negative_response[7, 10] = -0.1
        for response in (np.zeros((15, 21)), negative_response):
            with self.subTest(maximum=float(response.max())):
                with patch.object(self.tool, "_cross_correlation_normalized_image", return_value=response):
                    result = self.run_match(source, self.template, self.tool.options[1], threshold=0.8)
                self.assertEqual(result.details["Detected peaks"], 0)
                self.assertEqual(result.details["Peak centers (x, y)"], [])

    def test_relative_correlation_finds_an_exact_low_contrast_template(self):
        template = Image.fromarray(np.array([
            [9, 10, 11],
            [12, 8, 10],
            [11, 10, 9],
        ], dtype=np.uint8))
        source = Image.new("L", (24, 17))
        source.paste(template, (9, 7))
        response = self.tool._cross_correlation_normalized_image(
            np.asarray(source, dtype=np.float64) / 255,
            np.asarray(template, dtype=np.float64) / 255,
        )
        self.assertGreater(float(response.max()), 0)
        self.assertLess(float(response.max()), 0.8)
        self.assertEqual(np.unravel_index(np.argmax(response), response.shape), (7, 9))

        result = self.run_match(source, template, self.tool.options[1], threshold=0.8)

        self.assertEqual(result.details["Peak centers (x, y)"], [(10, 8)])
        self.assertEqual(result.details["Detected peaks"], 1)
        self.assertEqual(result.image.getpixel((10, 8)), (255, 0, 0))

    def test_gray_and_color_inputs_dim_background_and_keep_markers_red(self):
        generator = np.random.default_rng(123)
        for source_mode in ("L", "RGB"):
            for template_mode in ("L", "RGB"):
                with self.subTest(source=source_mode, template=template_mode):
                    shape = (20, 30) if source_mode == "L" else (20, 30, 3)
                    source = Image.fromarray(generator.integers(0, 256, size=shape, dtype=np.uint8))
                    source.paste(self.template.convert(source_mode), (9, 7))
                    if source_mode == "L":
                        source.putpixel((25, 15), 128)
                        expected_background = (64, 64, 64)
                    else:
                        source.putpixel((25, 15), (51, 102, 153))
                        expected_background = (10, 41, 92)
                    result = self.run_match(source, self.template.convert(template_mode))
                    self.assertEqual(result.details["Peak centers (x, y)"], [(11, 8)])
                    self.assertEqual(result.image.getpixel((11, 8)), (255, 0, 0))
                    self.assertEqual(result.image.getpixel((25, 15)), expected_background)

    def test_even_height_template_and_single_valid_position(self):
        template = self.template.crop((0, 0, 3, 2))
        result = self.run_match(template, template)
        self.assertEqual(result.details["Peak centers (x, y)"], [(1, 1)])
        self.assertEqual(result.image.getpixel((1, 1)), (255, 0, 0))

    def test_flat_normalized_response_has_no_markers_on_dimmed_background(self):
        source = Image.new("RGB", (20, 15), (50, 100, 150))
        result = self.run_match(source, self.template)
        self.assertEqual(result.details["Detected peaks"], 0)
        expected = Image.new("RGB", source.size, (10, 39, 88))
        np.testing.assert_array_equal(np.asarray(result.image), np.asarray(expected))

    def test_raw_normalized_correlation_returns_a_source_sized_grayscale_map(self):
        generator = np.random.default_rng(123)
        source = Image.fromarray(generator.integers(0, 256, size=(20, 30, 3), dtype=np.uint8))
        source.paste(self.template.convert("RGB"), (9, 7))
        response = self.tool._normalized_cross_correlation(
            np.asarray(source.convert("L"), dtype=np.float64) / 255,
            np.asarray(self.template, dtype=np.float64) / 255,
            pad_input=True,
        )
        with patch.object(self.tool, "_choose_threshold") as choose_threshold, \
             patch.object(self.tool, "_normalized_cross_correlation", wraps=self.tool._normalized_cross_correlation) as correlate, \
             patch(f"{self.module}.feature.peak_local_max") as find_peaks, \
             patch(f"{self.module}.exposure.adjust_gamma") as adjust_gamma:
            result = self.run_match(source, self.template, visualization="Raw output")
        choose_threshold.assert_not_called()
        find_peaks.assert_not_called()
        adjust_gamma.assert_not_called()
        correlate.assert_called_once()
        self.assertTrue(correlate.call_args.kwargs["pad_input"])
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.size, source.size)
        self.assertEqual(result.details["Visualization"], "Raw output")
        self.assertEqual(result.details["Output mode"], "L")
        self.assertNotIn("Peak threshold", result.details)
        self.assertNotIn("Detected peaks", result.details)
        self.assertNotIn("Peak centers (x, y)", result.details)
        pixels = np.asarray(result.image)
        self.assertEqual(int(pixels.min()), 0)
        self.assertEqual(int(pixels.max()), 255)
        ordered_pixels = pixels.ravel()[np.argsort(response, axis=None)].astype(int)
        self.assertTrue(np.all(np.diff(ordered_pixels) >= 0))

    def test_raw_normalized_correlation_handles_flat_response_maps(self):
        source = Image.new("RGB", (20, 15), (50, 100, 150))
        for value in (0.0, 0.5):
            with self.subTest(response=value):
                response = np.full((15, 20), value)
                with patch.object(self.tool, "_normalized_cross_correlation", return_value=response):
                    result = self.run_match(source, self.template, visualization="Raw output")
                self.assertEqual(result.image.mode, "L")
                self.assertEqual(result.image.size, source.size)
                self.assertEqual(np.unique(np.asarray(result.image)).size, 1)

    def test_cancelling_the_ncc_visualization_stops_before_processing(self):
        source = Image.new("RGB", (20, 15), (50, 100, 150))
        with patch.object(self.tool, "_choose_threshold") as choose_threshold, \
             patch.object(self.tool, "_normalized_cross_correlation") as correlate, \
             patch(f"{self.module}.feature.peak_local_max") as find_peaks, \
             patch(f"{self.module}.exposure.adjust_gamma") as adjust_gamma:
            result = self.run_match(source, self.template, visualization=None)
        self.assertIsNone(result)
        choose_threshold.assert_not_called()
        correlate.assert_not_called()
        find_peaks.assert_not_called()
        adjust_gamma.assert_not_called()

    def test_other_matching_methods_do_not_offer_the_ncc_visualization(self):
        source = Image.new("L", (20, 15))
        source.paste(self.template, (9, 7))
        for choice in self.tool.options[:2]:
            with self.subTest(choice=choice):
                with patch.object(self.tool, "_choose_visualization") as choose_visualization, \
                     patch.object(self.tool, "_choose_threshold", wraps=self.tool._choose_threshold) as choose_threshold:
                    result = self.run_match(source, self.template, choice, visualization="Raw output")
                choose_visualization.assert_not_called()
                choose_threshold.assert_called_once()
                self.assertEqual(result.image.mode, "RGB")
                self.assertEqual(result.details["Peak centers (x, y)"], [(11, 8)])
                self.assertEqual(result.image.getpixel((11, 8)), (255, 0, 0))

    def test_ssd_accepts_a_constant_template(self):
        source = Image.new("L", (4, 3), 128)
        result = self.run_match(source, source, self.tool.options[0])
        self.assertEqual(result.details["Peak centers (x, y)"], [(2, 1)])

    def test_template_larger_in_either_dimension_is_rejected(self):
        source = Image.new("L", (8, 6))
        for size in ((9, 2), (2, 7)):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "larger"):
                self.run_match(source, Image.new("L", size))

    def test_cancellation_at_each_dialog_leaves_document_unchanged(self):
        document = ImageDocument()
        document.current = self.template.copy()
        before = document.current.tobytes()
        with patch(f"{self.module}.dialog_options", return_value=None), \
             patch(f"{self.module}.filedialog.askopenfilename") as ask_file:
            self.assertIsNone(self.tool.run(None, document))
            ask_file.assert_not_called()
        with patch(f"{self.module}.dialog_options", return_value=self.tool.options[2]), \
             patch(f"{self.module}.filedialog.askopenfilename", return_value=""), \
             patch(f"{self.module}.simpledialog.askfloat") as ask_threshold:
            self.assertIsNone(self.tool.run(None, document))
            ask_threshold.assert_not_called()
        self.assertIsNone(self.run_match(document.current, self.template, threshold=None))
        self.assertEqual(document.current.tobytes(), before)

    def test_tool_is_available_from_the_application_registry(self):
        tools = build_tool_registry().all()
        self.assertTrue(any(isinstance(tool, CrossCorrelationTool) and tool.category == "Set4" for tool in tools))

    def test_template_transformations_match_correct_locations(self):
        pixels = np.asarray(self.template)
        transformations = {
            "No transformation": pixels,
            "Rotation": np.rot90(pixels),
            "Horizontal flip": np.fliplr(pixels),
            "Vertical flip": np.flipud(pixels),
            "Transposition": pixels.T,
        }
        for transformation, expected in transformations.items():
            with self.subTest(transformation=transformation):
                transformed = Image.fromarray(expected)
                source = Image.new("L", (20, 16), 80)
                left, top = 20 - transformed.width, 16 - transformed.height
                source.paste(transformed, (left, top))
                result = self.run_match(source, self.template, transformation=transformation)
                center = (left + transformed.width // 2, top + transformed.height // 2)
                self.assertEqual(result.details["Peak centers (x, y)"], [center])
                self.assertEqual(result.details["Template size"], transformed.size)
                self.assertEqual(result.details["Template transformation"], transformation)
                self.assertEqual(result.image.getpixel(center), (255, 0, 0))

    def test_template_transformation_rotation_can_fit_after_dimensions_swap(self):
        source = Image.fromarray(np.rot90(np.asarray(self.template)))
        result = self.run_match(source, self.template, transformation="Rotation")
        self.assertEqual(result.details["Template size"], (3, 4))
        self.assertEqual(result.details["Peak centers (x, y)"], [(1, 2)])

    def test_template_transformation_expanded_rotation_size_is_checked(self):
        with self.assertRaisesRegex(ValueError, "larger"):
            self.run_match(self.template, self.template, transformation="Rotation", angle=45)

    def test_template_transformation_cancellation_stops_before_threshold(self):
        for transformation, angle in ((None, 90), ("Rotation", None)):
            with self.subTest(transformation=transformation, angle=angle):
                with patch.object(self.tool, "_choose_threshold") as choose_threshold:
                    result = self.run_match(
                        self.template, self.template, transformation=transformation, angle=angle
                    )
                self.assertIsNone(result)
                choose_threshold.assert_not_called()


if __name__ == "__main__":
    unittest.main()
