"""Small numerical checks; dialogs and plotting never open windows."""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from forensics_app.core import ImageDocument
from forensics_app.tools.set2.channel_split import ChannelSplitTool
from forensics_app.tools.set2.channel_swap import ChannelSwap
from forensics_app.tools.set2.contrast_operations import ContrastOperations
from forensics_app.tools.set3.convolution import Convolution
from forensics_app.tools.set3.filters import FiltersTool
from forensics_app.tools.set2.histogram import HistogramVisualization
from forensics_app.tools.set2.histogram_matching import HistogramMatching
from forensics_app.tools.set2.masking import Masking
from forensics_app.tools.set2.masking_superposition import MaskingSuperposition
from forensics_app.tools.set3.canny_edge import CannyEdge
from forensics_app.tools.set3.sharpening import Sharpening


class ProcessingToolTests(unittest.TestCase):
    def document(self, pixels):
        document = ImageDocument()
        document.current = Image.fromarray(np.asarray(pixels, dtype=np.uint8))
        return document

    def run_tool(self, tool, document):
        before = np.asarray(document.current).copy()
        result = tool.run(None, document)
        np.testing.assert_array_equal(np.asarray(document.current), before)
        return result

    def test_channel_split_extracts_green(self):
        document = self.document([[[10, 20, 30], [40, 50, 60]]])
        with patch("forensics_app.tools.set2.channel_split.dialog_options", return_value="Green"):
            result = self.run_tool(ChannelSplitTool(), document)
        np.testing.assert_array_equal(np.asarray(result.image), [[20, 50]])

    def test_channel_swap_exchanges_red_and_blue(self):
        document = self.document([[[10, 20, 30]]])
        with patch("forensics_app.tools.set2.channel_swap.dialog_options", side_effect=["Red", "Blue"]):
            result = self.run_tool(ChannelSwap(), document)
        np.testing.assert_array_equal(np.asarray(result.image), [[[30, 20, 10]]])

    def test_contrast_decrease_and_stretch(self):
        tool = ContrastOperations()
        with patch.object(tool, "_ask_cutoff", return_value=20):
            result = tool.contrast_decreased(None, np.array([[0., .5, 1.]]))
        np.testing.assert_allclose(result, [[.1, .5, .9]])
        with patch.object(tool, "_ask_cutoff", return_value=50):
            result = tool.contrast_stretching(None, np.array([[0., .25, .5, .75, 1.]]))
        np.testing.assert_allclose(result, [[0., 0., .5, 1., 1.]])

    def test_histogram_counts_grayscale_pixels(self):
        document = self.document([[0, 0, 255]])
        with patch("forensics_app.tools.set2.histogram.dialog_options", return_value="Default"), \
             patch("forensics_app.tools.set2.histogram.plt.show"), \
             patch("forensics_app.tools.set2.histogram.plt.close"):
            try:
                result = self.run_tool(HistogramVisualization(), document)
                counts = [bar.get_height() for bar in plt.gca().patches]
            finally:
                figure = plt.gcf()
        plt.close(figure)
        expected = np.zeros(256)
        expected[0], expected[255] = 2, 1
        np.testing.assert_array_equal(counts, expected)
        self.assertIsNone(result.image)

    def test_histogram_matching_maps_to_constant_reference_lightness(self):
        document = self.document([[0, 255]])
        with TemporaryDirectory() as folder:
            filename = str(Path(folder) / "reference.png")
            Image.new("RGB", (2, 1), (128, 128, 128)).save(filename)
            with patch("forensics_app.tools.set2.histogram_matching.filedialog.askopenfilename", return_value=filename):
                result = self.run_tool(HistogramMatching(), document)
        np.testing.assert_allclose(np.asarray(result.image), 128, atol=1)

    def test_masking_threshold_boundaries(self):
        document = self.document([[49, 50, 51, 100]])
        cases = [
            ("Binary", [50], [[0, 0, 51, 100]]),
            ("Inverse", [50], [[49, 50, 0, 0]]),
            ("Range", [50, 51], [[0, 50, 51, 0]]),
        ]
        for choice, values, expected in cases:
            with self.subTest(choice=choice), \
                 patch("forensics_app.tools.set2.masking.dialog_options", return_value=choice), \
                 patch("forensics_app.tools.set2.masking.simpledialog.askinteger", side_effect=values):
                result = self.run_tool(Masking(), document)
            np.testing.assert_array_equal(np.asarray(result.image), expected)

    def test_masking_cancellation(self):
        for choice in ("Binary", "Inverse", "Range"):
            with self.subTest(choice=choice), \
                 patch("forensics_app.tools.set2.masking.dialog_options", return_value=choice), \
                 patch("forensics_app.tools.set2.masking.simpledialog.askinteger", return_value=None) as ask:
                self.assertIsNone(self.run_tool(Masking(), self.document([[50]])))
                ask.assert_called_once()

    def test_convolution_average_spreads_an_impulse(self):
        pixels = np.zeros((7, 7), dtype=np.uint8)
        pixels[3, 3] = 90
        tool = Convolution()
        with patch.object(tool, "_choose_kernel", return_value="Average"), \
             patch.object(tool, "_choose_direction", return_value="2D"), \
             patch.object(tool, "_choose_size", return_value=(3, 3)):
            result = self.run_tool(tool, self.document(pixels))
        expected = np.zeros_like(pixels)
        expected[2:5, 2:5] = 10
        np.testing.assert_array_equal(np.asarray(result.image), expected)

    def test_convolution_2d_detects_a_vertical_boundary(self):
        pixels = np.zeros((9, 9), dtype=np.uint8)
        pixels[:, 4:] = 255
        for kernel in ("Sobel", "Prewitt"):
            tool = Convolution()
            with self.subTest(kernel=kernel), \
                 patch.object(tool, "_choose_kernel", return_value=kernel), \
                 patch.object(tool, "_choose_direction", return_value="2D"):
                result = self.run_tool(tool, self.document(pixels))
            edges = np.asarray(result.image)
            self.assertTrue(np.all(edges[:, 3:5] == 255))
            self.assertTrue(np.all(edges[:, :3] == 0))
            self.assertTrue(np.all(edges[:, 5:] == 0))

    def test_superposition_preserves_red_and_black_background(self):
        document = self.document([[[10, 20, 30], [40, 50, 60]]])
        with TemporaryDirectory() as folder:
            filename = str(Path(folder) / "overlay.png")
            self.document([[[0, 0, 0], [255, 0, 0]]]).current.save(filename)
            with patch("forensics_app.tools.set2.masking_superposition.dialog_options", return_value="Put over an image"), \
                 patch("forensics_app.tools.set2.masking_superposition.filedialog.askopenfilename", return_value=filename):
                result = self.run_tool(MaskingSuperposition(), document)
        np.testing.assert_array_equal(np.asarray(result.image), [[[10, 20, 30], [255, 0, 0]]])

    def test_texture_replaces_nonblack_pixels_including_red(self):
        document = self.document([[[0, 0, 0], [255, 0, 0]]])
        with TemporaryDirectory() as folder:
            filename = str(Path(folder) / "texture.png")
            Image.new("RGB", (2, 1), (10, 20, 30)).save(filename)
            with patch("forensics_app.tools.set2.masking_superposition.dialog_options", return_value="Change texture"), \
                 patch("forensics_app.tools.set2.masking_superposition.filedialog.askopenfilename", return_value=filename):
                result = self.run_tool(MaskingSuperposition(), document)
        np.testing.assert_array_equal(np.asarray(result.image), [[[0, 0, 0], [10, 20, 30]]])

    def test_filters_median_removes_an_isolated_bright_pixel(self):
        pixels = np.zeros((5, 5), dtype=np.uint8)
        pixels[2, 2] = 255
        tool = FiltersTool()
        with patch.object(tool, "_choose_filter", return_value="Edge-preserving denoising"):
            result = self.run_tool(tool, self.document(pixels))
        np.testing.assert_array_equal(np.asarray(result.image), np.zeros_like(pixels))

    def test_filters_otsu_keeps_bright_or_dark_pixels(self):
        pixels = [[50, 50, 200, 200]]
        for choice, expected in [("Binary", [[0, 0, 200, 200]]), ("Inverse", [[50, 50, 0, 0]])]:
            tool = FiltersTool()
            with self.subTest(choice=choice), \
                 patch.object(tool, "_choose_filter", return_value="Otsu threshold (mask)"), \
                 patch.object(tool, "_choose_mask", return_value=choice):
                result = self.run_tool(tool, self.document(pixels))
            np.testing.assert_array_equal(np.asarray(result.image), expected)

    def test_canny_detects_a_boundary(self):
        pixels = np.zeros((32, 32), dtype=np.uint8)
        pixels[:, 16:] = 255
        with patch("forensics_app.tools.set3.canny_edge.dialog_options", side_effect=["Default thresholds", "Canny edges"]), \
             patch("forensics_app.tools.set3.canny_edge.simpledialog.askfloat", return_value=1.):
            result = self.run_tool(CannyEdge(), self.document(pixels))
        edges = np.asarray(result.image)
        self.assertGreater(np.count_nonzero(edges[2:-2, 14:18]), 0)
        self.assertFalse(np.any(edges[:, :10]))
        self.assertFalse(np.any(edges[:, 22:]))
        self.assertTrue(set(np.unique(edges)).issubset({0, 255}))

    def test_sharpening_does_not_mix_constant_rgb_channels(self):
        pixels = np.full((8, 8, 3), (30, 100, 200), dtype=np.uint8)
        with patch("forensics_app.tools.set3.sharpening.simpledialog.askfloat", side_effect=[1., 1.]):
            result = self.run_tool(Sharpening(), self.document(pixels))
        np.testing.assert_allclose(np.asarray(result.image), pixels, atol=1)


if __name__ == "__main__":
    unittest.main()
