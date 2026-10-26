"""FAST intensity units, peak filtering, and cancellation without a display."""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.set5.corner_detection import CornerDetectionTool


MODULE = "forensics_app.tools.set5.corner_detection"


class CornerDetectionTests(unittest.TestCase):
    def setUp(self):
        self.tool = CornerDetectionTool()
        self.pixels = np.zeros((72, 104), dtype=np.uint8)
        rows, columns = np.indices(self.pixels.shape)
        # Two separated triangles have three corners each, with different contrast.
        for center, intensity in ((24, 255), (76, 80)):
            triangle = (rows >= 14) & (rows <= 50) & (abs(columns - center) <= (rows - 14) // 2)
            self.pixels[triangle] = intensity
        self.document = ImageDocument()
        self.document.current = Image.fromarray(self.pixels)

    def run_fast(self, intensity_threshold, peak_threshold, display_mode):
        with patch(f"{MODULE}.dialog_options", side_effect=["Fast", display_mode]), \
             patch(f"{MODULE}.simpledialog.askfloat", side_effect=[intensity_threshold, peak_threshold]):
            result = self.tool.run(None, self.document)
        np.testing.assert_array_equal(np.asarray(self.document.current), self.pixels)
        self.assertFalse(self.document.can_undo)
        return result

    def test_intensity_threshold_uses_pixel_units(self):
        for intensity, expected_count in ((0, 6), (20, 6), (100, 3), (255, 0)):
            with self.subTest(intensity=intensity):
                result = self.run_fast(intensity, .1, self.tool.display_options[0])
                self.assertEqual(result.details["Number of corners"], expected_count)
                self.assertEqual(result.image.size, self.document.current.size)
                # The bright triangle remains detectable at 20 and 100 pixel units.
                if 0 < intensity < 255:
                    self.assertEqual(result.image.getpixel((26, 14)), (255, 0, 0))

    def test_peak_threshold_filters_weaker_corners_in_both_displays(self):
        for display_mode in self.tool.display_options:
            with self.subTest(display_mode=display_mode):
                low = self.run_fast(20, .1, display_mode)
                high = self.run_fast(20, .5, display_mode)
                self.assertEqual(low.details["Number of corners"], 6)
                self.assertEqual(high.details["Number of corners"], 3)
                if display_mode == self.tool.display_options[0]:
                    self.assertEqual(low.image.getpixel((78, 14)), (255, 0, 0))
                    self.assertNotEqual(high.image.getpixel((78, 14)), (255, 0, 0))
                else:
                    # Peak selection changes the count, not the FAST response map.
                    np.testing.assert_array_equal(np.asarray(low.image), np.asarray(high.image))

    def test_cancelling_either_threshold_or_display_skips_detection(self):
        for answers, displays in (([None], ["Fast"]), ([20, None], ["Fast"]), ([20, .1], ["Fast", None])):
            with self.subTest(answers=answers, displays=displays), \
                 patch(f"{MODULE}.dialog_options", side_effect=displays) as choose_display, \
                 patch(f"{MODULE}.simpledialog.askfloat", side_effect=answers) as ask, \
                 patch(f"{MODULE}.feature.corner_fast") as detect:
                self.assertIsNone(self.tool.run(None, self.document))
            self.assertEqual(ask.call_count, len(answers))
            self.assertEqual(choose_display.call_count, len(displays))
            detect.assert_not_called()
            np.testing.assert_array_equal(np.asarray(self.document.current), self.pixels)
            self.assertFalse(self.document.can_undo)


if __name__ == "__main__":
    unittest.main()
