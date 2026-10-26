"""HOG parameter validation and visualization checks without dialogs."""

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.set4.hog import HOGTool


class HOGTests(unittest.TestCase):
    def setUp(self):
        self.tool = HOGTool()
        self.document = ImageDocument()

    def test_one_pixel_cells_fail_with_a_clear_validation_error(self):
        self.document.current = Image.new("L", (32, 32))
        with patch.object(self.tool, "choose_parameters", return_value=((1, 1), (3, 3), 9)):
            with self.assertRaisesRegex(ValueError, "at least 2 pixels per cell"):
                self.tool.run(None, self.document)

    def test_block_must_fit_both_image_dimensions(self):
        for size in ((32, 48), (48, 32), (32, 32)):
            with self.subTest(size=size):
                self.document.current = Image.new("L", size)
                with patch.object(self.tool, "choose_parameters", return_value=((16, 16), (3, 3), 9)):
                    with self.assertRaisesRegex(ValueError, "require at least a 48 x 48 image"):
                        self.tool.run(None, self.document)

    def test_exactly_fitting_block_visualizes_gray_and_color_without_mutation(self):
        generator = np.random.default_rng(42)
        for pixels_per_cell in (2, 8):
            for mode in ("L", "RGB"):
                with self.subTest(pixels_per_cell=pixels_per_cell, mode=mode):
                    side = pixels_per_cell * 3
                    shape = (side, side) if mode == "L" else (side, side, 3)
                    original = generator.integers(0, 256, shape, dtype=np.uint8)
                    self.document.current = Image.fromarray(original)
                    params = ((pixels_per_cell, pixels_per_cell), (3, 3), 9)
                    with patch.object(self.tool, "choose_parameters", return_value=params):
                        result = self.tool.run(None, self.document)
                    self.assertEqual(result.image.mode, "L")
                    self.assertEqual(result.image.size, (side, side))
                    self.assertGreater(np.asarray(result.image).max(), 0)
                    np.testing.assert_array_equal(np.asarray(self.document.current), original)
                    self.assertEqual(self.document.current.mode, mode)


if __name__ == "__main__":
    unittest.main()
