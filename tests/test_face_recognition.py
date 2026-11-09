"""Test faces and identical copies must never become their own nearest match."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.set6.face_recognition import FaceRecognition


class FaceRecognitionTests(unittest.TestCase):
    module = "forensics_app.tools.set6.face_recognition"

    def _images(self, root):
        person = Path(root) / "Alice"
        person.mkdir()
        paths = []
        for name, value in (("test", 20), ("copy", 20), ("other1", 60), ("other2", 140)):
            path = person / f"{name}.png"
            Image.new("RGB", (6, 5), (value, value, value)).save(path)
            paths.append(path)
        return paths

    def test_loader_excludes_the_test_image_and_identical_copies(self):
        with TemporaryDirectory() as root:
            paths = self._images(root)
            with Image.open(paths[0]) as image:
                test = np.asarray(image.convert("RGB"), dtype=np.float32)
            tool = FaceRecognition()
            matrix = tool._load_images(paths, exclude=test)
            self.assertEqual(matrix.shape, (2, 90))
            self.assertEqual(tool.loaded_paths_, [path.resolve() for path in paths[2:]])
            self.assertEqual(tool._get_labels(Path(root).resolve()), ["Alice", "Alice"])
            self.assertFalse(any(np.array_equal(row, test.ravel()) for row in matrix))

    def test_recognition_does_not_match_the_test_image_to_itself(self):
        with TemporaryDirectory() as root:
            paths = self._images(root)
            tool = FaceRecognition()
            with patch.object(tool, "_choose_set", return_value=root), \
                 patch.object(tool, "_choose_test_image", return_value=str(paths[0])), \
                 patch(f"{self.module}.simpledialog.askinteger", return_value=2), \
                 patch.object(tool, "_visualize_results") as show:
                result = tool.run(None, ImageDocument())
            show.assert_called_once()
            self.assertEqual(result.details["Training images"], 2)
            matches = result.details["Closest matches"]
            self.assertEqual({match["Image"] for match in matches}, {str(path.resolve()) for path in paths[2:]})
            self.assertTrue(all(match["Distance"] > 0 for match in matches))

    def test_too_few_images_after_exclusion_has_a_clear_error(self):
        with TemporaryDirectory() as root:
            paths = self._images(root)
            with Image.open(paths[0]) as image:
                test = np.asarray(image.convert("RGB"))
            with self.assertRaisesRegex(ValueError, "At least two"):
                FaceRecognition()._load_images(paths[:3], exclude=test)


if __name__ == "__main__":
    unittest.main()
