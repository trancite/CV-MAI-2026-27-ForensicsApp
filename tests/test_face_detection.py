"""YuNet workflow, model caching, and registration without GUI windows or downloads."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.set6.face_detection import FaceDetection


class FaceDetectionTests(unittest.TestCase):
    module = "forensics_app.tools.set6.face_detection"

    def setUp(self):
        self.tool = FaceDetection()
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (100, 80), (30, 100, 200))
        self.parameters = (0.6, 0.3, 5000)
        self.face = np.array([
            10, 20, 30, 35, 18, 30, 30, 30, 24, 38, 19, 46, 29, 46, 0.95,
        ], dtype=np.float32)

    def test_registered_under_set6_without_downloading_model(self):
        with patch(f"{self.module}.urllib.request.urlopen") as download:
            tools = [tool for tool in build_tool_registry().all() if tool.tool_id == "face_detection"]
        download.assert_not_called()
        self.assertEqual(len(tools), 1)
        self.assertIsInstance(tools[0], FaceDetection)
        self.assertEqual(tools[0].category, "Set6")
        self.assertTrue(tools[0].model_path.is_absolute())

    def test_workflow_converts_to_bgr_and_marks_all_faces_without_mutation(self):
        for mode in ("RGB", "L", "RGBA", "P"):
            with self.subTest(mode=mode):
                self.document.current = self.document.current.convert(mode)
                before = self.document.current.copy()
                second_face = self.face.copy()
                second_face[0:14:2] += 50
                with patch.object(self.tool, "_choose_parameters", return_value=self.parameters), \
                     patch.object(self.tool, "_check_model"), \
                     patch("cv2.FaceDetectorYN") as factory:
                    detector = factory.create.return_value
                    detector.detect.return_value = (1, np.stack((self.face, second_face)))
                    result = self.tool.run(None, self.document)
                factory.create.assert_called_once_with(
                    str(self.tool.model_path), "", (100, 80), *self.parameters,
                )
                pixels = detector.detect.call_args.args[0]
                np.testing.assert_array_equal(pixels, np.asarray(before.convert("RGB"))[:, :, ::-1])
                self.assertTrue(pixels.flags.c_contiguous)
                self.assertEqual(result.details["Number of faces"], 2)
                self.assertAlmostEqual(result.details["Face 1 confidence"], 0.95, places=6)
                self.assertEqual(result.details["Face 1 box (x, y, width, height)"], (10, 20, 30, 35))
                self.assertEqual(result.details["Face 1 Nose"], (24, 38))
                self.assertEqual(result.image.getpixel((10, 20)), (0, 255, 0))
                self.assertEqual(result.image.getpixel((60, 20)), (0, 255, 0))
                self.assertEqual(result.image.getpixel((24, 38)), (0, 0, 255))
                self.assertEqual(result.image.getpixel((0, 79)), before.convert("RGB").getpixel((0, 79)))
                self.assertEqual(result.image.mode, "RGB")
                self.assertEqual(result.image.size, before.size)
                self.assertEqual(self.document.current.mode, mode)
                self.assertEqual(self.document.current.tobytes(), before.tobytes())
                self.assertFalse(self.document.can_undo)

    def test_no_detections_returns_unmarked_image(self):
        for faces in (None, np.empty((0, 15), dtype=np.float32)):
            with self.subTest(faces=faces), \
                 patch.object(self.tool, "_choose_parameters", return_value=self.parameters), \
                 patch.object(self.tool, "_check_model"), \
                 patch("cv2.FaceDetectorYN") as factory:
                factory.create.return_value.detect.return_value = (1, faces)
                result = self.tool.run(None, self.document)
                self.assertEqual(result.details["Number of faces"], 0)
                self.assertEqual(result.message, "No faces detected.")
                self.assertEqual(result.image.tobytes(), self.document.current.tobytes())

    def test_cancellation_stops_immediately_without_loading_model(self):
        cases = [([None], [], 1, 0), ([0.6, None], [], 2, 0), ([0.6, 0.3], [None], 2, 1)]
        for floats, integers, float_calls, int_calls in cases:
            with self.subTest(floats=floats, integers=integers), \
                 patch(f"{self.module}.simpledialog.askfloat", side_effect=floats) as askfloat, \
                 patch(f"{self.module}.simpledialog.askinteger", side_effect=integers) as askinteger, \
                 patch.object(self.tool, "_check_model") as check:
                before = self.document.current.tobytes()
                self.assertIsNone(self.tool.run(None, self.document))
                self.assertEqual(askfloat.call_count, float_calls)
                self.assertEqual(askinteger.call_count, int_calls)
                check.assert_not_called()
                self.assertEqual(self.document.current.tobytes(), before)

    def test_parameter_defaults_and_dialog_parent(self):
        parent = object()
        with patch(f"{self.module}.simpledialog.askfloat", side_effect=[0.6, 0.3]) as askfloat, \
             patch(f"{self.module}.simpledialog.askinteger", return_value=5000) as askinteger:
            self.assertEqual(self.tool._choose_parameters(parent), self.parameters)
        self.assertEqual([call.kwargs["initialvalue"] for call in askfloat.call_args_list], [0.6, 0.3])
        for call in askfloat.call_args_list + askinteger.call_args_list:
            self.assertIs(call.kwargs["parent"], parent)
        self.assertEqual(askinteger.call_args.kwargs["minvalue"], 1)

    def test_missing_opencv_provides_install_instructions(self):
        with patch.object(self.tool, "_choose_parameters", return_value=self.parameters), \
             patch.dict("sys.modules", {"cv2": None}), \
             self.assertRaisesRegex(RuntimeError, "pip install -r requirements.txt"):
            self.tool.run(None, self.document)

    def test_completed_download_is_cached(self):
        with TemporaryDirectory() as folder:
            tool = FaceDetection(Path(folder) / "models" / "yunet.onnx")
            payload = b"x" * 100_001
            with patch(f"{self.module}.urllib.request.urlopen", return_value=BytesIO(payload)) as download:
                tool._check_model()
                tool._check_model()
            download.assert_called_once_with(tool.model_url, timeout=30)
            self.assertEqual(tool.model_path.read_bytes(), payload)
            self.assertEqual(list(tool.model_path.parent.iterdir()), [tool.model_path])

    def test_failed_downloads_are_cleaned_up(self):
        for failure in (BytesIO(b"incomplete"), URLError("offline")):
            with self.subTest(failure=failure), TemporaryDirectory() as folder:
                tool = FaceDetection(Path(folder) / "yunet.onnx")
                kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
                with patch(f"{self.module}.urllib.request.urlopen", **kwargs), \
                     self.assertRaises(RuntimeError):
                    tool._check_model()
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_existing_incomplete_model_is_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "yunet.onnx"
            path.write_bytes(b"incomplete")
            with patch(f"{self.module}.urllib.request.urlopen") as download, \
                 self.assertRaisesRegex(RuntimeError, "incomplete"):
                FaceDetection(path)._check_model()
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
