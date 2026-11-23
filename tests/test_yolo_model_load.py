"""YOLO inference must preserve RGB and compare predictions with real labels."""

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.set9.YOLOmodel_load import YOLOModelLoader


class YOLOModelLoadTests(unittest.TestCase):
    module = "forensics_app.tools.set9.YOLOmodel_load"

    def setUp(self):
        self.tool = YOLOModelLoader()
        self.tool.model = SimpleNamespace(
            names={0: "PC", 1: "RBC", 2: "WBC"},
            task="detect",
            predict=Mock(),
        )
        self.tool.model_path = Path("blood_cells.pt")

    @staticmethod
    def _document(color="red", size=(200, 100)):
        document = ImageDocument()
        document.original = Image.new("RGB", size, color)
        document.current = document.original.copy()
        document.path = Path("unseen.png")
        return document

    @staticmethod
    def _result(annotated_rgb, boxes=()):
        bgr = np.asarray(annotated_rgb)[:, :, ::-1].copy()
        return SimpleNamespace(boxes=list(boxes), plot=Mock(return_value=bgr))

    def test_predict_passes_pil_rgb_with_correct_red_and_blue_channels(self):
        rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
        self.tool.model.predict.return_value = [self._result(rgb)]

        self.tool._predict(rgb)

        arguments = self.tool.model.predict.call_args.kwargs
        source = arguments["source"]
        self.assertIsInstance(source, Image.Image)
        self.assertEqual(source.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(source), rgb)
        self.assertEqual(arguments["conf"], 0.25)
        self.assertEqual(arguments["imgsz"], 640)
        self.assertFalse(arguments["verbose"])

    def test_detect_returns_pil_rgb_and_class_coordinates(self):
        rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
        box = SimpleNamespace(
            cls=np.array([1]),
            conf=np.array([0.8]),
            xyxy=np.array([[2.0, 3.0, 14.0, 15.0]]),
        )
        self.tool.model.predict.return_value = [self._result(rgb, [box])]

        detections, annotated = self.tool._detect_image(rgb)

        self.assertIsInstance(annotated, Image.Image)
        self.assertEqual(annotated.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(annotated), rgb)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["Class"], "RBC")
        self.assertEqual(detections[0]["Bounding box"], [2.0, 3.0, 14.0, 15.0])

    def test_zero_detections_still_return_the_annotated_rgb_image(self):
        rgb = np.full((3, 4, 3), [255, 0, 0], dtype=np.uint8)
        self.tool.model.predict.return_value = [self._result(rgb)]

        detections, annotated = self.tool._detect_image(rgb)

        self.assertEqual(detections, [])
        self.assertIsInstance(annotated, Image.Image)
        np.testing.assert_array_equal(np.asarray(annotated), rgb)

    def test_inspect_model_is_available_without_an_image(self):
        info = {"Classes": {0: "PC", 1: "RBC", 2: "WBC"}}
        self.assertFalse(self.tool.requires_image)
        with patch(f"{self.module}.dialog_options", return_value="Inspect model"), \
             patch(f"{self.module}.filedialog.askopenfilename", return_value="blood_cells.pt"), \
             patch.object(self.tool, "_load_model") as load_model, \
             patch.object(self.tool, "_show_model_info", return_value=info), \
             patch.object(self.tool, "_predict") as predict:
            result = self.tool.run(None, ImageDocument())
        load_model.assert_called_once_with("blood_cells.pt")
        predict.assert_not_called()
        self.assertEqual(result.details, info)
        self.assertIsNone(result.image)

    def test_detection_requires_an_image_before_loading_weights(self):
        with patch(f"{self.module}.dialog_options", return_value="Detect objects"), \
             patch.object(self.tool, "_load_model") as load_model, \
             self.assertRaisesRegex(ValueError, "(?i)image"):
            self.tool.run(None, ImageDocument())
        load_model.assert_not_called()

    def test_run_displays_an_image_even_when_no_objects_are_detected(self):
        annotated = Image.new("RGB", (200, 100), "blue")
        with patch(f"{self.module}.dialog_options", return_value="Detect objects"), \
             patch(f"{self.module}.filedialog.askopenfilename", return_value="blood_cells.pt"), \
             patch.object(self.tool, "_load_model"), \
             patch.object(self.tool, "_detect_image", return_value=([], annotated)):
            result = self.tool.run(None, self._document())
        self.assertIs(result.image, annotated)
        self.assertEqual(result.details["Detections"], [])

    def test_loading_a_model_uses_ultralytics_lazily(self):
        ultralytics = ModuleType("ultralytics")
        model = SimpleNamespace(names={0: "PC"}, task="detect")
        ultralytics.YOLO = Mock(return_value=model)
        with patch.dict(sys.modules, {"ultralytics": ultralytics}):
            self.tool._load_model("blood_cells.pt")
        ultralytics.YOLO.assert_called_once_with("blood_cells.pt")
        self.assertIs(self.tool.model, model)
        self.assertEqual(self.tool.model_path, Path("blood_cells.pt"))

    def test_missing_ultralytics_has_an_installation_message(self):
        with patch.dict(sys.modules, {"ultralytics": None}), \
             self.assertRaisesRegex(RuntimeError, "(?i)install|pip"):
            self.tool._load_model("blood_cells.pt")

    def test_ground_truth_converts_normalized_boxes_to_pixel_coordinates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "unseen.txt"
            path.write_text("1 0.5 0.5 0.4 0.2\n\n0 0.1 0.2 0.1 0.2\n", encoding="utf-8")
            boxes = self.tool._read_ground_truth(path, (200, 100))
        self.assertEqual([box["class_id"] for box in boxes], [1, 0])
        self.assertEqual([box["class_name"] for box in boxes], ["RBC", "PC"])
        np.testing.assert_allclose(boxes[0]["bbox"], [60, 40, 140, 60])
        np.testing.assert_allclose(boxes[1]["bbox"], [10, 10, 30, 30])

    def test_an_empty_ground_truth_file_is_a_valid_negative_image(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "negative.txt"
            path.write_text("\n", encoding="utf-8")
            self.assertEqual(self.tool._read_ground_truth(path, (200, 100)), [])

    def test_corrupt_ground_truth_reports_the_invalid_line(self):
        invalid_rows = [
            "1 0.5 0.5 0.2",  # Missing height.
            "1 bad 0.5 0.2 0.2",
            "1 nan 0.5 0.2 0.2",
            "1 0.5 inf 0.2 0.2",
            "1.5 0.5 0.5 0.2 0.2",
            "9 0.5 0.5 0.2 0.2",  # Not one of the model's classes.
            "1 1.2 0.5 0.2 0.2",
            "1 0.05 0.5 0.2 0.2",  # Center is valid, but the box crosses the edge.
            "1 0.5 0.5 -0.2 0.2",
            "1 0.5 0.5 0 0.2",
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "unseen.txt"
            for row in invalid_rows:
                with self.subTest(row=row):
                    path.write_text(f"0 0.5 0.5 0.1 0.1\n{row}\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "(?i)line.?2"):
                        self.tool._read_ground_truth(path, (200, 100))

    def test_comparison_returns_gt_and_predictions_side_by_side_in_rgb(self):
        original = Image.new("RGB", (200, 100), "red")
        predicted = Image.new("RGB", original.size, "blue")
        detections = [{"Class": "RBC", "Confidence": "80.0%", "Bounding box": [60, 40, 140, 60]}]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "unseen.txt"
            path.write_text("1 0.5 0.5 0.4 0.2\n", encoding="utf-8")
            with patch.object(self.tool, "_detect_image", return_value=(detections, predicted)):
                actual_detections, ground_truth, comparison = self.tool._compare_image(
                    np.asarray(original), path
                )
        self.assertEqual(actual_detections, detections)
        self.assertEqual(ground_truth[0]["class_name"], "RBC")
        np.testing.assert_allclose(ground_truth[0]["bbox"], [60, 40, 140, 60])
        self.assertIsInstance(comparison, Image.Image)
        self.assertEqual(comparison.mode, "RGB")
        self.assertEqual(comparison.width, 2 * original.width)
        self.assertGreaterEqual(comparison.height, original.height)
        self.assertEqual(comparison.getpixel((100, comparison.height - 1)), (255, 0, 0))
        self.assertEqual(comparison.getpixel((300, comparison.height - 1)), (0, 0, 255))
        body_top = comparison.height - original.height
        gt_box_region = np.asarray(comparison)[body_top + 40:body_top + 61, 60:141]
        self.assertTrue(np.any(gt_box_region != [255, 0, 0]), "The GT bounding box must be drawn.")

    def test_comparison_uses_the_original_image_that_matches_the_labels(self):
        document = self._document("red")
        document.current = Image.new("RGB", document.original.size, "blue")
        composite = Image.new("RGB", (400, 130))
        with patch(f"{self.module}.dialog_options", return_value="Compare ground truth and predictions"), \
             patch(f"{self.module}.filedialog.askopenfilename", side_effect=["blood_cells.pt", "unseen.txt"]), \
             patch.object(self.tool, "_load_model"), \
             patch.object(self.tool, "_compare_image", return_value=([], [], composite)) as compare:
            result = self.tool.run(None, document)
        np.testing.assert_array_equal(compare.call_args.args[0], np.asarray(document.original))
        self.assertEqual(Path(compare.call_args.args[1]), Path("unseen.txt"))
        self.assertIs(result.image, composite)

    def test_cancelling_before_or_after_selecting_weights_does_not_predict(self):
        cases = [(None, []), ("Detect objects", [""]),
                 ("Compare ground truth and predictions", ["blood_cells.pt", ""])]
        for option, paths in cases:
            with self.subTest(option=option), \
                 patch(f"{self.module}.dialog_options", return_value=option), \
                 patch(f"{self.module}.filedialog.askopenfilename", side_effect=paths), \
                 patch.object(self.tool, "_load_model"), \
                 patch.object(self.tool, "_predict") as predict, \
                 patch.object(self.tool, "_compare_image") as compare:
                result = self.tool.run(None, self._document())
            self.assertIsNone(result)
            predict.assert_not_called()
            compare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
