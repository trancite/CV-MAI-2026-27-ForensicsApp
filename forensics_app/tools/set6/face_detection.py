"""Detect faces and five facial landmarks with the pretrained YuNet model."""

from __future__ import annotations

from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
import tkinter as tk
from tkinter import simpledialog
import urllib.request

import numpy as np
from PIL import ImageDraw

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult


class FaceDetection(ForensicsTool):
    tool_id = "face_detection"
    title = "Face Detection"
    category = "Set6"
    description = "Detect faces and facial landmarks using YuNet."
    requires_image = True

    model_url = (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    landmark_names = (
        "Right eye", "Left eye", "Nose", "Right mouth corner", "Left mouth corner",
    )

    def __init__(self, model_path: str | Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[3]
        filename = "face_detection_yunet_2023mar.onnx"
        course_model = project_root / "Set6_materials" / filename
        self.model_path = (
            Path(model_path) if model_path is not None else
            course_model if course_model.is_file() else project_root / "models" / filename
        )

    def _check_model(self) -> None:
        if self.model_path.exists():
            if not self.model_path.is_file() or self.model_path.stat().st_size < 100_000:
                raise RuntimeError(f"YuNet model is incomplete: {self.model_path}")
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with NamedTemporaryFile(dir=self.model_path.parent, suffix=".onnx", delete=False) as target:
                temporary_path = Path(target.name)
                with urllib.request.urlopen(self.model_url, timeout=30) as response:
                    shutil.copyfileobj(response, target)
            if temporary_path.stat().st_size < 100_000:
                raise RuntimeError("YuNet model download is incomplete.")
            temporary_path.replace(self.model_path)
        except OSError as error:
            raise RuntimeError(
                f"Could not download the YuNet model. Download it from {self.model_url} "
                f"and save it to {self.model_path}."
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _choose_parameters(self, parent: tk.Misc) -> tuple[float, float, int] | None:
        confidence_threshold = simpledialog.askfloat(
            "Confidence threshold", "Choose a confidence threshold (0-1):",
            initialvalue=0.6, minvalue=0, maxvalue=1, parent=parent,
        )
        if confidence_threshold is None:
            return None
        nms_threshold = simpledialog.askfloat(
            "Non-maximum suppression threshold", "Choose an NMS threshold (0-1):",
            initialvalue=0.3, minvalue=0, maxvalue=1, parent=parent,
        )
        if nms_threshold is None:
            return None
        maximum_number_of_candidates = simpledialog.askinteger(
            "Maximum number of candidates", "Choose the maximum candidates before NMS:",
            initialvalue=5000, minvalue=1, parent=parent,
        )
        if maximum_number_of_candidates is None:
            return None
        return confidence_threshold, nms_threshold, maximum_number_of_candidates

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        parameters = self._choose_parameters(parent)
        if parameters is None:
            return None

        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "Face detection requires OpenCV. Run: "
                "python -m pip install -r requirements.txt"
            ) from error

        self._check_model()
        output = document.current.convert("RGB")
        model_bytes = np.fromfile(str(self.model_path), dtype=np.uint8)
        detector = cv2.FaceDetectorYN.create(
            "onnx", model_bytes, np.empty(0, dtype=np.uint8), output.size, *parameters,
        )
        bgr = cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)
        _, faces = detector.detect(bgr)
        draw = ImageDraw.Draw(output)
        count = 0 if faces is None else len(faces)
        details = {
            "Detector": "YuNet",
            "Number of faces": count,
            "Confidence threshold": parameters[0],
            "NMS threshold": parameters[1],
            "Maximum candidates": parameters[2],
        }
        if faces is not None:
            for index, face in enumerate(faces, start=1):
                x, y, width, height = (int(value) for value in face[:4])
                confidence = float(face[14])
                draw.rectangle((x, y, x + width, y + height), outline="lime", width=2)
                draw.text((max(0, x), max(0, y - 14)), f"Face: {confidence:.2f}", fill="lime")
                details[f"Face {index} confidence"] = confidence
                details[f"Face {index} box (x, y, width, height)"] = (x, y, width, height)
                landmarks = face[4:14].reshape(5, 2)
                for name, (px, py) in zip(self.landmark_names, landmarks):
                    px, py = int(px), int(py)
                    draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill="blue")
                    details[f"Face {index} {name}"] = (px, py)

        return ToolResult(
            message=f"Detected {count} face(s)." if count else "No faces detected.",
            image=output,
            details=details,
        )
