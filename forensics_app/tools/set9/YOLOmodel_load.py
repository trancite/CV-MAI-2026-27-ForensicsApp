"""Inspect YOLO models, detect objects, and compare test labels with predictions."""

import json
from pathlib import Path
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageDraw

from ..base import ForensicsTool, ToolResult
from ..utilities import dialog_options


class YOLOModelLoader(ForensicsTool):
    tool_id = "yolomodelloader"
    title = "Object detection with YOLO"
    category = "Set9"
    description = "Inspect a YOLO detector, detect objects, or compare test annotations with predictions."
    requires_image = False

    def __init__(self):
        self.options = ["Inspect model", "Detect objects", "Compare ground truth and predictions"]
        self.model = None
        self.model_path = None

    def _load_model(self, path: str):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Install the YOLO dependencies with: python -m pip install '.[yolo]'") from error
        model = YOLO(str(path))
        if model.task != "detect":
            raise ValueError("Select a YOLO object detection model (.pt).")
        self.model = model
        self.model_path = Path(path)

    def _show_model_info(self):
        info = {
            "Model": self.model_path.name,
            "Task": self.model.task,
            "Classes": self.model.names,
            "Number of classes": len(self.model.names),
        }
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return info

    def _predict(self, image: np.ndarray):
        """Ultralytics accepts PIL in RGB; its NumPy input convention is BGR."""
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 image for YOLO prediction.")
        return self.model.predict(
            source=Image.fromarray(image), conf=0.25, imgsz=640, verbose=False,
        )

    def _detect_image(self, image: np.ndarray):
        results = self._predict(image)
        if len(results) != 1:
            raise ValueError("The detector must return one result for the selected image.")
        result = results[0]
        if result.boxes is None:
            raise ValueError("The selected model did not return object detection boxes.")
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "Class": self.model.names[class_id],
                "Confidence": f"{float(box.conf[0]):.1%}",
                "Bounding box": [round(value, 2) for value in box.xyxy[0].tolist()],
            })

        # plot() returns BGR NumPy pixels; the app's image contract is PIL RGB.
        plotted = result.plot()
        if isinstance(plotted, Image.Image):
            annotated_image = plotted.convert("RGB")
        else:
            annotated_image = Image.fromarray(np.asarray(plotted)[..., ::-1].copy())
        return detections, annotated_image

    def _read_ground_truth(self, label_path, image_size):
        """Read YOLO class/cx/cy/width/height labels into pixel coordinates."""
        width, height = image_size
        annotations = []
        for line_number, line in enumerate(Path(label_path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError("expected class_id cx cy width height")
                class_id = int(parts[0])
                names = self.model.names
                if isinstance(names, dict):
                    if class_id not in names:
                        raise ValueError("class ID is absent from the selected model")
                    class_name = names[class_id]
                else:
                    if not 0 <= class_id < len(names):
                        raise ValueError("class ID is absent from the selected model")
                    class_name = names[class_id]
                values = np.asarray(parts[1:], dtype=float)
                if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                    raise ValueError("coordinates must be finite normalized values in [0, 1]")
                cx, cy, box_width, box_height = values
                if box_width <= 0 or box_height <= 0:
                    raise ValueError("box width and height must be positive")
                normalized = np.array([
                    cx - box_width / 2, cy - box_height / 2,
                    cx + box_width / 2, cy + box_height / 2,
                ])
                if np.any(normalized < -1e-6) or np.any(normalized > 1 + 1e-6):
                    raise ValueError("bounding box extends outside the image")
                bbox = np.clip(normalized, 0, 1) * [width, height, width, height]
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid YOLO annotation at line {line_number}: {error}") from error
            annotations.append({"class_id": class_id, "class_name": class_name, "bbox": bbox.tolist()})
        return annotations

    def _compare_image(self, image: np.ndarray, label_path):
        source = Image.fromarray(image).convert("RGB")
        ground_truth = self._read_ground_truth(label_path, source.size)
        detections, predicted_image = self._detect_image(image)
        actual_image = source.copy()
        draw = ImageDraw.Draw(actual_image)
        line_width = max(1, round(min(source.size) / 200))
        for annotation in ground_truth:
            bbox = annotation["bbox"]
            draw.rectangle(bbox, outline=(0, 180, 0), width=line_width)
            draw.text((bbox[0], max(0, bbox[1] - 12)), annotation["class_name"], fill=(0, 180, 0))

        header_height = 32
        comparison = Image.new("RGB", (source.width * 2, source.height + header_height), (30, 30, 30))
        comparison.paste(actual_image, (0, header_height))
        comparison.paste(predicted_image, (source.width, header_height))
        header = ImageDraw.Draw(comparison)
        header.text((8, 8), "Ground truth", fill="white")
        header.text((source.width + 8, 8), "Predictions", fill="white")
        return detections, ground_truth, comparison

    @staticmethod
    def _default_label_path(image_path):
        if image_path is None:
            return None
        image_path = Path(image_path)
        for parent in image_path.parents:
            if parent.name.lower() == "images":
                return (parent.parent / "labels" / image_path.relative_to(parent)).with_suffix(".txt")
        return image_path.with_suffix(".txt")

    def run(self, parent, document):
        option = dialog_options(parent, self.title, "Choose an option", self.options)
        if option is None:
            return None
        if option != self.options[0] and (document is None or not document.is_loaded):
            raise ValueError("Open an image before running object detection.")

        path = filedialog.askopenfilename(
            title="Select YOLO detection model", parent=parent,
            initialdir=str(Path(__file__).parent), filetypes=[("YOLO model", "*.pt")],
        )
        if not path:
            return None
        label_path = None
        if option == self.options[2]:
            default = self._default_label_path(document.path)
            location = {"initialdir": str(default.parent), "initialfile": default.name} if default else {}
            label_path = filedialog.askopenfilename(
                title="Select ground truth YOLO labels for this image", parent=parent,
                filetypes=[("YOLO annotations", "*.txt")], **location,
            )
            if not label_path:
                return None

        self._load_model(path)
        if option == self.options[0]:
            return ToolResult(message="YOLO model information displayed.", details=self._show_model_info())
        if option == self.options[2]:
            # The label file belongs to the original image, before any app edits.
            source = document.original if document.original is not None else document.current
            image = np.asarray(source.convert("RGB"))
            detections, ground_truth, comparison = self._compare_image(image, label_path)
            return ToolResult(
                message=f"{len(ground_truth)} ground truth boxes; {len(detections)} predicted boxes.",
                image=comparison,
                details={"Ground truth boxes": len(ground_truth), "Predicted boxes": len(detections),
                         "Ground truth": ground_truth, "Detections": detections},
            )

        detections, annotated_image = self._detect_image(np.asarray(document.current.convert("RGB")))
        return ToolResult(
            message=f"{len(detections)} objects detected." if detections else "No objects detected.",
            details={"Detections": detections}, image=annotated_image,
        )
