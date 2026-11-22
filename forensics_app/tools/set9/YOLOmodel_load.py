
"""Load a trained YOLO model and perform object detection."""

from pathlib import Path
from tkinter import filedialog

import numpy as np

from ..base import ForensicsTool, ToolResult
from ..utilities import dialog_options


class YOLOModelLoader(ForensicsTool):
    tool_id = "yolomodelloader"
    title = "YOLO model loader"
    category = "Set9"
    description = "Load a trained YOLO model and detect objects."
    requires_image = True

    def __init__(self):
        self.model = None
        self.options = ["Inspect model", "Detect objects"]

    def _load_model(self, path):
        from ultralytics import YOLO

        self.model = YOLO(path)

    def _show_model_info(self):
        print("YOLO model information")
        print("----------------------")
        print("Model:", self.model.ckpt_path)
        print("Classes:", self.model.names)

    def _detect_image(self, image):
        results = self.model.predict(
            source=image,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                coordinates = box.xyxy[0].cpu().numpy().tolist()

                detections.append({
                    "Class": self.model.names[cls],
                    "Confidence": f"{confidence:.1%}",
                    "Bounding box": coordinates
                })

        return detections, result.plot()

    def run(self, parent, document):
        option = dialog_options(
            parent,
            self.title,
            "Choose an option",
            self.options
        )

        if option is None:
            return None

        path = filedialog.askopenfilename(
            title="Select trained YOLO model",
            parent=parent,
            filetypes=[("YOLO model", "*.pt")]
        )

        if not path:
            return None

        self._load_model(path)

        if option == self.options[0]:
            self._show_model_info()

            return ToolResult(
                message="YOLO model loaded successfully.",
                details={
                    "Model": Path(path).name,
                    "Classes": self.model.names
                }
            )

        image = np.asarray(document.current.convert("RGB"))

        detections, annotated_image = self._detect_image(image)

        return ToolResult(
            message=f"{len(detections)} objects detected.",
            details={
                "Detections": detections,
                "Annotated image": annotated_image
            }
        )
