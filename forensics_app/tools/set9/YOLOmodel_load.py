
"""Inspect a trained YOLO model's passport or detect objects in the current image."""

from pathlib import Path
from tkinter import filedialog

import numpy as np

from ..base import ForensicsTool, ToolResult
from ..utilities import dialog_options


class YOLOModelLoader(ForensicsTool):
    tool_id = "yolomodelloader"
    title = "Object detection with YOLO."
    category = "Set9"
    description = "Load a trained YOLO model and detect objects in images."
    requires_image = True

    def __init__(self):
        self.options = ["Inspect model", "Detect objects"]
        self.model = None
        self.model_path = None

    def _load_model(self, path: str):
        from ultralytics import YOLO

        self.model_path = Path(path)
        self.model = YOLO(str(self.model_path))

    def _show_model_info(self):
        info = {
            "Model": self.model_path.name,
            "Architecture": self.model.task,
            "Classes": self.model.names,
            "Number of classes": len(self.model.names),
        }

        print("YOLO MODEL PASSPORT")
        print("-------------------")
        import json
        print(json.dumps(info, indent=2, ensure_ascii=False))

        return info

    def _detect_image(self, image: np.ndarray):
        results = self.model.predict(
            source=image,
            conf=0.25,
            imgsz=640,
            verbose=False,
        )

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                coordinates = box.xyxy[0].tolist()

                detections.append({
                    "Class": self.model.names[class_id],
                    "Confidence": f"{confidence:.1%}",
                    "Bounding box": [round(value, 2) for value in coordinates],
                })

            # Ultralytics returns an annotated image in BGR format.
            annotated_image = result.plot()

        return detections, annotated_image

    def run(self, parent, document):
        option = dialog_options(
            parent,
            self.title,
            "Choose an option",
            self.options,
        )

        if option is None:
            return None

        path = filedialog.askopenfilename(
            title="Select trained YOLO model",
            parent=parent,
            filetypes=[("YOLO model", "*.pt")],
        )

        if not path:
            return None

        self._load_model(path)

        if option == self.options[0]:
            info = self._show_model_info()

            return ToolResult(
                message="YOLO model information displayed.",
                details=info,
            )

        image = np.asarray(document.current.convert("RGB"))

        detections, annotated_image = self._detect_image(image)

        if not detections:
            return ToolResult(
                message="No objects detected.",
                details={"Detections": []},
            )

        return ToolResult(
            message=f"{len(detections)} objects detected.",
            details={"Detections": detections},
            image=annotated_image,
        )
