"""Save pooled YOLO features and project them with their true dataset labels."""

from pathlib import Path
from tkinter import filedialog

import numpy as np
from PIL import Image

from ..base import ToolResult
from ..utilities import dialog_options
from ..set8.feature_projection import FeatureProjection
from .YOLOmodel_load import YOLOModelLoader


class YOLOFeatureProjection(YOLOModelLoader):
    tool_id = "yolofeatureprojection"
    title = "YOLO feature projection"
    description = "Save YOLO test features and visualize them using t-SNE and UMAP."
    requires_image = False

    def __init__(self):
        super().__init__()
        self.options = ["Extract and save features", "Load and visualize features"]
        self.features = None
        self.labels = None
        self.paths = None
        self.projections = {}

    def _get_dataset_folders(self, folder):
        root = Path(folder)
        if root.name.lower() == "images":
            images_root, labels_root = root, root.parent / "labels"
        else:
            images_root, labels_root = root / "images", root / "labels"
        if not images_root.is_dir() or not labels_root.is_dir():
            raise ValueError("Select a test folder with images and labels directories, or its images directory.")
        return images_root, labels_root

    def _get_image_label(self, image_path, images_root, labels_root):
        label_path = (labels_root / image_path.relative_to(images_root)).with_suffix(".txt")
        if not label_path.is_file():
            return "Unlabelled"
        names = self.model.names
        class_ids = set()
        for number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError as error:
                raise ValueError(f"Invalid class ID in {label_path.name}, line {number}.") from error
            if class_id < 0 or (class_id not in names if isinstance(names, dict) else class_id >= len(names)):
                raise ValueError(f"Unknown class ID {class_id} in {label_path.name}, line {number}.")
            class_ids.add(class_id)
        if not class_ids:
            return "No objects"
        return ", ".join(str(names[class_id]) for class_id in sorted(class_ids))

    def _extract_features(self, folder):
        images_root, labels_root = self._get_dataset_folders(folder)
        if self.model is None:
            raise ValueError("Load a YOLO detection model before extracting features.")
        detection_head = self.model.model.model[-1]
        captured = []

        def capture_features(module, inputs):
            captured.clear()
            feature_maps = inputs[0]
            if not isinstance(feature_maps, (list, tuple)):
                feature_maps = [feature_maps]
            for feature_map in feature_maps:
                if (not hasattr(feature_map, "detach") or len(feature_map.shape) != 4
                        or feature_map.shape[0] != 1):
                    raise ValueError("Expected a single-image feature map from the YOLO detection head.")
                pooled = feature_map.detach().float().mean(dim=(2, 3))
                captured.append(pooled.cpu().numpy().ravel())

        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        vectors, labels, paths = [], [], []
        hook = detection_head.register_forward_pre_hook(capture_features)
        try:
            for image_path in sorted(images_root.rglob("*")):
                relative = image_path.relative_to(images_root)
                if (not image_path.is_file() or image_path.suffix.lower() not in extensions
                        or any(part.startswith(".") or part == "__MACOSX" for part in relative.parts)):
                    continue
                try:
                    with Image.open(image_path) as image:
                        image_array = np.asarray(image.convert("RGB"))
                    label = self._get_image_label(image_path, images_root, labels_root)
                except (OSError, ValueError) as error:
                    print(f"Skipping {relative.as_posix()}: {error}")
                    continue
                captured.clear()
                # The same RGB-safe prediction path is used for detection and extraction.
                self._predict(image_array)
                if not captured:
                    raise ValueError("The YOLO detection head did not return feature maps.")
                vector = np.concatenate(captured)
                vectors.append(vector)
                labels.append(label)
                paths.append(relative.as_posix())
        finally:
            hook.remove()
        if not vectors:
            raise ValueError("No valid image features were extracted. Check the images and YOLO labels.")
        features, labels, paths = np.vstack(vectors), np.asarray(labels), np.asarray(paths)
        self._validate_features(features, labels, paths)
        return features, labels, paths

    # NPZ validation and 2D visualization are shared; model loading remains YOLO-specific.
    @staticmethod
    def _validate_features(features, labels, paths=None):
        FeatureProjection._validate_features(features, labels, paths)

    def _load_features(self, path):
        FeatureProjection._load_features(self, path)

    def _visualize_features(self):
        FeatureProjection._visualize_features(self)

    def run(self, parent, document):
        option = dialog_options(parent, self.title, "Choose an option", self.options)
        if option is None:
            return None
        if option == self.options[0]:
            model_path = filedialog.askopenfilename(
                title="Select trained YOLO model", parent=parent, filetypes=[("YOLO model", "*.pt")])
            if not model_path:
                return None
            folder = filedialog.askdirectory(
                title="Select test folder or test/images", parent=parent, mustexist=True)
            if not folder:
                return None
            path = filedialog.asksaveasfilename(
                title="Save test features", parent=parent, defaultextension=".npz",
                filetypes=[("Feature vectors", "*.npz")])
            if not path:
                return None
            self._load_model(model_path)
            self.features, self.labels, self.paths = self._extract_features(folder)
            np.savez_compressed(path, features=self.features, labels=self.labels, paths=self.paths)
            message = "YOLO test features extracted and saved."
        else:
            path = filedialog.askopenfilename(
                title="Load test features", parent=parent, filetypes=[("Feature vectors", "*.npz")])
            if not path:
                return None
            self._load_features(path)
            self._visualize_features()
            message = "t-SNE and UMAP projections displayed."
        return ToolResult(message=message, details={
            "Images": len(self.features), "Feature dimensions": self.features.shape[1],
            "Labels": {str(label): int(count) for label, count in zip(*np.unique(self.labels, return_counts=True))},
            "File": str(path),
        })
