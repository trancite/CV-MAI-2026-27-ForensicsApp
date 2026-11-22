
"""Extract YOLO test-image features and visualize them
using t-SNE and UMAP."""

from pathlib import Path
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE

from ..base import ToolResult
from ..utilities import dialog_options
from .model_load import YOLOModelLoader


class YOLOFeatureProjection(YOLOModelLoader):
    tool_id = "yolofeatureprojection"
    title = "YOLO feature projection"
    description = "Extract test features and visualize them using t-SNE and UMAP."
    requires_image = False

    def __init__(self):
        super().__init__()
        self.options = [
            "Extract and save features",
            "Load and visualize features"
        ]

    def _extract_features(self, folder):
        import torch

        root = Path(folder)

        extensions = {
            ".png", ".jpg", ".jpeg",
            ".bmp", ".tif", ".tiff", ".webp"
        }

        batches = []
        labels = []
        paths = []

        # YOLO detection head
        detection_head = self.model.model.model[-1]

        captured = []

        def capture_features(module, inputs):
            captured.clear()

            feature_maps = inputs[0]

            if isinstance(feature_maps, torch.Tensor):
                feature_maps = [feature_maps]

            for feature_map in feature_maps:
                if isinstance(feature_map, torch.Tensor):
                    # Global Average Pooling
                    pooled = feature_map.detach().float().mean(
                        dim=(2, 3)
                    )

                    captured.append(
                        pooled.cpu().numpy().flatten()
                    )

        # Hook the inputs of the detection head
        hook = detection_head.register_forward_pre_hook(
            capture_features
        )

        try:
            for image_path in sorted(root.rglob("*")):

                if (
                    image_path.suffix.lower() not in extensions
                    or image_path.name.startswith(".")
                ):
                    continue

                try:
                    with Image.open(image_path) as image:
                        image = np.asarray(
                            image.convert("RGB")
                        )

                    # Run YOLO prediction to obtain feature maps
                    self.model.predict(
                        source=image,
                        verbose=False,
                        conf=0.25
                    )

                    if not captured:
                        continue

                    # Concatenate the multi-scale feature maps
                    vector = np.concatenate(captured)

                    batches.append(vector)

                    labels.append(
                        image_path.relative_to(root).parts[0]
                        if image_path.parent != root
                        else "Unlabelled"
                    )

                    paths.append(
                        image_path.relative_to(root).as_posix()
                    )

                except (OSError, ValueError) as error:
                    print(
                        f"Could not process {image_path}: {error}"
                    )

        finally:
            hook.remove()

        if not batches:
            raise ValueError(
                "No valid image features were extracted."
            )

        return (
            np.vstack(batches),
            np.asarray(labels),
            np.asarray(paths)
        )

    def _load_features(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.features = data["features"]
            self.labels = data["labels"]
            self.paths = data["paths"]

    def _visualize_features(self):
        from umap import UMAP

        count = len(self.features)

        if count < 3:
            raise ValueError(
                "At least three test images are needed."
            )

        self.projections = {
            "t-SNE": TSNE(
                n_components=2,
                perplexity=min(30, count - 1),
                init="random",
                learning_rate="auto",
                random_state=42
            ).fit_transform(self.features),

            "UMAP": UMAP(
                n_components=2,
                n_neighbors=min(15, count - 1),
                init="random",
                random_state=42,
                n_jobs=1
            ).fit_transform(self.features)
        }

        figure, axes = plt.subplots(
            1, 2, figsize=(12, 5)
        )

        for axis, (method, points) in zip(
            axes, self.projections.items()
        ):

            for label in np.unique(self.labels):

                selected = self.labels == label

                axis.scatter(
                    points[selected, 0],
                    points[selected, 1],
                    label=label,
                    s=15,
                    alpha=0.7
                )

            axis.set(
                title=method,
                xlabel="Dimension 1",
                ylabel="Dimension 2"
            )

            axis.legend()

        figure.tight_layout()
        plt.show()
        plt.close(figure)

    def run(self, parent, document):

        option = dialog_options(
            parent,
            self.title,
            "Choose an option",
            self.options
        )

        if option is None:
            return None

        if option == self.options[0]:

            model_path = filedialog.askopenfilename(
                title="Select trained YOLO model",
                parent=parent,
                filetypes=[("YOLO model", "*.pt")]
            )

            if not model_path:
                return None

            folder = filedialog.askdirectory(
                title="Select test images",
                parent=parent,
                mustexist=True
            )

            if not folder:
                return None

            path = filedialog.asksaveasfilename(
                title="Save test features",
                parent=parent,
                defaultextension=".npz",
                filetypes=[("Feature vectors", "*.npz")]
            )

            if not path:
                return None

            self._load_model(model_path)

            (
                self.features,
                self.labels,
                self.paths
            ) = self._extract_features(folder)

            np.savez_compressed(
                path,
                features=self.features,
                labels=self.labels,
                paths=self.paths
            )

            message = "YOLO test features saved."

        else:

            path = filedialog.askopenfilename(
                title="Load test features",
                parent=parent,
                filetypes=[("Feature vectors", "*.npz")]
            )

            if not path:
                return None

            self._load_features(path)

            self._visualize_features()

            message = "t-SNE and UMAP projections displayed."

        return ToolResult(
            message=message,
            details={
                "Images": len(self.features),
                "Feature dimensions": self.features.shape[1],
                "File": str(path)
            }
        )
