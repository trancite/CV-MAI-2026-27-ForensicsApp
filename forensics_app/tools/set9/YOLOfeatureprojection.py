
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
from .YOLOmodel_load import YOLOModelLoader


class YOLOFeatureProjection(YOLOModelLoader):

    tool_id = "yolofeatureprojection"
    title = "YOLO feature projection"
    description = (
        "Extract test features and visualize them using "
        "t-SNE and UMAP."
    )
    requires_image = False

    def __init__(self):
        super().__init__()

        self.options = [
            "Extract and save features",
            "Load and visualize features"
        ]

        self.features = None
        self.labels = None
        self.paths = None
        self.projections = {}

    # --------------------------------------------------
    # FIND IMAGE AND LABEL FOLDERS
    # --------------------------------------------------

    def _get_dataset_folders(self, folder):

        root = Path(folder)

        # User selected test/images directly
        if root.name.lower() == "images":

            images_root = root
            labels_root = root.parent / "labels"

        # User selected test
        elif (
            (root / "images").is_dir()
            and (root / "labels").is_dir()
        ):

            images_root = root / "images"
            labels_root = root / "labels"

        else:
            raise ValueError(
                "Invalid dataset folder. Select the test folder "
                "or its images folder. Both images and labels "
                "directories must exist."
            )

        if not labels_root.is_dir():
            raise FileNotFoundError(
                f"Labels folder not found: {labels_root}"
            )

        return images_root, labels_root

    # --------------------------------------------------
    # READ YOLO LABELS
    # --------------------------------------------------

    def _get_image_label(
        self,
        image_path,
        images_root,
        labels_root
    ):

        relative_path = image_path.relative_to(images_root)

        label_path = (
            labels_root / relative_path
        ).with_suffix(".txt")

        if not label_path.exists():
            return "Unlabelled"

        class_ids = set()

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                parts = line.strip().split()

                if not parts:
                    continue

                try:
                    class_id = int(parts[0])
                    class_ids.add(class_id)

                except ValueError:
                    continue

        if not class_ids:
            return "No objects"

        # Get class names from the trained YOLO model
        names = self.model.names

        class_names = []

        for class_id in sorted(class_ids):

            if isinstance(names, dict):
                class_name = names.get(class_id)
            else:
                class_name = (
                    names[class_id]
                    if 0 <= class_id < len(names)
                    else None
                )

            if class_name is not None:
                class_names.append(str(class_name))

        if not class_names:
            return "Unknown"

        return ", ".join(class_names)

    # --------------------------------------------------
    # EXTRACT YOLO FEATURES
    # --------------------------------------------------

    def _extract_features(self, folder):

        import torch

        images_root, labels_root = (
            self._get_dataset_folders(folder)
        )

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

        # Register hook on detection head
        hook = detection_head.register_forward_pre_hook(
            capture_features
        )

        try:

            image_files = sorted(
                image_path
                for image_path in images_root.rglob("*")
                if (
                    image_path.suffix.lower() in extensions
                    and not image_path.name.startswith(".")
                )
            )

            if not image_files:
                raise ValueError(
                    "No images found in the selected folder."
                )

            for image_path in image_files:

                try:

                    with Image.open(image_path) as image:

                        image_array = np.asarray(
                            image.convert("RGB")
                        )

                    # Run YOLO prediction
                    self.model.predict(
                        source=image_array,
                        verbose=False,
                        conf=0.25
                    )

                    if not captured:
                        print(
                            f"No features extracted: {image_path}"
                        )
                        continue

                    # Concatenate multi-scale feature maps
                    vector = np.concatenate(captured)

                    batches.append(vector)

                    # Read real class labels from YOLO .txt files
                    label = self._get_image_label(
                        image_path,
                        images_root,
                        labels_root
                    )

                    labels.append(label)

                    paths.append(
                        image_path.relative_to(
                            images_root
                        ).as_posix()
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

        features = np.vstack(batches)

        labels = np.asarray(labels, dtype=str)

        paths = np.asarray(paths, dtype=str)

        print("\nYOLO FEATURE EXTRACTION")
        print("-----------------------")
        print(f"Images processed: {len(features)}")

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        for label, count in zip(unique_labels, counts):
            print(f"{label}: {count}")

        return features, labels, paths

    # --------------------------------------------------
    # LOAD SAVED FEATURES
    # --------------------------------------------------

    def _load_features(self, path):

        with np.load(
            path,
            allow_pickle=False
        ) as data:

            self.features = data["features"]
            self.labels = data["labels"]
            self.paths = data["paths"]

    # --------------------------------------------------
    # VISUALIZE FEATURES WITH t-SNE AND UMAP
    # --------------------------------------------------

    def _visualize_features(self):

        from umap import UMAP

        count = len(self.features)

        if count < 3:
            raise ValueError(
                "At least three test images are needed."
            )

        if len(self.labels) != count:
            raise ValueError(
                "The number of labels does not match "
                "the number of feature vectors."
            )

        # t-SNE projection
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, count - 1),
            init="random",
            learning_rate="auto",
            random_state=42
        )

        tsne_points = tsne.fit_transform(
            self.features
        )

        # UMAP projection
        umap = UMAP(
            n_components=2,
            n_neighbors=min(15, count - 1),
            init="random",
            random_state=42,
            n_jobs=1
        )

        umap_points = umap.fit_transform(
            self.features
        )

        self.projections = {
            "t-SNE": tsne_points,
            "UMAP": umap_points
        }

        # Plot both projections
        figure, axes = plt.subplots(
            1, 2,
            figsize=(12, 5)
        )

        unique_labels = np.unique(self.labels)

        for axis, (method, points) in zip(
            axes,
            self.projections.items()
        ):

            for label in unique_labels:

                selected = self.labels == label

                axis.scatter(
                    points[selected, 0],
                    points[selected, 1],
                    label=label,
                    s=20,
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

    # --------------------------------------------------
    # MAIN TOOL
    # --------------------------------------------------

    def run(self, parent, document):

        option = dialog_options(
            parent,
            self.title,
            "Choose an option",
            self.options
        )

        if option is None:
            return None

        # ----------------------------------------------
        # OPTION 1: EXTRACT AND SAVE FEATURES
        # ----------------------------------------------

        if option == self.options[0]:

            model_path = filedialog.askopenfilename(
                title="Select trained YOLO model",
                parent=parent,
                filetypes=[
                    ("YOLO model", "*.pt")
                ]
            )

            if not model_path:
                return None

            folder = filedialog.askdirectory(
                title="Select test folder or test/images",
                parent=parent,
                mustexist=True
            )

            if not folder:
                return None

            path = filedialog.asksaveasfilename(
                title="Save test features",
                parent=parent,
                defaultextension=".npz",
                filetypes=[
                    ("Feature vectors", "*.npz")
                ]
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

            message = (
                "YOLO test features extracted and saved."
            )

        # ----------------------------------------------
        # OPTION 2: LOAD AND VISUALIZE FEATURES
        # ----------------------------------------------

        else:

            path = filedialog.askopenfilename(
                title="Load test features",
                parent=parent,
                filetypes=[
                    ("Feature vectors", "*.npz")
                ]
            )

            if not path:
                return None

            self._load_features(path)

            self._visualize_features()

            message = (
                "t-SNE and UMAP projections displayed."
            )

        # ----------------------------------------------
        # RETURN RESULTS
        # ----------------------------------------------

        return ToolResult(
            message=message,
            details={
                "Images": len(self.features),
                "Feature dimensions": (
                    self.features.shape[1]
                ),
                "Labels": {
                    str(label): int(count)
                    for label, count in zip(
                        *np.unique(
                            self.labels,
                            return_counts=True
                        )
                    )
                },
                "File": str(path)
            }
        )
