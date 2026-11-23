"""Extract test-image features and visualize saved vectors with t-SNE and UMAP."""

from pathlib import Path
from tkinter import filedialog
from zipfile import BadZipFile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE

from ..base import ToolResult
from ..utilities import dialog_options
from .model_load import ClassifyWithPretrainedModel


class FeatureProjection(ClassifyWithPretrainedModel):
    tool_id = "featureprojection"
    title = "Feature projection"
    description = "Save test features and visualize them using t-SNE and UMAP."
    requires_image = False

    def __init__(self):
        super().__init__()
        self.options = ["Extract and save features", "Load and visualize features"]

    def _extract_features(self, folder):
        import tensorflow as tf

        root = Path(folder)
        if not root.is_dir():
            raise ValueError("Select an existing folder of test images.")
        if len(self.model.layers) < 2:
            raise ValueError("The model needs a feature layer before its classification output.")
        # Loaded Sequential models expose inputs even when their .input property is unset.
        extractor = tf.keras.Model(self.model.inputs[0], self.model.layers[-2].output)
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        batches, images, labels, paths = [], [], [], []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions or path.name.startswith("."):
                continue
            try:
                with Image.open(path) as image:
                    image = np.asarray(image.convert("RGB"), dtype=np.float32)
            except OSError:
                continue
            images.append(self._preprocess_image(image))
            labels.append(path.relative_to(root).parts[0] if path.parent != root else "Unlabelled")
            paths.append(path.relative_to(root).as_posix())
            if len(images) == 32:
                batches.append(extractor(np.stack(images), training=False).numpy())
                images = []
        if images:
            batches.append(extractor(np.stack(images), training=False).numpy())
        if not batches:
            raise ValueError("No valid images found in the test folder.")
        features = np.concatenate(batches, axis=0)
        features = features.reshape(len(labels), -1)
        labels, paths = np.asarray(labels), np.asarray(paths)
        self._validate_features(features, labels, paths)
        return features, labels, paths

    @staticmethod
    def _validate_features(features, labels, paths=None):
        if (features.ndim != 2 or not all(features.shape)
                or features.dtype.kind not in "fiu" or not np.isfinite(features).all()):
            raise ValueError("Features must be a non-empty matrix of finite numbers.")
        for name, values in (("Labels", labels), ("Paths", paths)):
            if values is None and name == "Paths":
                continue
            if values.ndim != 1 or len(values) != len(features) or values.dtype.kind not in "US":
                raise ValueError(f"{name} must contain one text value per feature vector.")

    def _load_features(self, path):
        try:
            data = np.load(path, allow_pickle=False)
            if not isinstance(data, np.lib.npyio.NpzFile):
                raise ValueError("Select a .npz file containing features, labels and paths.")
            with data:
                features, labels, paths = data["features"], data["labels"], data["paths"]
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
            raise ValueError("Could not load test features. Select a valid .npz file with features, labels and paths.") from error
        self._validate_features(features, labels, paths)
        self.features, self.labels, self.paths = features, labels, paths

    def _visualize_features(self):
        self._validate_features(self.features, self.labels)
        count = len(self.features)
        if count < 3:
            raise ValueError("At least three test images are needed for the projections.")
        from umap import UMAP

        self.projections = {
            "t-SNE": TSNE(n_components=2, perplexity=min(30, count - 1),
                          init="random", learning_rate="auto", random_state=42).fit_transform(self.features),
            "UMAP": UMAP(n_components=2, n_neighbors=min(15, count - 1),
                         init="random", random_state=42, n_jobs=1).fit_transform(self.features),
        }
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for axis, (method, points) in zip(axes, self.projections.items()):
            for label in np.unique(self.labels):
                selected = self.labels == label
                axis.scatter(points[selected, 0], points[selected, 1], label=label, s=15, alpha=0.7)
            axis.set(title=method, xlabel="Dimension 1", ylabel="Dimension 2")
            axis.legend()
        figure.tight_layout()
        plt.show()
        plt.close(figure)

    def run(self, parent, document):
        option = dialog_options(parent, self.title, "Choose an option", self.options)
        if option is None:
            return None
        if option == self.options[0]:
            model_path = filedialog.askopenfilename(
                title="Select trained model", parent=parent,
                filetypes=[("Trained model", "*.zip *.keras")])
            if not model_path:
                return None
            folder = filedialog.askdirectory(title="Select test images", parent=parent, mustexist=True)
            if not folder:
                return None
            path = filedialog.asksaveasfilename(
                title="Save test features", parent=parent, defaultextension=".npz",
                filetypes=[("Feature vectors", "*.npz")])
            if not path:
                return None
            self._load_image(model_path)
            self.features, self.labels, self.paths = self._extract_features(folder)
            np.savez_compressed(path, features=self.features, labels=self.labels, paths=self.paths)
            message = "Test features saved."
        else:
            path = filedialog.askopenfilename(
                title="Load test features", parent=parent, filetypes=[("Feature vectors", "*.npz")])
            if not path:
                return None
            self._load_features(path)
            self._visualize_features()
            message = "t-SNE and UMAP projections displayed."
        return ToolResult(message=message, details={
            "Images": len(self.features), "Feature dimensions": self.features.shape[1], "File": str(path),
        })
