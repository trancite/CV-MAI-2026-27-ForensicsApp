"""Extract test-image features and visualize saved vectors with t-SNE and UMAP."""

from pathlib import Path
from tkinter import filedialog

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
        self.options = ["Extract and save features", "Load and visualize features"]

    def _extract_features(self, folder):
        import tensorflow as tf

        extractor = tf.keras.Model(self.model.input, self.model.layers[-2].output)
        root = Path(folder)
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        batches, images, labels, paths = [], [], [], []
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in extensions or path.name.startswith("."):
                continue
            try:
                with Image.open(path) as image:
                    image = np.asarray(image.convert("RGB"), dtype=np.float32)
            except OSError:
                continue
            image = tf.image.resize(image, self.model.input_shape[1:3]).numpy()
            images.append(image / 127.5 - 1)
            labels.append(path.relative_to(root).parts[0] if path.parent != root else "Unlabelled")
            paths.append(path.relative_to(root).as_posix())
            if len(images) == 32:
                batches.append(extractor(np.stack(images), training=False).numpy())
                images = []
        if images:
            batches.append(extractor(np.stack(images), training=False).numpy())
        if not batches:
            raise ValueError("No valid images found in the test folder.")
        return np.vstack(batches), np.asarray(labels), np.asarray(paths)

    def _load_features(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.features = data["features"]
            self.labels = data["labels"]
            self.paths = data["paths"]

    def _visualize_features(self):
        from umap import UMAP

        count = len(self.features)
        if count < 3:
            raise ValueError("At least three test images are needed for the projections.")
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
                title="Select model ZIP", parent=parent, filetypes=[("Model ZIP", "*.zip")])
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
