"""Compute PCA from training images and display the leading eigenfaces."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult


class EigenFaces(ForensicsTool):
    tool_id = "eigenfaces"
    title = "Eigenfaces"
    category = "Set6"
    description = "Compute PCA from training images and visualize eigenfaces."
    requires_image = False

    def _choose_set(self, parent: tk.Misc, current_path: Path | None, set_name: str) -> str:
        return filedialog.askdirectory(
            title=f"Select the {set_name} folder", parent=parent, mustexist=True,
            initialdir=str(current_path.parent if current_path else Path.cwd()),
        )

    def _load_images(self, images_paths: list) -> np.ndarray:
        rows = []
        loaded_paths = []
        image_shape = None

        for path in images_paths:
            try:
                with Image.open(path) as candidate_file:
                    candidate_image = np.asarray(candidate_file.convert("RGB"), dtype=np.float32)
            except (OSError, RuntimeError, ValueError):
                continue

            if image_shape is None:
                image_shape = candidate_image.shape
            elif candidate_image.shape != image_shape:
                raise ValueError(f"All images must have the same size. Unexpected size in: {path}")

            rows.append(candidate_image.ravel())
            loaded_paths.append(Path(path).resolve())

        if not rows:
            raise ValueError("No valid images could be loaded.")

        self.loaded_paths_ = loaded_paths
        self.image_shape_ = image_shape
        return np.stack(rows)

    def _choose_pov(self, parent: tk.Misc) -> float | None:
        percentage = simpledialog.askfloat(
            "PCA variance", "Percentage of variance to preserve (POV, 0.1-100%):",
            initialvalue=95.0, minvalue=0.1, maxvalue=100.0, parent=parent,
        )
        return None if percentage is None else percentage / 100.0

    def _perform_PCA(
        self, X: np.ndarray, n_components: int | None = None, pov: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0:
            raise ValueError("PCA requires a matrix with at least two images and one feature.")
        if not np.isfinite(X).all():
            raise ValueError("PCA input must contain only finite pixel values.")
        if n_components is not None and pov is not None:
            raise ValueError("Choose either a component count or a POV target.")
        if pov is not None and (not np.isfinite(pov) or not 0 < pov <= 1):
            raise ValueError("POV must be a fraction greater than 0 and at most 1.")

        max_components = min(X.shape[0] - 1, X.shape[1])
        if n_components is not None and (
            isinstance(n_components, (bool, np.bool_))
            or not isinstance(n_components, (int, np.integer))
            or not 1 <= n_components <= max_components
        ):
            raise ValueError(f"The component count must be an integer from 1 to {max_components}.")

        mean = X.mean(axis=0)
        centered = X - mean
        _, singular_values, vectors_t = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = singular_values[:max_components] ** 2 / (X.shape[0] - 1)
        total_variance = eigenvalues.sum()
        if total_variance <= 0:
            raise ValueError("PCA requires variation between the training images.")
        ratios = eigenvalues / total_variance
        if pov is not None:
            cumulative = np.cumsum(ratios)
            cumulative[-1] = 1.0
            count = int(np.searchsorted(cumulative, pov)) + 1
        else:
            count = max_components if n_components is None else int(n_components)

        components = vectors_t[:count]
        coefficients = centered @ components.T
        self.mean_ = mean
        self.components_ = components
        self.eigenvalues_ = eigenvalues[:count]
        self.explained_variance_ratio_ = ratios[:count]
        self.coefficients_ = coefficients
        return coefficients, self.eigenvalues_, components.T

    def _visualize_eigenfaces(self, k: int) -> None:
        columns = min(k, 4)
        rows = (k + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
        try:
            figure.suptitle("First PCA eigenfaces")
            for index, axis in enumerate(axes.flat):
                axis.axis("off")
                if index >= k:
                    continue
                eigenface = self.components_[index].reshape(self.image_shape_)
                minimum, maximum = eigenface.min(), eigenface.max()
                scaled = (eigenface - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(eigenface)
                axis.imshow(scaled)
                axis.set_title(f"Eigenface {index + 1} ({self.explained_variance_ratio_[index]:.1%})")
            figure.tight_layout()
            plt.show()
        finally:
            plt.close(figure)

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        current_path = document.path.resolve() if document.path else None
        training_folder = self._choose_set(parent, current_path, "training faces")
        if not training_folder:
            return None
        pov = self._choose_pov(parent)
        if pov is None:
            return None

        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        paths = sorted(
            path for path in Path(training_folder).rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        X_train = self._load_images(paths)
        self._perform_PCA(X_train, pov=pov)

        count = len(self.components_)
        k = simpledialog.askinteger(
            "Eigenface visualization", f"How many of the {count} eigenfaces should be shown?",
            initialvalue=min(12, count), minvalue=1, maxvalue=count, parent=parent,
        )
        if k is None:
            return None
        self._visualize_eigenfaces(k)
        return ToolResult(
            message=f"Computed {count} PCA components; displayed the first {k} eigenfaces.",
            details={
                "Algorithm": "PCA", "Training images": len(X_train),
                "Image shape": self.image_shape_,
                "POV requested": f"{pov:.2%}",
                "POV retained": f"{self.explained_variance_ratio_.sum():.2%}",
                "Components": count, "Eigenfaces shown": k,
            },
        )
