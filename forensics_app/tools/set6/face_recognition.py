
"""Recognize a test face using PCA coefficients and nearest neighbours."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult


class FaceRecognition(ForensicsTool):
    tool_id = "face_recognition"
    title = "Face Recognition"
    category = "Set6"
    description = "Recognize a face using Eigenfaces and nearest neighbours."
    requires_image = False

    def _choose_set(
        self, parent: tk.Misc, current_path: Path | None
    ) -> str:
        return filedialog.askdirectory(
            title="Select the training faces folder",
            parent=parent,
            mustexist=True,
            initialdir=str(
                current_path.parent if current_path else Path.cwd()
            ),
        )

    def _choose_test_image(
        self, parent: tk.Misc, initial_dir: str
    ) -> str:
        return filedialog.askopenfilename(
            title="Select a test face image",
            parent=parent,
            initialdir=initial_dir,
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )

    def _load_images(self, images_paths: list, exclude: np.ndarray | None = None) -> np.ndarray:
        rows = []
        loaded_paths = []
        image_shape = None

        for path in images_paths:
            try:
                with Image.open(path) as candidate_file:
                    candidate_image = np.asarray(
                        candidate_file.convert("RGB"),
                        dtype=np.float32,
                    )
            except (OSError, RuntimeError, ValueError):
                continue

            if exclude is not None and np.array_equal(candidate_image, exclude):
                continue

            if image_shape is None:
                image_shape = candidate_image.shape
            elif candidate_image.shape != image_shape:
                raise ValueError(
                    f"All training images must have the same size. "
                    f"Unexpected size in: {path}"
                )

            rows.append(candidate_image.ravel())
            loaded_paths.append(Path(path).resolve())

        if len(rows) < 2:
            raise ValueError(
                "At least two valid training images are required."
            )

        self.loaded_paths_ = loaded_paths
        self.image_shape_ = image_shape

        return np.stack(rows)

    def _load_test_image(self, path: str) -> np.ndarray:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image = image.resize(
                    (self.image_shape_[1], self.image_shape_[0]),
                    Image.Resampling.LANCZOS,
                )
                return np.asarray(image, dtype=np.float32)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(
                f"Could not load test image: {path}"
            ) from error

    def _perform_PCA(
        self,
        X: np.ndarray,
        pov: float = 0.95,
    ) -> np.ndarray:

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("PCA requires at least two training images.")

        if not np.isfinite(X).all():
            raise ValueError("Training images contain invalid values.")

        if not 0 < pov <= 1:
            raise ValueError("POV must be between 0 and 1.")

        self.mean_ = X.mean(axis=0)

        centered = X - self.mean_

        _, singular_values, vectors_t = np.linalg.svd(
            centered,
            full_matrices=False,
        )

        max_components = min(X.shape[0] - 1, X.shape[1])

        eigenvalues = (
            singular_values[:max_components] ** 2
            / (X.shape[0] - 1)
        )

        total_variance = eigenvalues.sum()

        if total_variance <= 0:
            raise ValueError(
                "There is no variation between training images."
            )

        ratios = eigenvalues / total_variance

        cumulative = np.cumsum(ratios)
        cumulative[-1] = 1.0

        count = int(np.searchsorted(cumulative, pov)) + 1

        self.components_ = vectors_t[:count]
        self.eigenvalues_ = eigenvalues[:count]
        self.explained_variance_ratio_ = ratios[:count]

        self.coefficients_ = centered @ self.components_.T

        return self.coefficients_

    def _get_test_coefficients(
        self, test_image: np.ndarray
    ) -> np.ndarray:

        flattened = test_image.ravel().astype(np.float64)

        centered = flattened - self.mean_

        coefficients = centered @ self.components_.T

        return coefficients

    def _get_labels(
        self,
        training_folder: Path,
    ) -> list[str]:

        labels = []

        for path in self.loaded_paths_:
            relative_path = path.relative_to(training_folder)

            if len(relative_path.parts) > 1:
                label = relative_path.parts[0]
            else:
                label = path.stem

            labels.append(label)

        return labels

    def _find_closest_faces(
        self,
        test_coefficients: np.ndarray,
        n_neighbors: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:

        # Euclidean distance in PCA coefficient space
        differences = self.coefficients_ - test_coefficients

        distances = np.linalg.norm(differences, axis=1)

        indices = np.argsort(distances)[:n_neighbors]

        return indices, distances[indices]

    def _visualize_results(
        self,
        test_image: np.ndarray,
        indices: np.ndarray,
        distances: np.ndarray,
        labels: list[str],
    ) -> None:

        count = len(indices)

        figure, axes = plt.subplots(
            1,
            count + 1,
            figsize=(4 * (count + 1), 5),
            squeeze=False,
        )

        try:
            axes = axes.flat

            axes[0].imshow(test_image.astype(np.uint8))
            axes[0].set_title("Test image")
            axes[0].axis("off")

            for position, (index, distance) in enumerate(
                zip(indices, distances),
                start=1,
            ):
                with Image.open(self.loaded_paths_[index]) as image:
                    axes[position].imshow(
                        image.convert("RGB")
                    )

                axes[position].set_title(
                    f"Match {position}\n"
                    f"{labels[index]}\n"
                    f"Distance: {distance:.2f}"
                )
                axes[position].axis("off")

            figure.suptitle("Face Recognition - Closest Matches")
            figure.tight_layout()
            plt.show()

        finally:
            plt.close(figure)

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        current_path = (
            document.path.resolve() if document.path else None
        )

        # 1. Select training folder
        training_folder = self._choose_set(parent, current_path)

        if not training_folder:
            return None

        training_folder_path = Path(training_folder).resolve()

        # 2. Select test image
        test_path = self._choose_test_image(
            parent,
            training_folder,
        )

        if not test_path:
            return None

        extensions = {
            ".png", ".jpg", ".jpeg",
            ".bmp", ".tif", ".tiff", ".webp",
        }

        paths = sorted(
            path
            for path in training_folder_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
        )

        with Image.open(test_path) as image:
            test_pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        X_train = self._load_images(paths, exclude=test_pixels)

        pov = 0.95

        self._perform_PCA(X_train, pov=pov)

        test_image = self._load_test_image(test_path)

        test_coefficients = self._get_test_coefficients(
            test_image
        )

        labels = self._get_labels(training_folder_path)

        max_neighbors = min(3, len(X_train))

        n_neighbors = simpledialog.askinteger(
            "Nearest neighbours",
            f"How many closest faces should be displayed? "
            f"(1-{max_neighbors})",
            initialvalue=max_neighbors,
            minvalue=1,
            maxvalue=max_neighbors,
            parent=parent,
        )

        if n_neighbors is None:
            return None

        indices, distances = self._find_closest_faces(
            test_coefficients,
            n_neighbors,
        )

        self._visualize_results(
            test_image,
            indices,
            distances,
            labels,
        )

        matches = [
            {
                "Rank": position + 1,
                "Name": labels[index],
                "Image": str(self.loaded_paths_[index]),
                "Distance": round(float(distance), 4),
            }
            for position, (index, distance) in enumerate(
                zip(indices, distances)
            )
        ]

        return ToolResult(
            message=(
                f"Face recognition completed. "
                f"Displayed the {n_neighbors} closest faces."
            ),
            details={
                "Algorithm": "Eigenfaces (PCA) + Euclidean distance",
                "Training images": len(X_train),
                "Image shape": self.image_shape_,
                "PCA components": len(self.components_),
                "POV retained": (
                    f"{self.explained_variance_ratio_.sum():.2%}"
                ),
                "Test image": str(test_path),
                "Closest matches": matches,
            },
        )
