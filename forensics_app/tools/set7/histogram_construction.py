"""Construct and display the visual-word histogram of a test image."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, simpledialog

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

from forensics_app.core import ImageDocument
from ..base import ToolResult
from ..set5.keypoint_tools import KeypointTool
from .BoW import BagOfWords


class HistogramConstruction(BagOfWords):
    tool_id = "histogramconstruction"
    title = "Histogram Construction"
    category = "Set7"
    description = "Construct and display the visual-word histogram of a test image."
    requires_image = True

    def _frequency_histogram(self, descriptors, centroids):
        """Assign each descriptor to its nearest visual word."""
        if descriptors is None or not len(descriptors):
            return np.zeros(len(centroids), dtype=int)

        words, _ = pairwise_distances_argmin_min(
            descriptors.astype(np.float32),
            centroids.astype(np.float32),
        )

        return np.bincount(words, minlength=len(centroids))

    def _construct_codebooks(
        self,
        parent: tk.Misc,
        algorithm: str,
        parameters: dict,
        set_path: str,
        n_words: int,
    ):
        """Build one visual codebook for each training class."""
        classes_and_images = self._return_classes_images(parent, set_path)

        classes_and_descriptors = self._classes_and_descriptors(
            classes_and_images,
            algorithm,
            parameters,
        )

        codebooks = {}
        details = {}
        skipped = []

        for clas, descriptors in classes_and_descriptors.items():
            if not len(descriptors):
                skipped.append(clas)
                continue

            model = self.perform_k_means(descriptors, n_words)
            codebooks[clas] = model

            details[clas] = (
                f"{len(classes_and_images[clas])} images, "
                f"{len(descriptors)} descriptors, "
                f"{model.n_clusters} words"
            )

        if skipped:
            details["Classes without descriptors"] = ", ".join(skipped)

        return codebooks, details

    def _test_histogram(
        self,
        document: ImageDocument,
        codebooks: dict,
        algorithm: str,
        parameters: dict,
    ) -> ToolResult:
        """Calculate and display the test image histogram."""

        # Combine the visual words from all class codebooks.
        centroids = np.vstack([
            model.cluster_centers_
            for model in codebooks.values()
        ])

        # Extract descriptors from the test image.
        gray = np.asarray(document.current.convert("L"))

        _, descriptors = self._generate_keypoints_and_descriptors(
            gray,
            algorithm,
            parameters,
        )

        # Assign each descriptor to its nearest visual word.
        histogram = self._frequency_histogram(
            descriptors,
            centroids,
        )

        self.histogram_ = histogram
        self.centroids_ = centroids
        self.codebooks_ = codebooks
        self.algorithm_ = algorithm
        self.parameters_ = parameters.copy()

        # Display histogram.
        figure, axis = plt.subplots(figsize=(10, 4))

        axis.bar(
            np.arange(1, len(histogram) + 1),
            histogram,
        )

        axis.set_title(
            document.path.name
            if document.path
            else "Test image"
        )
        axis.set_xlabel("Visual word")
        axis.set_ylabel("Frequency")
        axis.set_xticks(np.arange(1, len(histogram) + 1))

        figure.tight_layout()
        plt.show()
        plt.close(figure)

        return ToolResult(
            message="Displayed the test image visual-word histogram.",
            details={
                "Descriptor algorithm": algorithm,
                "Number of descriptors": len(descriptors),
                "Number of visual words": len(centroids),
                "Histogram dimensions": len(histogram),
                "Histogram": histogram.tolist(),
            },
        )

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        # A test image must be open.
        if document.current is None:
            return ToolResult(
                message="Open a test image first, then choose Histogram construction."
            )

        current_path = (
            document.path.resolve()
            if document.path
            else None
        )

        # Select descriptor algorithm.
        algorithm = self._choose_algorithm(parent)
        if algorithm is None:
            return None

        parameters = self._choose_parameters(parent, algorithm)
        if parameters is None:
            return None

        # Select training dataset.
        set_path = self._choose_set(
            parent,
            current_path,
            "Choose the training folder with one subfolder per class.",
        )

        if not set_path:
            return None

        # Select number of visual words per class.
        n_words = simpledialog.askinteger(
            "Visual words",
            "Number of visual words per class:",
            initialvalue=10,
            minvalue=1,
            parent=parent,
        )

        if n_words is None:
            return None

        # Construct one codebook per class.
        codebooks, details = self._construct_codebooks(
            parent,
            algorithm,
            parameters,
            set_path,
            n_words,
        )

        if not codebooks:
            return ToolResult(
                message="No visual words found. Choose training images with detectable keypoints.",
                details=details,
            )

        # Construct and display the test histogram.
        result = self._test_histogram(
            document,
            codebooks,
            algorithm,
            parameters,
        )

        details.update(result.details)

        return ToolResult(
            message=result.message,
            details=details,
        )