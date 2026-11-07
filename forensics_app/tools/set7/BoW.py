"""Build visual vocabularies and display visual words or test-image histograms."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ..utilities import dialog_options
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

from forensics_app.core import ImageDocument
from ..base import ToolResult
from ..set5.keypoint_tools import KeypointTool


class BagOfWords(KeypointTool):
    tool_id = "bagofwords"
    title = "Bag Of Words"
    category = "Set7"
    description = "Use Bag Of Words for Codebook construction, Histogram construction and Classification."
    requires_image = False
    options = ["Code Book construction", "Histogram construction", "Classification"]

    def _choose_option(self, parent: tk.Misc):
        return dialog_options(parent, "Option Selection", "Choose what do you want to do", self.options)

    def _choose_set(self, parent: tk.Misc, current_path: Path | None, set_name: str):
        return filedialog.askdirectory(
            title=set_name, parent=parent, mustexist=True,
            initialdir=str(current_path.parent if current_path else Path.cwd()),
        )

    def _return_classes_images(self, parent, set_path):
        return {
            folder.name: self._load_images(sorted(folder.iterdir()))
            for folder in sorted(Path(set_path).iterdir())
            if folder.is_dir() and not folder.name.startswith((".", "__MACOSX"))
        }

    def _classes_and_descriptors(self, classes_and_images: dict, algorithm, parameters):
        classes_and_descriptors = {}
        self.descriptor_sources_ = {}
        width, dtype = (256, bool) if algorithm == "ORB" else (128, np.uint8)
        for clas, images in classes_and_images.items():
            batches, sources = [], []
            for image_index, image in enumerate(images):
                gray = np.asarray(Image.fromarray(image.astype(np.uint8)).convert("L"))
                keypoints, descriptors = self._generate_keypoints_and_descriptors(gray, algorithm, parameters)
                batches.append(descriptors)
                sources.extend((image_index, y, x) for y, x in keypoints)
            classes_and_descriptors[clas] = np.vstack(batches) if batches else np.empty((0, width), dtype=dtype)
            self.descriptor_sources_[clas] = np.asarray(sources).reshape(-1, 3)
        return classes_and_descriptors


    def perform_k_means(self, feature_descriptors: np.ndarray, ncl: int = 10):
        descriptors = np.asarray(feature_descriptors, dtype=np.float32)
        count = min(ncl, len(np.unique(descriptors, axis=0)))
        return MiniBatchKMeans(n_clusters=count, random_state=0, n_init=3).fit(descriptors)

    def _load_images(self, images_paths):
        images = []
        for path in images_paths:
            try:
                with Image.open(path) as image:
                    images.append(np.asarray(image.convert("RGB")))
            except (OSError, ValueError):
                continue
        return images

    def _representative_patches(self, images, descriptors, sources, model, patch_size: int = 32):
        indices, _ = pairwise_distances_argmin_min(model.cluster_centers_, descriptors)
        patches = []
        for index in indices:
            image_index, row, column = sources[index]
            image = Image.fromarray(images[int(image_index)].astype(np.uint8))
            x, y = int(round(column)), int(round(row))
            left, top = x - patch_size // 2, y - patch_size // 2
            patches.append(image.crop((left, top, left + patch_size, top + patch_size)))
        return patches

    def _visualize_visual_words(self, classes_and_patches: dict[str, list[Image.Image]]) -> None:
        figures = []
        for clas, patches in classes_and_patches.items():
            columns = min(6, len(patches))
            rows = (len(patches) + columns - 1) // columns
            figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
            figures.append(figure)
            figure.suptitle(f"{clas}: {len(patches)} visual words")
            for axis in axes.flat:
                axis.axis("off")
            for index, (axis, image) in enumerate(zip(axes.flat, patches), start=1):
                axis.imshow(image)
                axis.set_title(f"Word {index}")
            figure.tight_layout()
        plt.show()
        for figure in figures:
            plt.close(figure)

    def _frequency_histogram(self, descriptors, centroids):
        if not len(descriptors):
            return np.zeros(len(centroids), dtype=int)
        words, _ = pairwise_distances_argmin_min(descriptors, centroids)
        return np.bincount(words, minlength=len(centroids))

    def _test_histogram(self, document):
        centroids = np.vstack([model.cluster_centers_ for model in self.codebooks_.values()])
        gray = np.asarray(document.current.convert("L"))
        _, descriptors = self._generate_keypoints_and_descriptors(gray, self.algorithm_, self.parameters_)
        self.histogram_ = self._frequency_histogram(descriptors, centroids)
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.bar(np.arange(1, len(centroids) + 1), self.histogram_)
        axis.set(title=document.path.name if document.path else "Test image", xlabel="Visual word", ylabel="Frequency")
        figure.tight_layout()
        plt.show()
        plt.close(figure)
        return ToolResult(
            message="Displayed the selected image's visual-word histogram.",
            details={"Histogram dimensions": len(centroids), "Descriptors": len(descriptors)},
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        current_path = document.path.resolve() if document.path else None
        option = self._choose_option(parent)
        if option is None: return None
        if option not in self.options[:2]:
            return ToolResult(message=f"{option} has not been implemented yet.")
        if option == self.options[1] and document.current is None:
            return ToolResult(message="Open a test image first, then choose Histogram construction.")
        algorithm = self._choose_algorithm(parent)
        if algorithm is None: return None
        parameters = self._choose_parameters(parent, algorithm)
        if parameters is None: return None

        set_path = self._choose_set(parent, current_path, "Choose the training folder with one subfolder per class.")
        if not set_path: return None
        n_words = simpledialog.askinteger(
            "Visual words", "Number of visual words per class:",
            initialvalue=10, minvalue=1, parent=parent,
        )
        if n_words is None: return None

        classes_and_images = self._return_classes_images(parent, set_path)
        classes_and_descriptors = self._classes_and_descriptors(classes_and_images, algorithm, parameters)
        self.codebooks_ = {}
        self.visual_word_patches_ = {}
        details = {"Algorithm": algorithm, "Words requested per class": n_words}
        skipped = []
        for clas, descriptors in classes_and_descriptors.items():
            if not len(descriptors):
                skipped.append(clas)
                continue
            model = self.perform_k_means(descriptors, n_words)
            self.codebooks_[clas] = model
            if option == self.options[0]:
                self.visual_word_patches_[clas] = self._representative_patches(
                    classes_and_images[clas], descriptors, self.descriptor_sources_[clas], model,
                )
            details[clas] = f"{len(classes_and_images[clas])} images, {len(descriptors)} descriptors, {model.n_clusters} words"
        if skipped:
            details["Classes without descriptors"] = ", ".join(skipped)
        if not self.codebooks_:
            return ToolResult(message="No visual words found. Choose class folders with textured images.", details=details)

        self.algorithm_, self.parameters_ = algorithm, parameters.copy()
        if option == self.options[1]:
            return self._test_histogram(document)
        self._visualize_visual_words(self.visual_word_patches_)
        return ToolResult(message=f"Displayed visual words for {len(self.codebooks_)} class(es).", details=details)
