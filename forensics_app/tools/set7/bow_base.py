"""Shared loading, descriptor extraction and vocabulary training for Set7."""

from abc import abstractmethod
from pathlib import Path
from tkinter import filedialog, simpledialog
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from ..base import ToolResult
from ..set5.keypoint_tools import KeypointTool


class BoWTool(KeypointTool):
    category = "Set7"

    def _load_images(self, paths):
        images = []
        for path in paths:
            try:
                with Image.open(path) as image:
                    images.append(np.asarray(image.convert("RGB")))
            except (OSError, ValueError):
                continue
        return images

    def _return_classes_images(self, parent, folder, exclude=None):
        return {
            path.name: self._load_images(p for p in sorted(path.iterdir()) if p.resolve() != exclude)
            for path in sorted(Path(folder).iterdir())
            if path.is_dir() and not path.name.startswith((".", "__MACOSX"))
        }

    def _classes_and_descriptors(self, classes, algorithm, parameters):
        self.features_ = {}
        for name, images in classes.items():
            self.features_[name] = []
            for image in images:
                gray = np.asarray(Image.fromarray(np.uint8(image)).convert("L"))
                points, descriptors = self._generate_keypoints_and_descriptors(gray, algorithm, parameters)
                self.features_[name].append((image, points, descriptors))
        return {name: np.vstack([d for _, _, d in features])
                for name, features in self.features_.items() if features}

    def perform_k_means(self, descriptors, ncl=10):
        descriptors = np.asarray(descriptors, dtype=np.float32)
        count = min(ncl, len(np.unique(descriptors, axis=0)))
        return MiniBatchKMeans(n_clusters=count, random_state=0, n_init=3).fit(descriptors)

    def _frequency_histogram(self, descriptors, centroids):
        words = pairwise_distances_argmin_min(descriptors, centroids)[0] if len(descriptors) else np.array([], dtype=int)
        return np.bincount(words, minlength=len(centroids))

    def run(self, parent, document):
        if self.requires_image and document.current is None:
            return ToolResult(message="Open a test image first.")
            
        algorithm = self._choose_algorithm(parent)
        if algorithm is None: return None
        
        parameters = self._choose_parameters(parent, algorithm)
        if parameters is None: return None
        
        folder = filedialog.askdirectory(title="Select training classes", parent=parent, mustexist=True)
        if not folder: return None
        
        n_words = simpledialog.askinteger(
            "Visual words", "Total number of visual words (K):", initialvalue=100, minvalue=1, parent=parent,
        )
        if n_words is None: return None
        
        exclude = document.path.resolve() if self.requires_image and document.path else None
        images = self._return_classes_images(parent, folder, exclude)
        classes = self._classes_and_descriptors(images, algorithm, parameters)
        
        if not classes: 
            return ToolResult(message="No descriptors found in the training images.")
        
        all_descriptors = np.vstack(list(classes.values()))
        model = self.perform_k_means(all_descriptors, n_words)
        
        if model.n_clusters < n_words:
            return ToolResult(message=f"Only {model.n_clusters} distinct descriptors available. Choose a smaller number.")
            
        self.codebook_ = model
        self.n_words_ = n_words
        self.centroids_ = model.cluster_centers_
        self.algorithm_, self.parameters_ = algorithm, parameters
        
        return self._show_result(document, classes)

    @abstractmethod
    def _show_result(self, document, classes):
        """Display the result specific to the selected tool."""
