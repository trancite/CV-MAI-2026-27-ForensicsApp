"""Build a visual vocabulary and display the visual words for each class."""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from ..base import ToolResult
from .bow_base import BoWTool


class BagOfWords(BoWTool):
    tool_id = "bagofwords"
    title = "Codebook Construction"
    description = "Build and display visual words for each training class."
    requires_image = False

    def _representative_patches(self, classes, patch_size=32):
        patches = {}
        for name, descriptors in classes.items():
            if not len(descriptors): continue
            sources = [(image, y, x) for image, points, _ in self.features_[name] for y, x in points]
            indices, _ = pairwise_distances_argmin_min(self.codebooks_[name].cluster_centers_, descriptors)
            patches[name] = []
            for index in indices:
                image, y, x = sources[index]
                left, top = round(x) - patch_size // 2, round(y) - patch_size // 2
                patches[name].append(Image.fromarray(np.uint8(image)).crop((left, top, left + patch_size, top + patch_size)))
        return patches

    def _visualize_visual_words(self, classes):
        figures = []
        for name, patches in classes.items():
            columns = min(6, len(patches))
            rows = (len(patches) + columns - 1) // columns
            figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
            figures.append(figure)
            figure.suptitle(f"{name}: {len(patches)} visual words")
            for index, axis in enumerate(axes.flat):
                axis.axis("off")
                if index < len(patches):
                    axis.imshow(patches[index])
                    axis.set_title(f"Word {index + 1}")
            figure.tight_layout()
        plt.show()
        for figure in figures:
            plt.close(figure)

    def _show_result(self, document, classes):
        self.visual_word_patches_ = self._representative_patches(classes)
        self._visualize_visual_words(self.visual_word_patches_)
        return ToolResult(message="Visual words displayed.",
                          details={"Classes": len(self.codebooks_), "Visual words per class": self.words_per_class_,
                                   "Total visual words": len(self.centroids_)})
