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
        sources = [(image, y, x)
                for name in classes
                for image, points, _ in self.features_[name]
                for y, x in points]
        descriptors = np.vstack(list(classes.values()))
        indices, _ = pairwise_distances_argmin_min(self.centroids_, descriptors)
        patches = []
        for index in indices:
            image, y, x = sources[index]
            left, top = round(x) - patch_size // 2, round(y) - patch_size // 2
            patches.append(Image.fromarray(np.uint8(image)).crop((left, top, left + patch_size, top + patch_size)))
        return patches

    def _class_histograms(self):
        return {
            name: np.mean([self._frequency_histogram(d, self.centroids_) / max(len(d), 1)
                        for _, _, d in feats], axis=0)
            for name, feats in self.features_.items() if feats
        }

    def _visualize_visual_words(self, patches, histograms):
        columns = min(10, len(patches))
        rows = (len(patches) + columns - 1) // columns
        fig_words, axes = plt.subplots(rows, columns, figsize=(1.6 * columns, 1.8 * rows), squeeze=False)
        fig_words.suptitle(f"{len(patches)} visual words")
        for index, axis in enumerate(axes.flat):
            axis.axis("off")
            if index < len(patches):
                axis.imshow(patches[index])
                axis.set_title(f"{index + 1}", fontsize=8)
        fig_words.tight_layout()

        fig_hist, axes = plt.subplots(len(histograms), 1, figsize=(9, 2 * len(histograms)),
                                    sharex=True, squeeze=False)
        for axis, (name, hist) in zip(axes.flat, histograms.items()):
            axis.bar(np.arange(1, len(hist) + 1), hist)
            axis.set_ylabel(name)
        axes.flat[-1].set_xlabel("Visual word")
        fig_hist.tight_layout()
        plt.show()
        plt.close(fig_words)
        plt.close(fig_hist)

    def _show_result(self, document, classes):
        self.visual_word_patches_ = self._representative_patches(classes)
        self.class_histograms_ = self._class_histograms()
        self._visualize_visual_words(self.visual_word_patches_, self.class_histograms_)
        return ToolResult(message="Visual words displayed.",
                        details={"Classes": len(classes), "Visual words": len(self.centroids_)})