"""Display the visual-word frequency histogram of the current test image."""

import matplotlib.pyplot as plt
import numpy as np
from ..base import ToolResult
from .bow_base import BoWTool


class HistogramConstruction(BoWTool):
    tool_id = "histogramconstruction"
    title = "Histogram Construction"
    description = "Display the visual-word histogram of the current image."
    requires_image = True

    def _plot_histogram(self, document):
        gray = np.asarray(document.current.convert("L"))
        _, descriptors = self._generate_keypoints_and_descriptors(gray, self.algorithm_, self.parameters_)
        self.histogram_ = self._frequency_histogram(descriptors, self.centroids_)
        details = {"Histogram dimensions": len(self.centroids_), "Descriptors": len(descriptors)}
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.bar(np.arange(1, len(self.centroids_) + 1), self.histogram_)
        axis.set(title=document.path.name if document.path else "Test image", xlabel="Visual word", ylabel="Frequency")
        figure.tight_layout()
        return figure, axis, details

    def _show_result(self, document, classes):
        figure, axis, details = self._plot_histogram(document)
        plt.show()
        plt.close(figure)
        return ToolResult(message=axis.get_title(), details=details)
