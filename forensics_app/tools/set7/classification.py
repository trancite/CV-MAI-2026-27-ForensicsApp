"""Classify the current image with an SVM using a chi-squared kernel."""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from ..base import ToolResult
from .histogram_construction import HistogramConstruction


class Classification(HistogramConstruction):
    tool_id = "bowclassification"
    title = "Classification"
    description = "Classify the current image using a chi-squared SVM."

    def _show_result(self, document, classes):
        samples = [(self._frequency_histogram(d, self.centroids_), name)
                   for name, features in self.features_.items() for _, _, d in features]
        histograms, labels = zip(*samples)
        self.classifier_ = make_pipeline(Normalizer(norm="l1"),
            SVC(kernel=chi2_kernel, probability=True, random_state=0)).fit(histograms, labels)
        figure, axis, details = self._plot_histogram(document)
        figure.subplots_adjust(bottom=0.25)
        button = Button(figure.add_axes([0.4, 0.03, 0.2, 0.08]), "Classify")

        def predict(event):
            probabilities = self.classifier_.predict_proba(self.histogram_[None, :])[0]
            best = int(np.argmax(probabilities))
            label, probability = str(self.classifier_.classes_[best]), float(probabilities[best])
            details.update({"Prediction": label, "Probability": probability})
            axis.set_title(f"{label}: {probability:.1%}")
            figure.canvas.draw_idle()

        button.on_clicked(predict)
        plt.show()
        plt.close(figure)
        return ToolResult(message=axis.get_title(), details=details)
