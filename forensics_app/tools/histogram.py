from __future__ import annotations

import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class HistogramVisualization(ForensicsTool):
    def __init__(self):
        self.tool_id = "histogramvisualization"
        self.title = "Histogram Visualization"
        self.category = "Set2"
        self.description = "Visualize histogram of channels"
        self.requires_image = True

    def histogram_rgb(self, image_array: np.ndarray) -> None:
        red = image_array[:, :, 0].ravel()
        green = image_array[:, :, 1].ravel()
        blue = image_array[:, :, 2].ravel()
        plt.hist(red, bins=256, color="red", range=(0, 256), alpha=0.5, label="Red")
        plt.hist(green, bins=256, color="green", range=(0, 256), alpha=0.5, label="Green")
        plt.hist(blue, bins=256, color="blue", range=(0, 256), alpha=0.5, label="Blue")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.title("RGB histogram")
        plt.legend()

    def histogram_greyscale(self, image_array: np.ndarray) -> None:
        plt.hist(image_array.ravel(), bins=256, color="grey", range=(0, 256), alpha=0.8)
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.title("Greyscale histogram")

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        image = document.current
        if image is None:
            raise ValueError("No image is loaded.")

        image_for_histogram = image if image.mode == "L" else image.convert("RGB")
        image_array = np.asarray(image_for_histogram)
        plt.figure("Image histogram")
        if image_for_histogram.mode == "L":
            self.histogram_greyscale(image_array)
        else:
            self.histogram_rgb(image_array)
        plt.tight_layout()
        plt.show()
        plt.close()

        return ToolResult(
            message="Displayed the image histogram.",
            details={
                "Mode": image_for_histogram.mode,
                "Size": f"{image.width} x {image.height}",
            },
        )


