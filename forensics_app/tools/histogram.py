from __future__ import annotations

import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utilities import dialog_options


class HistogramVisualization(ForensicsTool):
    def __init__(self):
        self.tool_id = "histogramvisualization"
        self.title = "Histogram Visualization"
        self.category = "Set2"
        self.description = "Visualize histogram of channels"
        self.requires_image = True
        self.options = ["Default", "Luminostiy"]

    def __choose_histogram(self, parent: tk.Misc):
        return  dialog_options(parent, "Histogram selection", "Choose one type of histogram", self.options)

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

    def histogram_luminosity(self, image_array: np.ndarray) -> None:
        plt.hist(image_array.ravel(), bins=256, color="grey", range=(0, 100), alpha=0.8)
        plt.xlabel("L* value")
        plt.ylabel("Frequency")
        plt.title("L* histogram")

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        image = document.current
        if image is None:
            raise ValueError("No image is loaded.")

        option = self.__choose_histogram(parent)
        if option is None:
            return None

        plt.figure("Image histogram")
        if option == "Luminostiy":
            rgb_array = np.asarray(image.convert("RGB"))
            lab_array = color.rgb2lab(rgb_array)
            self.histogram_luminosity(lab_array[:, :, 0])
            details = "L* histogram"
        else:
            image_for_histogram = image if image.mode == "L" else image.convert("RGB")
            image_array = np.asarray(image_for_histogram)
            if image_for_histogram.mode == "L":
                self.histogram_greyscale(image_array)
            else:
                self.histogram_rgb(image_array)
            details = "Image histogram"
        plt.tight_layout()
        plt.show()
        plt.close()

        return ToolResult(
            message=f"Displayed the {details}.",
            details={
                "Histogram": details,
                "Mode": image.mode,
                "Size": f"{image.width} x {image.height}",
            },
        )


