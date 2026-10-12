from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
import skimage.color as color
from PIL import Image
from skimage.feature import canny

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utilities import dialog_options


class CannyEdge(ForensicsTool):
    def __init__(self):
        self.tool_id = "cannyedge"
        self.title = "Canny Edge Detection"
        self.category = "Set3"
        self.description = "Detect edges with the Canny edge detection algorithm."
        self.requires_image = True
        self.canny_options = ["Default thresholds", "Configurable thresholds"]
        self.visualization_options = ["Canny edges", "Canny edges with superposition"]

    def __apply_segments(self, image_array: np.ndarray, edges: np.ndarray):
        arr = np.asarray(image_array)
        overlay = arr.copy()

        if overlay.ndim == 2:
            overlay = np.stack([overlay, overlay, overlay], axis=-1)

        overlay[edges] = [255, 0, 0]
        return overlay

    def __choose_visualization_type(self, parent: tk.Misc) -> str | None:
        return dialog_options(parent, "Edge visualization",
                              "Choose how to visualize Canny edge detection", self.visualization_options)

    def __choose_type(self, parent: tk.Misc) -> str | None:
        return dialog_options(parent, "Canny Edge Detection",
                              "Choose a Canny edge detection type", self.canny_options)

    def __choose_value(self, parent: tk.Misc) -> tuple[float, float] | None:
        low_thr = simpledialog.askfloat(title="Low threshold", prompt="Low threshold: number from 0 to 1 (inclusive).", minvalue=0.0, maxvalue=1.0, parent=parent)
        if low_thr is None: return None
        high_thr = simpledialog.askfloat(title="High threshold", prompt=f"High threshold: number from {low_thr:g} to 1 (inclusive).", minvalue=low_thr, maxvalue=1.0, parent=parent)

        if high_thr is None: return None
        return low_thr, high_thr

    def __choose_sigma(self, parent: tk.Misc) -> float | None:
        return simpledialog.askfloat(title="Sigma", prompt="Sigma: number from 0.1 to 1,000,000,000 (inclusive).", minvalue=0.1, maxvalue=1e9, parent=parent)

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        img = document.current.convert("RGB")
        arr = np.asarray(img)

        edge_type = self.__choose_type(parent)
        if edge_type is None:
            return None

        if arr.ndim == 3:
            gray = color.rgb2gray(arr)
        else:
            gray = arr

        if edge_type == self.canny_options[0]:
            sigma = self.__choose_sigma(parent)
            if sigma is None: return None
            edges = canny(gray, sigma=sigma)

        elif edge_type == self.canny_options[1]:
            thresholds = self.__choose_value(parent)
            if thresholds is None: return None

            sigma = self.__choose_sigma(parent)
            if sigma is None: return None

            low_thr, high_thr = thresholds
            edges = canny(gray, low_threshold=low_thr, high_threshold=high_thr, sigma=sigma)
        else:
            return None

        visualization_mode = self.__choose_visualization_type(parent)
        if visualization_mode is None: return None

        if visualization_mode == self.visualization_options[0]:
            output_image = Image.fromarray((edges * 255).astype(np.uint8))
        else:
            new_image_array = self.__apply_segments(image_array=arr, edges=edges)
            output_image = Image.fromarray(new_image_array.astype(np.uint8))

        return ToolResult(
            image=output_image,
            message="Canny edge detection applied.",
        )
