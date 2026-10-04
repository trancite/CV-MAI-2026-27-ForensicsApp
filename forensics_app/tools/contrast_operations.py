from __future__ import annotations

import tkinter as tk

import numpy as np
import skimage.exposure as exposure
from PIL import Image, ImageOps
from tkinter import simpledialog
from skimage.util import img_as_float   # import nuevo
from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utilities import dialog_options


class ContrastOperations(ForensicsTool):
    def __init__(self):
        self.tool_id = "contrastoperations"
        self.title = "Contrast Operations"
        self.category = "Set2"
        self.description = "Various operations regarding contrast"
        self.requires_image = True
        self.options = [
            "Low contrast",
            "Contrast Stretching",
            "Histogram Equalization",
            "Adaptive histogram equalization",
        ]

    def _choose_operation(self, parent: tk.Misc) -> str | None:
        return dialog_options(parent, "Contrast operations",
                              "Choose a contrast operation", self.options)

    def _ask_cutoff(self, parent: tk.Misc, text: str) -> float | None:
        return simpledialog.askfloat("Cutoff", text, minvalue=0.0,
                                     maxvalue=100.0, parent=parent)

    @staticmethod
    def _percentiles(img: np.ndarray, cutoff: float) -> tuple[float, float]:
        lo, hi = np.percentile(img, [cutoff / 2, 100 - cutoff / 2])
        return float(lo), float(hi)

    @staticmethod
    def _to_pil(arr: np.ndarray) -> Image.Image:
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def contrast_decreased(self, parent: tk.Misc, img: np.ndarray):
        cutoff = self._ask_cutoff(parent, "Percentage of pixels at the boundaries (defines the output range).")
        if cutoff is None:
            return None
        c, d = cutoff / 200, 1 - cutoff / 200
        return exposure.rescale_intensity(img, in_range="image", out_range=(c, d))

    def contrast_stretching(self, parent: tk.Misc, img: np.ndarray):
        cutoff = self._ask_cutoff(parent, "Percentage of pixels at the boundaries (defines the input range).")
        if cutoff is None:
            return None
        a, b = self._percentiles(img, cutoff)
        return exposure.rescale_intensity(img, in_range=(a, b), out_range="dtype")

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        image = document.current
        assert image is not None
        img = img_as_float(np.asarray(ImageOps.grayscale(image)))

        operation = self._choose_operation(parent)
        if operation == "Low contrast":
            result = self.contrast_decreased(parent, img)
        elif operation == "Contrast Stretching":
            result = self.contrast_stretching(parent, img)
        elif operation == "Histogram Equalization":
            result = exposure.equalize_hist(img)
        elif operation == "Adaptive histogram equalization":
            result = exposure.equalize_adapthist(img, clip_limit = 0.03)
        else:
            return None  

        if result is None: 
            return None

        new_image = self._to_pil(result)
        return ToolResult(
            image=new_image,
            message="Result.",
            details={"Operation": operation, "Output mode": new_image.mode},
        )