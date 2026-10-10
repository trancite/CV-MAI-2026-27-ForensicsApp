from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import color, filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utilities import dialog_options


class FiltersTool(ForensicsTool):
    def __init__(self):
        self.tool_id = "filters"
        self.title = "Different filters"
        self.category = "Set3"
        self.description = "Different filters from skimage.filters to apply"
        self.requires_image = True
        self.filter_options = [
            "Smoothing/noise reduction",
            "Edge-preserving denoising",
            "Edge detection (Sobel)",
            "Edge detection (Prewitt)",
            "Stronger rotationally symmetric edge filter",
            "Sharpening",
            "Otsu threshold (mask)",
        ]
        self.mask_options = ["Binary", "Inverse"]

    def _choose_filter(self, parent: tk.Misc):
        return dialog_options(parent, "Filter selection", "Choose a filter to apply", self.filter_options)

    def _choose_mask(self, parent: tk.Misc):
        return dialog_options(parent, "Mask selection",
                              "Choose to see near objects (binary) or background (inverse)",
                              self.mask_options)

    def _choose_sigma(self, parent: tk.Misc):
        return simpledialog.askfloat("Sigma", "Choose sd", initialvalue=2.0,
                                     minvalue=0.1, maxvalue=100.0, parent=parent)

    def _choose_amount(self, parent: tk.Misc):
        return simpledialog.askfloat("Intensity of details augmentation",
                                     "Choose amount of details increasing", initialvalue=1.0,
                                     minvalue=0.1, maxvalue=20.0, parent=parent)

    @staticmethod
    def _to_float(img: Image.Image) -> np.ndarray:
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        return np.asarray(img, dtype=np.float64) / 255.0

    @staticmethod
    def _to_gray(arr: np.ndarray) -> np.ndarray:
        return arr if arr.ndim == 2 else color.rgb2gray(arr)

    @staticmethod
    def _to_pil(arr: np.ndarray, normalize: bool = False) -> Image.Image:
        arr = np.nan_to_num(arr)
        if normalize and arr.max() > 0:
            arr = arr / arr.max() 
        return Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8))

    def apply_filter(self, parent: tk.Misc, image: Image.Image, filter: str) -> Image.Image | None:
        arr = self._to_float(image)
        is_color = arr.ndim == 3
        channel_axis = arr.ndim - 1 if is_color else None   # 2 en RGB, None en gris

        if filter == self.filter_options[0]:
            sigma = self._choose_sigma(parent)
            if sigma is None:
                return None
            return self._to_pil(filters.gaussian(arr, sigma=sigma, channel_axis=channel_axis))

        elif filter == self.filter_options[1]:
            footprint = np.ones((3, 3, 1) if is_color else (3, 3))
            return self._to_pil(filters.median(arr, footprint=footprint))

        elif filter == self.filter_options[2]:
            return self._to_pil(filters.sobel(self._to_gray(arr)), normalize=True)

        elif filter == self.filter_options[3]:
            return self._to_pil(filters.prewitt(self._to_gray(arr)), normalize=True)

        elif filter == self.filter_options[4]:
            return self._to_pil(filters.scharr(self._to_gray(arr)), normalize=True)

        elif filter == self.filter_options[5]:
            amount = self._choose_amount(parent)
            if amount is None:
                return None
            return self._to_pil(filters.unsharp_mask(arr, amount=amount, channel_axis=channel_axis))

        elif filter == self.filter_options[6]:
            gray = self._to_gray(arr)
            threshold = filters.threshold_otsu(gray)
            mask_type = self._choose_mask(parent)
            if mask_type is None:
                return None
            mask = gray > threshold if mask_type == self.mask_options[0] else gray <= threshold
            if is_color:
                mask = mask[..., None]
            return self._to_pil(arr * mask)

        return None

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        choice = self._choose_filter(parent)
        if choice is None:
            return None
        new_image = self.apply_filter(parent, document.current, choice)
        if new_image is None:
            return None
        return ToolResult(image=new_image, message=f"Applied: {choice}", details={"Operation": f"Applied filter {choice}", 
                                                                               "Output mode": new_image.mode},
                )