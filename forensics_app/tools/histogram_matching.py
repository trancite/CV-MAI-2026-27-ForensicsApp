from __future__ import annotations

import tkinter as tk

from PIL import Image
from tkinter import filedialog

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
import numpy as np


class HistogramMatching(ForensicsTool):
    def __init__(self):
        self.tool_id = "histogrammatching"
        self.title = "Histogram Matching"
        self.category = "Set2"
        self.description = "Matchs histogram between two images"
        self.requires_image = True

    def match_channel(self, src: np.ndarray, ref: np.ndarray) -> np.ndarray:
        s_vals, s_inv, s_counts = np.unique(src.ravel(), return_inverse=True, return_counts=True)
        r_vals, r_counts = np.unique(ref.ravel(), return_counts=True)
        s_cdf = np.cumsum(s_counts) / src.size
        r_cdf = np.cumsum(r_counts) / ref.size
        mapped = np.interp(s_cdf, r_cdf, r_vals)   
        return mapped[s_inv].reshape(src.shape)

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        reference_path = filedialog.askopenfilename(
            title="Select reference image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
            parent=parent,
        )
        if not reference_path:
            return None
        assert document.current is not None

        with Image.open(reference_path) as reference_image:
            reference = reference_image.copy()

        source = document.current
        y, cb, cr = source.convert("YCbCr").split()
        y_ref, _, _ = reference.convert("YCbCr").split()

        y_new = np.rint(self.match_channel(np.array(y), np.array(y_ref))).astype(np.uint8)
        new_image = Image.merge("YCbCr", (Image.fromarray(y_new), cb, cr)).convert("RGB")
        return ToolResult(
            image=new_image,
            message="Matched the source histogram to the reference image.",
            details={"Operation": "Histogram matching", "Output mode": new_image.mode},
        )
    