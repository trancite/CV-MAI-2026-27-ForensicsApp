from __future__ import annotations

import tkinter as tk

from PIL import Image
from tkinter import filedialog

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
import skimage.exposure as exposure
import skimage.color as color
import numpy as np


class HistogramMatching(ForensicsTool):
    def __init__(self):
        self.tool_id = "histogrammatching"
        self.title = "Histogram Matching"
        self.category = "Set2"
        self.description = "Matches histogram between two images"
        self.requires_image = True

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

        
        src_arr = np.asarray(source.convert("RGB"))
        ref_arr = np.asarray(reference.convert("RGB"))

        src_lab = color.rgb2lab(src_arr)
        ref_lab = color.rgb2lab(ref_arr)

        src_L = src_lab[:, :, 0]
        ref_L = ref_lab[:, :, 0]

        matched_L = exposure.match_histograms(src_L, ref_L)

        matched_lab = src_lab.copy()
        matched_lab[:, :, 0] = matched_L

        matched_rgb = np.clip(color.lab2rgb(matched_lab), 0, 1)
        
        new_image = Image.fromarray((matched_rgb * 255).astype(np.uint8))

        return ToolResult(
            image=new_image,
            message="Matched the source histogram to the reference image.",
            details={"Operation": "Histogram matching", "Output mode": new_image.mode},
        )