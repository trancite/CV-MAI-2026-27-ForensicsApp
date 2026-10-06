from __future__ import annotations

import tkinter as tk
from PIL import ImageOps, Image
from tkinter import simpledialog, filedialog
import numpy as np

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class MaskingSuperposition(ForensicsTool):
    def __init__(self):
        self.tool_id = "masking superposition"
        self.title = "Masking Superposition"
        self.category = "Set2"
        self.description = "Superpose two images"
        self.requires_image = True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        reference_path = filedialog.askopenfilename(
            title="Select image to superpose",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
            parent=parent,
        )
        if not reference_path:
            return None

        assert document.current is not None

        # Load reference image
        with Image.open(reference_path) as reference_image:
            reference = reference_image.copy()

        # Base image
        src = document.current.convert("RGB")

        reference = reference.resize(src.size, Image.BILINEAR)

        src_arr = np.asarray(src)
        ref_arr = np.asarray(reference.convert("RGB"))

        mask = (src_arr == 0)

        new_arr = np.where(mask, ref_arr, src_arr)

        new_arr = new_arr.astype(np.uint8)

        new_arr = np.clip(new_arr, 0, 255).astype(np.uint8)

        new_image = Image.fromarray(new_arr)

        return ToolResult(
            image=new_image,
            message="Superposed image.",
            details={"Output mode": new_image.mode, "Reference resized": True},
        )





