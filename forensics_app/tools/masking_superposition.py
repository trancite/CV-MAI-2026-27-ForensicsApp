from __future__ import annotations

import tkinter as tk
from PIL import Image
from tkinter import filedialog
import numpy as np

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utilities import dialog_options


class MaskingSuperposition(ForensicsTool):
    def __init__(self):
        self.tool_id = "masking superposition"
        self.title = "Mask superposition"
        self.category = "Set2"
        self.description = "Superimpose two images."
        self.requires_image = True
        self.options = ["Put over an image", "Change texture"]

    def _choose_mask_option(self, parent: tk.Misc) -> str | None:
        return dialog_options(parent, "Mask superposition",
                              "Choose how to use the mask", self.options)

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        option = self._choose_mask_option(parent)
        if option is None: return None

        reference_path = filedialog.askopenfilename(
            title="Select an image to superimpose",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
            parent=parent,
        )
        if not reference_path: return None

        assert document.current is not None

        with Image.open(reference_path) as reference_image:
            reference = reference_image.copy()

        if option == self.options[0]:
            source = document.current
            resized = reference.size != source.size
            reference = reference.convert("RGB").resize(source.size, Image.Resampling.BILINEAR)
            src_arr = np.asarray(source.convert("RGB"))
            ref_arr = np.asarray(reference)
            mask = np.all(ref_arr == 0, axis=-1, keepdims=True)
            new_image = Image.fromarray(np.where(mask, src_arr, ref_arr).astype(np.uint8))
        
        else:
            source = document.current
            resized = reference.size != source.size
            reference = reference.convert("RGB").resize(source.size, Image.Resampling.BILINEAR)
            src_arr = np.asarray(source.convert("RGB"))
            ref_arr = np.asarray(reference)
            mask = np.any(src_arr != 0, axis=-1, keepdims=True)
            new_image = Image.fromarray(np.where(mask, ref_arr, src_arr).astype(np.uint8))

        return ToolResult(
            image=new_image,
            message="Images superimposed.",
            details={"Output mode": new_image.mode, "Reference resized": resized},
        )
