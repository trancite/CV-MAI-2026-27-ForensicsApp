from __future__ import annotations

import tkinter as tk

from PIL import ImageOps, Image
from tkinter import simpledialog


from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
import numpy as np
class ContrastStreching(ForensicsTool):
    def __init__(self):
        self.tool_id = "contraststreching"
        self.title = "Contrast Streching"
        self.category = "Set2"
        self.description = "Strechs the contrast"
        self.requires_image = True

    def __choose_cutoff(self, parent: tk.Misc) -> float | None:
        cutoff = simpledialog.askfloat("Cutoff", "Percentage of pixels at the intensity boundaries to ignore.", minvalue=0.0, maxvalue=100, parent=parent)
        if cutoff is None:
            return None
        return cutoff
    
    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        image = document.current
        assert image is not None
        cutoff = self.__choose_cutoff(parent)
        if cutoff is None:
            return None
        rgb_image = document.current.convert("RGB")
        contrasted_image = ImageOps.autocontrast(rgb_image, cutoff = cutoff)
        return ToolResult(
                        image=contrasted_image,
                        message="New image with contrast streched.",
                        details={"Operation": "Contrast streched", "Output mode": contrasted_image.mode})
