from __future__ import annotations
import numpy as np
import skimage.exposure as exposure
import skimage.color as color
from PIL import Image, ImageOps
from tkinter import simpledialog
from skimage.util import img_as_float
from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utilities import dialog_options
import numpy as np
from scipy import ndimage
from skimage.feature import canny
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import unsharp_mask
import tkinter as tk



class Sharpening(ForensicsTool):
    def __init__(self):
            self.tool_id = "sharpening"
            self.title = "Sharpening"
            self.category = "Set3"
            self.description = "Apply sharpening to the image."
            self.requires_image = True

    def __choose_value(self, parent: tk.Misc) -> tuple | None:
        radius = simpledialog.askfloat(title="Radius", prompt="Radius: number from 0.5 to 10 (inclusive).", minvalue=0.5, maxvalue=10.0, parent=parent)
        if radius is None:
            return None

        amount = simpledialog.askfloat(title="Amount", prompt="Amount: number from 0.1 to 5 (inclusive).", minvalue=0.1, maxvalue=5.0, parent=parent)
        if amount is None:
            return None

        return radius, amount

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:

        assert document.current is not None
        img = document.current
        values = self.__choose_value(parent)
        if values is None:
            return None
        radius, amount = values

        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        arr = np.asarray(img)

        sharpened_image = unsharp_mask(
            arr, radius=radius, amount=amount,
            channel_axis=arr.ndim - 1 if arr.ndim == 3 else None,
        )

        output_image = Image.fromarray(
            (sharpened_image * 255).astype(np.uint8)
        )

        return ToolResult(
            image=output_image,
            message="Sharpening applied."
        )
