from __future__ import annotations
import numpy as np
import skimage.exposure as exposure
import skimage.color as color 
from PIL import Image, ImageOps
from tkinter import simpledialog
from skimage.util import img_as_float
from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utilities import dialog_options
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
        radius = simpledialog.askfloat(title="Radius", prompt="Choose a radius (0.5-10).", minvalue=0.5, maxvalue=10.0, parent=parent)
        
        amount = simpledialog.askfloat(title="Amount", prompt="Choose an amount (0.1-5).", minvalue=0.1, maxvalue=5.0, parent=parent)
        
        return radius, amount

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:

        assert document.current is not None
        img = document.current
        radius, amount = self.__choose_value(parent)

        if radius is None or amount is None:
            return None

        arr = np.asarray(img)

        sharpened_image = unsharp_mask(arr, radius=radius, amount=amount)

        output_image = Image.fromarray(
            (sharpened_image * 255).astype(np.uint8)
        )

        return ToolResult(
            image=output_image,
            message="Sharpening applied."
        )