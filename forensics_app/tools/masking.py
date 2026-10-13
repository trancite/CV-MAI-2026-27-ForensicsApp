from __future__ import annotations

import tkinter as tk
from PIL import ImageOps, Image
from tkinter import simpledialog
import tkinter as tk
from tkinter import ttk
from forensics_app.core import ImageDocument
from forensics_app.tools.utilities import dialog_options
from .base import ForensicsTool, ToolResult
from .utilities import dialog_options
import numpy as np

class Masking(ForensicsTool):
    def __init__(self):
        self.tool_id = "masking"
        self.title = "Masking"
        self.category = "Set2"
        self.description = "Apply a mask to the image."
        self.requires_image = True
        self.options = ["Binary", "Inverse", "Range"]

    def __choose_type(self, parent: tk.Misc) -> str | None:
            return dialog_options(parent, "Masking",
                                           "Choose a Mask", self.options)

    def __choose_value(self, parent: tk.Misc) -> int | None:
        mask_value = simpledialog.askinteger(title = "Mask value (0-255)", prompt="Mask threshold: integer from 0 to 255 (inclusive).", minvalue = 0, maxvalue=255, parent=parent)
        if mask_value is None:
                return None
        return mask_value   
                 

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        mask_type = self.__choose_type(parent)
        if mask_type is None:
            return None
        
        image = document.current
        assert image is not None
        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")

        if mask_type == "Binary":
            image_tensor = np.array(image)
            mask_value = self.__choose_value(parent)
            if mask_value is None: return None
            gray = np.array(image.convert("L"))
            mask = gray > mask_value
            if image_tensor.ndim == 3:
                mask = mask[..., None]
                masked_image_tensor = image_tensor * mask
                masked_image = Image.fromarray(masked_image_tensor)
            else:
                masked_image_tensor = mask.astype(np.uint8) * 255
                masked_image = Image.fromarray(masked_image_tensor)


            return ToolResult(
            image=masked_image,
            message="Masked image.",
            details={"Value:":mask_value},
            )
        
        elif mask_type == "Inverse":
            image_tensor = np.array(image)
            mask_value = self.__choose_value(parent)
            if mask_value is None: return None
            gray = np.array(image.convert("L"))
            mask = gray <= mask_value
            if image_tensor.ndim == 3:
                mask = mask[..., None]
                masked_image_tensor = image_tensor * mask
                masked_image = Image.fromarray(masked_image_tensor)
            else:
                masked_image_tensor = mask.astype(np.uint8) * 255
                masked_image = Image.fromarray(masked_image_tensor)

            return ToolResult(
            image=masked_image,
            message="Masked image.",
            details={"Value:":mask_value},
            )

        elif mask_type == "Range":
            low = simpledialog.askinteger(title = "Lower value", prompt="Lower value: integer from 0 to 255 (inclusive).", minvalue = 0, maxvalue=255, parent=parent)
            if low is None: return None
            high = simpledialog.askinteger(title = "Upper value", prompt=f"Upper value: integer from {low} to 255 (inclusive).", minvalue = low, maxvalue=255, parent=parent)
            if low is None or high is None:
                return None
            image_tensor = np.array(image)
            gray = np.array(image.convert("L"))
            mask = ((gray >= low) & (gray <= high))
            if image_tensor.ndim == 3: mask = mask[..., None]
            masked_image_tensor = image_tensor * mask
            masked_image = Image.fromarray(masked_image_tensor)

            return ToolResult(
            image=masked_image,
            message="Masked image.",
            details={"Lower Value:":low, "Upper Value:":high},
            )
             

            
        



       
