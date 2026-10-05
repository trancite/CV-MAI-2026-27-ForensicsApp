from __future__ import annotations

import tkinter as tk
from PIL import ImageOps, Image
from tkinter import simpledialog
import tkinter as tk
from tkinter import ttk
from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
import numpy as np

class Masking(ForensicsTool):
    def __init__(self):
        self.tool_id = "masking"
        self.title = "Masking"
        self.category = "Set2"
        self.description = "It applies a mask to the image"
        self.requires_image = True

    
    def __choose_mask(self, parent: tk.Misc) -> int | None:
        mask_value = simpledialog.askinteger(title = "Mask threshold", prompt="Choose a mask threshold", minvalue = 0, maxvalue=255, parent=parent)
        if mask_value is None:
                return None
        return mask_value   
                 

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        mask_value = self.__choose_mask(parent)
        if mask_value is None:
            return None
        image_tensor = np.array(document.current)
        mask = image_tensor > mask_value
        masked_image_tensor = image_tensor * mask
        masked_image = Image.fromarray(masked_image_tensor)
        return ToolResult(
            image=masked_image,
            message="Masked image.",
            details={"Threshold:":mask_value},
        )


            
        



       
