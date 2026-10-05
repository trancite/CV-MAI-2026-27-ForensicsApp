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
        mask_value = simpledialog.askinteger("Choose mask value (0-255)", parent=parent)
        if mask_value is None:
                return None
        if mask_value not in range (0, 256):
                print("Invalid value, please choose a value between 0 and 255")
                return self.__choose_mask(parent)
        return mask_value   
                 

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        mask_value = self.__choose_mask(parent)
        if mask_value is None:
            return None
        mask = document.current > mask_value
        masked_image = document.current * mask

        return ToolResult(
            result=masked_image,
            metadata={"mask": mask, "threshold": mask_value},
            success=True
)


            
        



       
