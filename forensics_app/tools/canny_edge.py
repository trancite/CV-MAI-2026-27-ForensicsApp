from __future__ import annotations

import tkinter as tk

import numpy as np
import skimage.exposure as exposure
import skimage.color as color  # Añadimos la importación de color para el espacio LAB
from PIL import Image, ImageOps
from tkinter import simpledialog
from skimage.util import img_as_float
from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utilities import dialog_options
import numpy as np
from scipy import ndimage
from skimage.feature import canny



class CannyEdge(ForensicsTool):
    def __init__(self):
            self.tool_id = "cannyedge"
            self.title = "Canny Edge Detection"
            self.category = "Set3"
            self.description = "Detect edges with the Canny edge detection algorithm."
            self.requires_image = True
            self.options = ["Regular", "Sigma 2", "With Thresholds"]


    def __choose_type(self, parent: tk.Misc) -> str | None:
                return dialog_options(parent, "Canny Edge Detection",
                                               "Choose a Canny Edge Detection Type", self.options)
    
    def __choose_value(self, parent: tk.Misc) -> tuple[float, float] | None:
        low_thr = simpledialog.askfloat(title = "Low threshold", prompt="Choose a low threshold (0-1)", minvalue = 0.0, maxvalue=1.0, parent=parent)
        
        high_thr  = simpledialog.askfloat(title = "High threshold", prompt="Choose a high threshold (0-1)", minvalue = low_thr, maxvalue=1.0, parent=parent)
        
        return low_thr, high_thr
    

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None 
        img = document.current 
        gray = color.rgb2gray(img)

        edge_type = self.__choose_type(parent)

        if edge_type is None:
            return None
        if edge_type == "Regular":
            edges = canny(gray, sigma=1.0)

        elif edge_type == "Sigma 2":
            edges = canny(gray, sigma=2.0)

        elif edge_type == "With Thresholds":
            thresholds = self.__choose_value(parent)
            if thresholds is None:
                return None
            low_thr, high_thr = thresholds
            edges = canny(gray, low_threshold=low_thr, high_threshold=high_thr,sigma=2.0) 
        output_image = Image.fromarray((edges * 255).astype(np.uint8))
        return ToolResult(
            image=output_image,
            message="Canny edge detection applied.",
        )

    