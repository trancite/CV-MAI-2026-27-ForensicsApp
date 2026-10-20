from __future__ import annotations
import tkinter as tk
from tkinter import simpledialog
from PIL import Image
from tkinter import filedialog
from matplotlib import image
import numpy as np
from skimage.feature import hog
from skimage import exposure
from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utilities import dialog_options

class HOGTool(ForensicsTool):
    def __init__(self):
        self.tool_id = "hog"
        self.title = "Histogram of Oriented Gradients (HOG)"
        self.category = "Set3"
        self.description = "Compute the Histogram of Oriented Gradients (HOG) of an image."
        self.requires_image = True

    def choose_parameters(self, parent: tk.Misc) -> tuple[tuple[int, int], tuple[int, int]] | None:
        pixels_per_cell = simpledialog.askinteger(title="Pixels per cell", prompt="Enter the number of pixels per cell:", minvalue=1, parent=parent)
        if pixels_per_cell is None:
            return None
        cells_per_block = simpledialog.askinteger(title="Cells per block", prompt="Enter the number of cells per block:", minvalue=1, parent=parent)
        if cells_per_block is None:
            return None
        pixels_per_cell = (pixels_per_cell, pixels_per_cell)
        cells_per_block = (cells_per_block, cells_per_block)
        return pixels_per_cell, cells_per_block

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        params = self.choose_parameters(parent)
        if params is None:
            return None
        img = document.current.convert("L")
        arr = np.asarray(img)
        hog_features, hog_image = hog(arr, pixels_per_cell=params[0], cells_per_block=params[1], visualize=True)
        
        hog_image_rescaled = exposure.rescale_intensity(hog_image)

        new_image = Image.fromarray((hog_image_rescaled * 255).astype(np.uint8))
        
        return ToolResult(
            image=new_image,
            message="HOG features computed and visualized.",
            details={"Operation": "HOG computation", "Output mode": new_image.mode},
        )
