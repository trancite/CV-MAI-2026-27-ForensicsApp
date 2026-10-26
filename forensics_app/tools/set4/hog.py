from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage import exposure

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult


class HOGTool(ForensicsTool):
    def __init__(self):
        self.tool_id = "hog"
        self.title = "Histogram of Oriented Gradients (HOG)"
        self.category = "Set4"
        self.description = "Compute the Histogram of Oriented Gradients (HOG) of an image."
        self.requires_image = True

    def choose_parameters(self, parent: tk.Misc) -> tuple[tuple[int, int], tuple[int, int], int] | None:
        pixels_per_cell = simpledialog.askinteger(title="Pixels per cell", prompt="Enter the number of pixels per cell (at least 2 for visualization):", minvalue=2, parent=parent)
        if pixels_per_cell is None:
            return None
        cells_per_block = simpledialog.askinteger(title="Cells per block", prompt="Enter the number of cells per block:", minvalue=1, parent=parent)
        if cells_per_block is None:
            return None
        orientations = simpledialog.askinteger(title="Orientations", prompt="Enter the number orientations.:", minvalue=1, parent=parent)
        if orientations is None:
            return None
        pixels_per_cell = (pixels_per_cell, pixels_per_cell)
        cells_per_block = (cells_per_block, cells_per_block)
        return pixels_per_cell, cells_per_block, orientations

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        params = self.choose_parameters(parent)
        if params is None:
            return None

        pixels_per_cell, cells_per_block, orientations = params
        if min(pixels_per_cell) < 2:
            raise ValueError("HOG visualization requires at least 2 pixels per cell.")
        if min(cells_per_block) < 1:
            raise ValueError("Cells per block must be at least 1.")
        required_height = pixels_per_cell[0] * cells_per_block[0]
        required_width = pixels_per_cell[1] * cells_per_block[1]
        width, height = document.current.size
        if height < required_height or width < required_width:
            raise ValueError(
                f"These parameters require at least a {required_width} x {required_height} image; "
                f"the current image is {width} x {height}."
            )

        img = document.current.convert("L")
        arr = np.asarray(img)
        _, hog_image = hog(arr, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)

        hog_image_rescaled = exposure.rescale_intensity(hog_image)

        new_image = Image.fromarray((hog_image_rescaled * 255).astype(np.uint8))

        return ToolResult(
            image=new_image,
            message="HOG visualization computed.",
            details={"Operation": "HOG computation", "Output mode": new_image.mode},
        )
