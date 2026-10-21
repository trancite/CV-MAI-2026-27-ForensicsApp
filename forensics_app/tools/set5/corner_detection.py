from __future__ import annotations
import tkinter as tk
from tkinter import simpledialog
from matplotlib.cm import gray
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage import feature
from skimage.feature import hog
from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult


class CornerDetectionTool(ForensicsTool):
    def __init__(self):
        self.tool_id = "corner_detection"
        self.title = "Corner Detection"
        self.category = "Set5"
        self.description = "Detect corners in the image."
        self.requires_image = True
        self.options = ["Harris", "Fast"]

    def choose_method(self, parent: tk.Misc) -> str | None:
        method = simpledialog.askstring(
            title="Corner Detection Method",
            prompt="Enter the corner detection method (Harris or Fast):",
            parent=parent,
        )
        if method is None:
            return None
        method = method.lower()
        if method not in ["harris", "fast"]:
            tk.messagebox.showerror("Invalid Method", "Please enter 'Harris' or 'Fast'.")
            return None
        return method

    def choose_parameters(self, parent: tk.Misc) -> dict | None:
        method = self.choose_method(parent)
        if method is None:
            return None
        params = {}
        if method == "harris":
            threshold = simpledialog.askfloat(
                title="Harris Parameters",
                prompt="Enter the threshold for Harris corner detection (0-1):",
                minvalue=0.0,
                maxvalue=1.0,
                parent=parent,
            )
            if threshold is None:
                return None
            params["threshold"] = threshold
           
        elif method == "fast":
            threshold = simpledialog.askfloat(
                title="FAST Parameters",
                prompt="Enter the threshold for FAST corner detection (0-1):",
                minvalue=0.0,
                maxvalue=1.0,
                parent=parent,
            )
            if threshold is None:
                return None
            params["threshold"] = threshold
        params["method"] = method
        return params
    
    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        gray = np.asarray(document.current.convert("L"))
        params = self.choose_parameters(parent)

        if params is None:
            return None
        
        method = params["method"]

        if method == "harris":
            response = feature.corner_harris(gray)
            corners = feature.corner_peaks(response, threshold_rel=params["threshold"], min_distance=3
            )
        elif method == "fast":
            response = feature.corner_fast(
                gray,
                threshold=params["threshold"]
                )
            corners = feature.corner_peaks(response)

        output_image = document.current.convert("RGB")
        draw = ImageDraw.Draw(output_image)
        
        for y, x in corners:
            draw.ellipse([(x-2, y-2, x+2, y+2)], fill=(255, 0, 0))

        return ToolResult(
            image=output_image,
            message="Corners detected",
            details={"Method": method, "Number of corners": len(corners)},
        )