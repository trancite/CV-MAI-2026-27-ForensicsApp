from __future__ import annotations
import tkinter as tk
from tkinter import simpledialog
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.cm as cm  # <-- AÑADIMOS ESTO PARA EL MAPA DE CALOR MORADO
from skimage import feature
from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utilities import dialog_options
from skimage.color import gray2rgb

class CornerDetectionTool(ForensicsTool):
    def __init__(self):
        self.tool_id = "corner_detection"
        self.title = "Corner Detection"
        self.category = "Set5"
        self.description = "Detect corners in the image."
        self.requires_image = True
        self.options = ["Harris", "Fast"]
        self.display_options = ["Peaks Overlay (Red Circles)", "Response Map (Heatmap)"] 

    def choose_method(self, parent: tk.Misc) -> str | None:
        return dialog_options(
            parent, 
            "Corner Detection Method",
            "Enter the corner detection method:",
            self.options,
        )

    def choose_parameters(self, parent: tk.Misc) -> dict | None:
        method = self.choose_method(parent)
        if method is None: return None
        
        params = {}
        
        if method == self.options[0]:  # Harris
            threshold = simpledialog.askfloat("Harris Parameters", "Enter threshold (0-1):", minvalue=0.0, maxvalue=1.0, parent=parent)
            if threshold is None: return None
            
            min_distance = simpledialog.askfloat("Harris Parameters", "Enter min_distance:", minvalue=0.0, maxvalue=100.0, parent=parent)
            if min_distance is None: return None
            
            params["threshold"] = threshold
            params["min_distance"] = min_distance
           
        else:  
            threshold = simpledialog.askfloat("FAST Parameters", "Enter intensity threshold (0-255):", minvalue=0.0, maxvalue=255.0, parent=parent)
            if threshold is None: return None
            params["threshold"] = threshold
            
        params["method"] = method
        
        display_mode = dialog_options(parent, "Display Mode", "How do you want to visualize the results?", self.display_options)
        if display_mode is None: return None
        params["display_mode"] = display_mode
        
        return params
    
    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        gray = np.asarray(document.current.convert("L"))
        
        params = self.choose_parameters(parent)
        if params is None: return None
        
        method = params["method"]
        display_mode = params["display_mode"]

        if method == self.options[0]:  
            response = feature.corner_harris(gray)
            corners = feature.corner_peaks(response, threshold_rel=params["threshold"], min_distance=int(params["min_distance"]))
        elif method == self.options[1]:  
            response = feature.corner_fast(gray, threshold=params["threshold"])
            corners = feature.corner_peaks(response)

        if display_mode == self.display_options[0]:
            output_image = Image.fromarray(gray2rgb(gray))
            draw = ImageDraw.Draw(output_image)
            for y, x in corners:
                draw.ellipse([x-2, y-2, x+2, y+2], outline=(255, 0, 0), width=1)
                
        else:

            resp_min, resp_max = response.min(), response.max()
            if resp_max > resp_min:
                resp_norm = (response - resp_min) / (resp_max - resp_min)
            else:
                resp_norm = np.zeros_like(response)
                
            colored_response = cm.viridis(resp_norm) 
            
            output_image = Image.fromarray((colored_response[:, :, :3] * 255).astype(np.uint8))

        return ToolResult(
            image=output_image,
            message="Corner detection completed",
            details={"Method": method, "Number of corners": len(corners), "Display": display_mode},
        )