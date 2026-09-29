from __future__ import annotations

import tkinter as tk

from PIL import ImageOps, Image
from tkinter import simpledialog
from tkinter import colorchooser

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
import numpy as np

class ChannelSplitTool(ForensicsTool):
    def __init__(self):
        self.tool_id = "channelsplit"
        self.title = "Channel Split"
        self.category = "Set2"
        self.description = "It allows splitting the image in three different channels"
        self.requires_image = True

    def __choose_color(self, parent: tk.Misc) -> np.array | None:
        rgb, hex_color = tk.colorchooser.askcolor(title="Choose a color", parent=parent)
        red, green, blue = rgb
        if None in (red, green, blue): return None
        if red == green == blue == 0: return np.full((3,), 1/3)
        colors = np.array([red, green, blue])
        return colors/colors.sum()
    
    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None
        colors = self.__choose_color(parent)
        if colors is None:
            return None
        rgb_image = document.current.convert("RGB")
        new_channels = colors.reshape(1,1,-1)
        image_tensor = np.array(rgb_image)
        new_image_tensor = (image_tensor * new_channels).sum(axis = -1).round().astype(np.uint8)
        new_image = Image.fromarray(new_image_tensor)
        return ToolResult(
                    image=new_image,
                    message="New channel image.",
                    details={"Operation": "Channel split", "Output mode": new_image.mode},
                )