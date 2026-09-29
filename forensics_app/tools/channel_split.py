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

    def __choose_channel(self, parent: tk.Misc) -> int | None:
        color = simpledialog.askinteger("Choose color", "1 for Red, 2 for Green and 3 Blue", parent=parent)
        if color is None:
            return None
        return color - 1
    
    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        channel = self.__choose_channel(parent)
        if channel is None:
            return None
        rgb_image = document.current.convert("RGB")
        image_tensor = np.array(rgb_image)
        new_image = Image.fromarray(image_tensor[:,:,channel])
        return ToolResult(
                    image=new_image,
                    message="New channel image.",
                    details={"Operation": "Channel split", "Output mode": new_image.mode},
                )