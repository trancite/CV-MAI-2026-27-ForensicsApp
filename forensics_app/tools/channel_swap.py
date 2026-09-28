from __future__ import annotations

import tkinter as tk

from PIL import ImageOps, Image
from tkinter import simpledialog
import tkinter as tk
from tkinter import ttk
from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
import numpy as np

class ChannelSwap(ForensicsTool):
    def __init__(self):
        self.tool_id = "channelswap"
        self.title = "Channel Swap"
        self.category = "Set2"
        self.description = "It swaps the (R, G, B) channels"
        self.requires_image = True


    def __choose_swap(self, parent: tk.Misc) -> list | None:
        first_color = simpledialog.askinteger("First color", "0 for Red, 1 for Green and 2 Blue", parent=parent)
        second_color = simpledialog.askinteger("Second color", "0 for Red, 1 for Green and 2 Blue", parent=parent)

        if None in (first_color, second_color): return None

        return [first_color, second_color]


    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None
        swap = self.__choose_swap(parent)
        if swap is None:
            return None
        rgb_image = document.current.convert("RGB")
        image_tensor = np.array(rgb_image)
        image_tensor[:,:,swap] = image_tensor[:,:,swap[::-1]]
        new_image = Image.fromarray(image_tensor)
        return ToolResult(
                            image=new_image,
                            message="New image with swapped channels.",
                            details={"Operation": "Channel swap", "Output mode": new_image.mode},
                        )

        
