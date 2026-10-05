from __future__ import annotations

import tkinter as tk

from PIL import ImageOps, Image
from tkinter import simpledialog
from tkinter import colorchooser

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utilities import dialog_options
import numpy as np

class ChannelSplitTool(ForensicsTool):
    def __init__(self):
        self.tool_id = "channelsplit"
        self.title = "Channel Split"
        self.category = "Set2"
        self.description = "It allows splitting the image in three different channels"
        self.requires_image = True
        self.options = ["Red", "Green", "Blue"]

    def _choose_channel(self, parent: tk.Misc) -> str | None:
            return dialog_options(parent, "Channel split",
                                  "Choose a channel", self.options)

    
    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        channel = self._choose_channel(parent)
        if channel is None:
            return None
        if channel == "Red": channel = 0
        elif channel == "Green": channel = 1
        else: channel = 2
        rgb_image = document.current.convert("RGB")
        image_tensor = np.array(rgb_image)
        new_image = Image.fromarray(image_tensor[:,:,channel])
        return ToolResult(
                    image=new_image,
                    message=f"New image showing the {channel}th channel.",
                    details={"Operation": "Channel split", "Output mode": new_image.mode},
                )