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

class Convolution(ForensicsTool):
    def __init__(self):
        self.tool_id = "convolution"
        self.title = "Convolution"
        self.category = "Set3"
        self.description = "Different kernels, directions and convolutions to apply"
        self.requires_image = True
        self.kernel_options = [
            "Average",
            "Gaussian",
            "Laplacian",
            "Sobel",
            "Prewitt"
        ]

        self.directions = ["2D", "Vertical", "Horizontal"]


    def _choose_kernel(self, parent: tk.Misc) -> str | None:
        return  dialog_options(parent, "Kernel selection", "Choose one of the following kernels", self.kernel_options)


    def _choose_direction(self, parent: tk.Misc) -> str | None:
        return dialog_options(parent, "Direction selection", "Choose the direction of the kernel", self.directions)

    def _choose_size(self, parent: tk.Misc, length_x: int, length_y: int, direction: str) -> tuple | None:
        size_x = size_y = 1
        if direction != "Vertical":
            size_x = simpledialog.askinteger("Size", "Horizontal size of the kernel",
                                            minvalue=1, maxvalue=length_x, parent=parent)
            if size_x is None: return None
        if direction != "Horizontal":
            size_y = simpledialog.askinteger("Size", "Vertical size of the kernel",
                                            minvalue=1, maxvalue=length_y, parent=parent)
            if size_y is None: return None
        return size_x, size_y
    
    def _generate_average_kernel(self, size_x: int, size_y: int) -> np.ndarray:
        array = np.ones((size_y, size_x))
        return array / array.sum()

    def _choose_sigma(self, parent):
        return simpledialog.askfloat("Sigma", "Choose sd for the gaussian kernel", minvalue=0.1, maxvalue=1e9, parent=parent)

    def _generate_gaussian_kernel(self, size_x, size_y, sigma):
        x = np.arange(size_x) - (size_x - 1) / 2
        y = np.arange(size_y) - (size_y - 1) / 2
        xx, yy = np.meshgrid(x, y)              
        K = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return K / K.sum()

    def _generate_directions_kernel(self, direction: str, kernel: str) -> np.ndarray:
        if kernel == "sobel":
            if direction.lower() == "vertical":
                return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            else:
                return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
        elif kernel.lower() == "prewitt":
            if direction.lower() == "vertical":
                return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            else:
                return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    def _generate_laplacian_kernel(self) -> np.ndarray:
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    def run(self, parent, document):
        image = document.current
        assert image is not None
        arr = np.asarray(image)

        

        choice = self._choose_kernel(parent)
        
        if not choice: return None
        kernel = choice.lower()
        if kernel in ["prewitt", "sobel", "laplacian"]:
            arr = np.asarray(image.convert("L"), dtype=np.float64)
            length_y, length_x = arr.shape 
            has_color = False
        else:
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            arr = np.asarray(image, dtype=np.float64)
            has_color = len(arr.shape) == 3
            
            if has_color:
                length_y, length_x, channels = arr.shape
            else:
                length_y, length_x = arr.shape
                channels = 1

        if kernel in ("average", "gaussian"):
            direction = self._choose_direction(parent)

            if not direction: return None

            sizes = self._choose_size(parent, length_x, length_y, direction)
            if sizes is None: return None

            size_x, size_y = sizes

            if kernel == "average":
                kernel_array = self._generate_average_kernel(size_x, size_y)
            else:
                sigma = self._choose_sigma(parent)

                if sigma is None: return None

                kernel_array = self._generate_gaussian_kernel(size_x, size_y, sigma)

            if has_color:
                out = np.zeros_like(arr)
                for i in range(channels): 
                    out[:,:,i] = ndimage.convolve(arr[:,:,i], kernel_array)
            
            else: out = ndimage.convolve(arr, kernel_array)

        elif kernel in ("prewitt", "sobel"):
            direction = self._choose_direction(parent)
            if not direction: return None
            if direction == "2D":   
                gx = ndimage.convolve(arr, self._generate_directions_kernel("horizontal", kernel))
                gy = ndimage.convolve(arr, self._generate_directions_kernel("vertical", kernel))
                out = np.hypot(gx, gy)
            else:
                out = np.abs(ndimage.convolve(arr, self._generate_directions_kernel(direction, kernel)))

        else: 
            out = np.abs(ndimage.convolve(arr, self._generate_laplacian_kernel()))

        if kernel in ("prewitt", "sobel", "laplacian") and out.max() > 0:
            out = out / out.max() * 255
        new_image = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        return ToolResult(
            image=new_image,
            message="Result.",
            details={"Operation": f"Applied kernel {kernel}", "Output mode": new_image.mode},
        )


    
        