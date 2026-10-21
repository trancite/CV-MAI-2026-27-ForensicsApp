
from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
import tkinter as tk
from tkinter import filedialog, simpledialog
from skimage.transform import resize
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import correlate
from skimage import feature

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utilities import dialog_options
from skimage import exposure


class PersonDetectionHog(ForensicsTool):
    def __init__(self):
        self.tool_id = "persondetectionhog"
        self.title = "Person Detection with HOG"
        self.category = "Set4"
        self.description = "Find a person in the image using HOG."
        self.requires_image = True
        self.metric_options = {
            "Euclidean": lambda a, b: float(np.linalg.norm(
                np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
            )),
            "Dot product": lambda a, b: float(np.dot(
                np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), out = None
            )),
            "Manhattan": lambda a, b: float(np.abs(
                np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
            ).sum()),
            "Chi-squared": lambda a, b: float(0.5 * np.sum(
                (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2
                / (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64) + 1e-12)
            )),
            "Minimum": lambda a, b: float(np.minimum(a, b).sum()),
        }

    def _choose_scale(self, parent: tk.Misc) -> float | None:
        return simpledialog.askfloat(
            "Background image scale",
            "Positive scale factor (1 = original, 0.5 = half size, 2 = double size):",
            initialvalue=1.0, minvalue=0.0, parent=parent,
        )
    
    def _choose_metric(self, parent: tk.Misc) -> Callable[[np.ndarray, np.ndarray], float] | None:
        choice = dialog_options(
            parent, "Metric selection", "Choose a metric", list(self.metric_options)
        )
        if choice is None:
            return None
        return self.metric_options[choice]

    def _choose_hog_parameters(self, parent: tk.Misc, person_h: int, person_w: int) -> tuple[tuple[int, int], tuple[int, int], int] | None:
        selection = dialog_options(parent, "Hog parameters", "Do you want to choose the hog parameters?", ["No", "Yes"])
        if selection is None:
            return None
        if selection == "No":
            return (8, 8), (2, 2), 16

        pixels_per_cell = simpledialog.askinteger(title="Pixels per cell", prompt="Enter the number of pixels per cell (at least 2 for visualization):", minvalue=2, parent=parent)
        if pixels_per_cell is None: return None
        
        cells_per_block = simpledialog.askinteger(title="Cells per block", prompt="Enter the number of cells per block:", minvalue=1, parent=parent)
        if cells_per_block is None: return None
        
        pixels_per_cell = (pixels_per_cell, pixels_per_cell)
        cells_per_block = (cells_per_block, cells_per_block)

        orientations = simpledialog.askinteger("Number of orientations", "Choose the number of orientations in hog", minvalue=1, maxvalue=360, parent=parent)
        if orientations is None: return None
        return pixels_per_cell, cells_per_block, orientations

    def _load_image_person(self, parent: tk.Misc) -> np.ndarray | None:
        template_path = filedialog.askopenfilename(
            title="Select a person image.",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
            parent=parent,
        )
        if not template_path: return None

        with Image.open(template_path) as template_file:
            template_image = template_file.convert("RGB")
        return np.array(template_image)

    def _compute_differences(
        self, image: np.ndarray, person: np.ndarray, hog_parameters: tuple, metric,
        maximize: bool = False,
    ) -> tuple[float, tuple[int, int]]:
        window_height, window_width = person.shape[:2]
        image_height, image_width = image.shape[:2]
        if window_height > image_height or window_width > image_width:
            raise ValueError("The person template must not be larger than the working image.")

        pixels_per_cell, cells_per_block, orientations = hog_parameters
        if min(pixels_per_cell) < 1 or min(cells_per_block) < 1 or orientations < 1:
            raise ValueError("HOG cell sizes, block sizes, and orientations must be positive.")
        required_height = pixels_per_cell[0] * cells_per_block[0]
        required_width = pixels_per_cell[1] * cells_per_block[1]
        if window_height < required_height or window_width < required_width:
            raise ValueError(
                f"These HOG parameters require a person template of at least "
                f"{required_width} x {required_height} pixels; "
                f"the template is {window_width} x {window_height}."
            )

        person_fd = self._compute_template_hog(person, hog_parameters)
        best_score = -np.inf if maximize else np.inf
        best_position = (0, 0)
        for row in range(image_height - window_height + 1):
            for column in range(image_width - window_width + 1):
                current_slice = image[row : row + window_height, column : column + window_width]
                image_fd = feature.hog(
                    current_slice, orientations=orientations,
                    pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                    visualize=False, channel_axis=-1 if current_slice.ndim == 3 else None,
                )
                score = float(metric(person_fd, image_fd))
                better = score > best_score if maximize else score < best_score
                if better:
                    best_score = score
                    best_position = (row, column)

        return best_score, best_position

    def _compute_template_hog(self, person: np.ndarray, hog_parameters: tuple) -> np.ndarray:
        pixels_per_cell, cells_per_block, orientations = hog_parameters[0], hog_parameters[1], hog_parameters[2]
        return feature.hog(
            person, orientations=orientations, pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block, visualize=False,
            channel_axis=-1 if person.ndim == 3 else None,
        )

    def _paint_red_square(
        self, image: Image.Image, position: tuple[int, int], window_shape: tuple[int, int]
    ) -> Image.Image:
        """Outline the matching window; position and shape use row, column order."""
        row, column = position
        height, width = window_shape
        output = image.convert("RGB")
        ImageDraw.Draw(output).rectangle(
            (column, row, column + width - 1, row + height - 1),
            outline=(255, 0, 0), width=2,
        )
        return output


    def _compute_scaled_match(
        self, image: np.ndarray, person: np.ndarray, scale: float,
        hog_parameters: tuple, metric, maximize: bool,
    ) -> tuple[float, tuple[int, int], tuple[int, int], int]:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("The background scale must be positive and finite.")
        height, width = image.shape[:2]
        window_height, window_width = person.shape[:2]
        scaled_height = max(1, round(height * scale))
        scaled_width = max(1, round(width * scale))
        if scaled_height < window_height or scaled_width < window_width:
            raise ValueError(
                f"The scaled background is {scaled_width} x {scaled_height}, smaller than "
                f"the {window_width} x {window_height} person template. Choose a larger scale."
            )
        if (scaled_height, scaled_width) == (height, width):
            current_scaled = image
        else:
            current_scaled = resize(
                image, (scaled_height, scaled_width) + image.shape[2:],
                preserve_range=True,
                anti_aliasing=scaled_height < height or scaled_width < width,
            )
        score, position = self._compute_differences(
            current_scaled, person, hog_parameters, metric, maximize=maximize,
        )
        row, column = position
        top = row * height // scaled_height
        left = column * width // scaled_width
        bottom = ((row + window_height) * height + scaled_height - 1) // scaled_height
        right = ((column + window_width) * width + scaled_width - 1) // scaled_width
        windows_compared = (scaled_height - window_height + 1) * (scaled_width - window_width + 1)
        return score, (top, left), (bottom - top, right - left), windows_compared

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        person = self._load_image_person(parent)
        if person is None:
            return None
        person_h, person_w = person.shape[0], person.shape[1]
        hog_parameters = self._choose_hog_parameters(parent, person_h, person_w)
        if hog_parameters is None:
            return None
        metric = self._choose_metric(parent)
        if metric is None:
            return None
        scale = self._choose_scale(parent)
        if scale is None:
            return None

        metric_name = next(name for name, function in self.metric_options.items() if function is metric)
        source = document.current
        image = np.asarray(source.convert("RGB"))
        
        maximize = metric_name in ("Dot product", "Minimum")
        score, position, window_shape, windows_compared = self._compute_scaled_match(
            image, person, scale, hog_parameters, metric, maximize,
        )
        row, column = position
        new_image = self._paint_red_square(source, position, window_shape)

        return ToolResult(
            image=new_image,
            message=f"Best HOG match marked in red ({metric_name}).",
            details={
                "Operation": "Person detection with HOG",
                "Metric": metric_name,
                "Best score": score,
                "Selection": "Maximum similarity" if maximize else "Minimum distance",
                "Top-left (x, y)": (column, row),
                "Window size": (window_shape[1], window_shape[0]),
                "Template size": (person.shape[1], person.shape[0]),
                "Background scale": scale,
                "Windows compared": windows_compared,
                "Pixels per cell": hog_parameters[0],
                "Cells per block": hog_parameters[1],
                "Orientations": hog_parameters[2],
                "Output mode": new_image.mode,
            },
        )
    

     



    
