"""Find the best HOG template match with a configurable sliding-window step."""

from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utilities import dialog_options


class PersonDetectionHog(ForensicsTool):
    def __init__(self):
        self.tool_id = "persondetectionhog"
        self.title = "Person Detection with HOG"
        self.category = "Set4"
        self.description = "Find a person in the image using HOG."
        self.requires_image = True
        self.metric_options = {
            "Euclidean": lambda a, b: float(np.linalg.norm(a - b)),
            "Dot product": lambda a, b: float(np.dot(a, b)),
            "Manhattan": lambda a, b: float(np.abs(a - b).sum()),
            "Chi-squared": lambda a, b: float(0.5 * np.sum(
                (a - b) ** 2 / (a + b + 1e-12)
            )),
            "Minimum": lambda a, b: float(np.minimum(a, b).sum()),
        }

    def _choose_step_size(self, parent: tk.Misc) -> int | None:
        return simpledialog.askinteger(
            "Step size", "Sliding-window step in pixels (1 checks every position):",
            initialvalue=8, minvalue=1, parent=parent,
        )

    def _choose_metric(self, parent: tk.Misc) -> Callable[[np.ndarray, np.ndarray], float] | None:
        choice = dialog_options(
            parent, "Metric selection", "Choose a metric", list(self.metric_options)
        )
        if choice is None:
            return None
        return self.metric_options[choice]

    def _choose_hog_parameters(self, parent: tk.Misc) -> tuple[tuple[int, int], tuple[int, int], int] | None:
        selection = dialog_options(parent, "Hog parameters", "Do you want to choose the hog parameters?", ["No", "Yes"])
        if selection is None:
            return None
        if selection == "No":
            return (8, 8), (2, 2), 16

        pixels_per_cell = simpledialog.askinteger("Pixels per cell", "Enter the number of pixels per cell:", minvalue=2, parent=parent)
        if pixels_per_cell is None: return None

        cells_per_block = simpledialog.askinteger(title="Cells per block", prompt="Enter the number of cells per block:", minvalue=1, parent=parent)
        if cells_per_block is None: return None

        orientations = simpledialog.askinteger("Number of orientations", "Choose the number of orientations in hog", minvalue=1, maxvalue=360, parent=parent)
        if orientations is None: return None
        return (pixels_per_cell, pixels_per_cell), (cells_per_block, cells_per_block), orientations

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
        maximize: bool = False, step_size: int = 1,
    ) -> tuple[float, tuple[int, int]]:
        if step_size < 1:
            raise ValueError("Step size must be at least 1 pixel.")
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

        person_fd = self._compute_hog(person, hog_parameters)
        best_score = -np.inf if maximize else np.inf
        best_position = (0, 0)
        for row in range(0, image_height - window_height + 1, step_size):
            for column in range(0, image_width - window_width + 1, step_size):
                current_slice = image[row : row + window_height, column : column + window_width]
                image_fd = self._compute_hog(current_slice, hog_parameters)
                score = float(metric(person_fd, image_fd))
                better = score > best_score if maximize else score < best_score
                if better:
                    best_score = score
                    best_position = (row, column)

        return best_score, best_position

    def _compute_hog(self, image: np.ndarray, hog_parameters: tuple) -> np.ndarray:
        pixels_per_cell, cells_per_block, orientations = hog_parameters
        return feature.hog(
            image, orientations=orientations, pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block, visualize=False,
            channel_axis=-1 if image.ndim == 3 else None,
        )

    def _paint_red_square(
        self, image: Image.Image, position: tuple[int, int], window_shape: tuple[int, int]
    ) -> Image.Image:
        row, column = position
        height, width = window_shape
        output = image.convert("RGB")
        ImageDraw.Draw(output).rectangle(
            (column, row, column + width - 1, row + height - 1),
            outline=(255, 0, 0), width=2,
        )
        return output

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        person = self._load_image_person(parent)
        if person is None:
            return None
        person_h, person_w = person.shape[:2]
        hog_parameters = self._choose_hog_parameters(parent)
        if hog_parameters is None:
            return None
        metric = self._choose_metric(parent)
        if metric is None:
            return None
        step_size = self._choose_step_size(parent)
        if step_size is None:
            return None

        metric_name = next(name for name, function in self.metric_options.items() if function is metric)
        source = document.current
        image = np.asarray(source.convert("RGB"))

        maximize = metric_name in ("Dot product", "Minimum")
        score, position = self._compute_differences(
            image, person, hog_parameters, metric, maximize=maximize, step_size=step_size,
        )
        row, column = position
        new_image = self._paint_red_square(source, position, (person_h, person_w))
        windows_compared = (
            len(range(0, source.height - person_h + 1, step_size))
            * len(range(0, source.width - person_w + 1, step_size))
        )

        return ToolResult(
            image=new_image,
            message=f"Best HOG match marked in red ({metric_name}).",
            details={
                "Operation": "Person detection with HOG",
                "Metric": metric_name,
                "Best score": score,
                "Selection": "Maximum similarity" if maximize else "Minimum distance",
                "Top-left (x, y)": (column, row),
                "Window size": (person_w, person_h),
                "Template size": (person_w, person_h),
                "Step size": step_size,
                "Windows compared": windows_compared,
                "Pixels per cell": hog_parameters[0],
                "Cells per block": hog_parameters[1],
                "Orientations": hog_parameters[2],
                "Output mode": new_image.mode,
            },
        )
