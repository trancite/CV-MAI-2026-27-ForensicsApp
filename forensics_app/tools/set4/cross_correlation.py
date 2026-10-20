"""Template matching with red markers at detected match centers."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import correlate
from skimage import feature

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utilities import dialog_options
from skimage import exposure


class CrossCorrelationTool(ForensicsTool):
    def __init__(self):
        self.tool_id = "crosscorrelation"
        self.title = "Cross-correlation"
        self.category = "Set4"
        self.description = "Find a template in the image and mark matching peaks in red."
        self.requires_image = True
        self.options = [
            "Sum square differences",
            "Cross correlation with normalized image",
            "Normalized cross-correlation",
        ]
        self.transformation_options = [
            "No transformation",
            "Rotation",
            "Horizontal flip",
            "Vertical flip",
            "Transposition",
        ]

    def _choose_transformation(self, parent: tk.Misc) -> str | None:
        return dialog_options(
            parent, "Transformation selection", "Choose a template transformation",
            self.transformation_options,
        )

    def _choose_angle(self, parent: tk.Misc) -> float | None:
        return simpledialog.askfloat(
            "Angle selection", "Choose an angle to rotate the template (counterclockwise)",
            minvalue=0, maxvalue=360, parent=parent,
        )

    def _choose_filter(self, parent: tk.Misc) -> str | None:
        return dialog_options(
            parent, "Template matching", "Choose a matching method", self.options
        )

    def _choose_threshold(self, parent: tk.Misc, choice: str) -> float | None:
        if choice == self.options[0]:
            prompt = "Minimum pixel similarity (0 to 1; 1 means identical):"
        elif choice == self.options[1]:
            prompt = "Minimum strength relative to the best match (0 to 1):"
        else:
            prompt = "Minimum normalized correlation (0 to 1; 1 means a perfect match):"
        return simpledialog.askfloat(
            "Peak threshold", prompt, initialvalue=0.8,
            minvalue=0.0, maxvalue=1.0, parent=parent,
        )

    def _sum_square_differences(self, image: np.ndarray, template: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float64)
        template = np.asarray(template, dtype=np.float64)
        image_squares = correlate(image**2, np.ones_like(template), mode="valid")
        correlation = correlate(image, template, mode="valid")
        return np.maximum(image_squares - 2 * correlation + (template**2).sum(), 0)

    def _cross_correlation_normalized_image(self, image: np.ndarray, template: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float64)
        template = np.asarray(template, dtype=np.float64)
        return correlate(image - np.mean(image), template, mode="valid")

    def _normalized_cross_correlation(self, image: np.ndarray, template: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float64)
        template = np.asarray(template, dtype=np.float64)
        if np.ptp(template) == 0: raise ValueError("Normalized cross-correlation needs a template with varying intensities.")
        return feature.match_template(image, template, pad_input=False)

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        choice = self._choose_filter(parent)
        if choice is None: return None


        template_path = filedialog.askopenfilename(
            title="Select a template image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
            parent=parent,
        )
        if not template_path: return None

        source = document.current

        with Image.open(template_path) as template_file:
            template_image = template_file.convert("L")

        transformation = self._choose_transformation(parent)
        if transformation is None:
            return None
        if transformation == "Rotation":
            angle = self._choose_angle(parent)
            if angle is None:
                return None
            template_image = template_image.rotate(angle, expand=True)
        elif transformation != "No transformation":
            operations = {
                "Horizontal flip": Image.Transpose.FLIP_LEFT_RIGHT,
                "Vertical flip": Image.Transpose.FLIP_TOP_BOTTOM,
                "Transposition": Image.Transpose.TRANSPOSE,
            }
            template_image = template_image.transpose(operations[transformation])

        if template_image.width > source.width or template_image.height > source.height:
            raise ValueError("The template must not be larger than the working image.")

        threshold = self._choose_threshold(parent, choice)
        if threshold is None: return None

        image = np.asarray(source.convert("L"), dtype=np.float64) / 255.0
        template = np.asarray(template_image, dtype=np.float64) / 255.0
        if choice == self.options[0]:
            differences = self._sum_square_differences(image, template)
            similarity = np.clip(1 - np.sqrt(differences), 0, 1)
        
        elif choice == self.options[1]:
            correlation = self._cross_correlation_normalized_image(image, template)
            similarity = correlation
        
        else:
            similarity = self._normalized_cross_correlation(image, template)

        peaks = feature.peak_local_max(
            similarity,
            min_distance=max(1, min(template.shape) // 2),
            threshold_abs=threshold - 1e-12,
            exclude_border=False,
        )
        peaks = peaks[similarity[tuple(peaks.T)] > 0]

        source_array = np.array(source)
        corrected_array = exposure.adjust_gamma(source_array, gamma=2)
        new_image = Image.fromarray(corrected_array).convert("RGB")
        drawing = ImageDraw.Draw(new_image)
        centers = []
        for row, column in peaks:
            x = int(column) + template_image.width // 2
            y = int(row) + template_image.height // 2
            centers.append((x, y))
            drawing.line((x - 3, y, x + 3, y), fill="red")
            drawing.line((x, y - 3, x, y + 3), fill="red")

        return ToolResult(
            image=new_image,
            message=f"Marked {len(centers)} matching peaks in red.",
            details={
                "Operation": choice,
                "Template": Path(template_path).name,
                "Template transformation": transformation,
                "Template size": template_image.size,
                "Peak threshold": threshold,
                "Detected peaks": len(centers),
                "Peak centers (x, y)": centers,
                "Output mode": new_image.mode,
            },
        )
