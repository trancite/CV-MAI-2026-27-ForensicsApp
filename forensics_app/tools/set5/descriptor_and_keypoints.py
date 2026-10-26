"""Detect and highlight ORB or SIFT keypoints in the working image."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from skimage.color import gray2rgb

from forensics_app.core import ImageDocument
from ..base import ToolResult
from .keypoint_tools import KeypointTool


class DescriptorAndKeypointsTool(KeypointTool):
    tool_id = "descriptor_and_keypoints"
    title = "Descriptor and Keypoints detection"
    description = "Detect and highlight ORB or SIFT keypoints in the image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        algorithm = self._choose_algorithm(parent)
        if algorithm is None: return None
        parameters = self._choose_parameters(parent, algorithm)
        if parameters is None: return None

        pixels = np.asarray(document.current.convert("L"))
        keypoints, _ = self._generate_keypoints_and_descriptors(pixels, algorithm, parameters)
        if len(keypoints) == 0:
            return ToolResult(
                message="No keypoints found in the working image.",
                details={"Method": algorithm, "Number of keypoints": 0},
            )
        rgb_image = gray2rgb(pixels)
        coordinates = np.round(keypoints).astype(int)
        rgb_image[coordinates[:, 0], coordinates[:, 1]] = [255, 0, 0]
        return ToolResult(
            image=Image.fromarray(rgb_image),
            message="Keypoints detected",
            details={"Method": algorithm, "Number of keypoints": len(keypoints)},
        )
