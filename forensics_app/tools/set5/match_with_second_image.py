"""Match the working image against a selected reference image."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ToolResult
from .keypoint_tools import KeypointMatchingTool


class MatchWithSecondImageTool(KeypointMatchingTool):
    tool_id = "match_with_second_image"
    title = "Match with second image"
    description = "Match ORB or SIFT keypoints with a reference image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        algorithm = self._choose_algorithm(parent)
        if algorithm is None:
            return None
        parameters = self._choose_parameters(parent, algorithm)
        if parameters is None:
            return None
        scoring = self._choose_scoring(parent)
        if scoring is None:
            return None

        ref_path = filedialog.askopenfilename(
            title="Select a reference image.",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
            parent=parent,
        )
        if not ref_path:
            return None
        with Image.open(ref_path) as reference_file:
            reference_image = reference_file.convert("RGB")

        source_image = document.current
        source_keypoints, source_descriptors = self._generate_keypoints_and_descriptors(
            np.asarray(source_image.convert("L")), algorithm, parameters,
        )
        reference_keypoints, reference_descriptors = self._generate_keypoints_and_descriptors(
            np.asarray(reference_image.convert("L")), algorithm, parameters,
        )
        metric = "hamming" if algorithm == "ORB" else "euclidean"
        points = (source_keypoints, reference_keypoints) if scoring == "RANSAC" else None
        matches, score = self._compare_descriptors(
            source_descriptors, reference_descriptors, metric=metric, keypoints=points,
        )
        return ToolResult(
            message=f"Found {len(matches)} keypoint matches.",
            details={
                "Method": algorithm,
                "Scoring": scoring,
                "Matches": len(matches),
                "Matching score": score,
            },
            match_preview=self._make_match_preview(
                source_image, reference_image, source_keypoints, reference_keypoints,
                matches, ref_path,
            ),
        )
