"""Find the folder image with the strongest keypoint correspondence."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ToolResult
from .keypoint_tools import KeypointMatchingTool


class FindClosestImageTool(KeypointMatchingTool):
    tool_id = "find_closest_image"
    title = "Find closest image"
    description = "Find the closest image in a folder using ORB or SIFT matches."

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

        current_path = document.path.resolve() if document.path else None
        folder = filedialog.askdirectory(
            title="Select the folder to search", parent=parent, mustexist=True,
            initialdir=str(current_path.parent if current_path else Path.cwd()),
        )
        if not folder:
            return None
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        candidates = sorted(
            path for path in Path(folder).iterdir()
            if path.is_file() and path.suffix.lower() in extensions and path.resolve() != current_path
        )
        if not candidates:
            return ToolResult(message="No other supported images found in the selected folder.")

        source_image = document.current
        source_pixels = np.asarray(source_image.convert("RGB"))
        source_keypoints, source_descriptors = self._generate_keypoints_and_descriptors(
            np.asarray(source_image.convert("L")), algorithm, parameters,
        )
        if len(source_keypoints) == 0:
            return ToolResult(
                message="No keypoints found in the working image.",
                details={"Method": algorithm, "Number of keypoints": 0},
            )

        metric = "hamming" if algorithm == "ORB" else "euclidean"
        best = None
        best_rank = (0.0, 0)
        for path in candidates:
            try:
                with Image.open(path) as candidate_file:
                    candidate_image = candidate_file.convert("RGB")
                if np.array_equal(source_pixels, np.asarray(candidate_image)):
                    continue
                keypoints, descriptors = self._generate_keypoints_and_descriptors(
                    np.asarray(candidate_image.convert("L")), algorithm, parameters,
                )
                points = (source_keypoints, keypoints) if scoring == "RANSAC" else None
                matches, score = self._compare_descriptors(
                    source_descriptors, descriptors, metric=metric, keypoints=points,
                )
            except (OSError, RuntimeError, ValueError):
                continue
            if len(matches) and (score, len(matches)) > best_rank:
                best_rank = (score, len(matches))
                best = (path, candidate_image, keypoints, matches)

        if best is None:
            return ToolResult(message="No matching keypoints found in the selected folder.")
        path, candidate_image, keypoints, matches = best
        return ToolResult(
            message=f"Closest image: {path.name} ({len(matches)} matches, score {best_rank[0]:.3f}).",
            details={
                "Method": algorithm,
                "Scoring": scoring,
                "Closest image": str(path),
                "Matching score": best_rank[0],
                "Matches": len(matches),
            },
            match_preview=self._make_match_preview(
                source_image, candidate_image, source_keypoints, keypoints, matches, path,
            ),
        )
