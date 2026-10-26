"""Shared ORB/SIFT settings, extraction, matching, and match previews."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import ORB, SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

from ..base import ForensicsTool, MatchPreview
from ..utilities import dialog_options


class KeypointTool(ForensicsTool):
    """Common extraction settings for the three Set 5 keypoint tools."""

    category = "Set5"
    algorithms = ["ORB", "SIFT"]

    def _choose_algorithm(self, parent: tk.Misc) -> str | None:
        return dialog_options(parent, "Algorithm selection", "Choose between ORB or SIFT", self.algorithms)

    def _choose_n_keypoints(self, parent: tk.Misc) -> int | None:
        return simpledialog.askinteger(
            "Number of keypoints", "Choose the maximum number of ORB keypoints",
            minvalue=1, parent=parent,
        )

    def _choose_sift_parameters(self, parent: tk.Misc) -> tuple[int, int, float] | None:
        n_octaves = simpledialog.askinteger("Number of octaves", "Choose the number of SIFT octaves", minvalue=1, parent=parent)
        if n_octaves is None: return None
        n_scales = simpledialog.askinteger("Number of scales", "Choose the number of SIFT scales", minvalue=1, parent=parent)
        if n_scales is None: return None
        sigma = simpledialog.askfloat(
            "Sigma", "Choose the initial Gaussian blur sigma for SIFT (at least 1.0)",
            initialvalue=1.6, minvalue=1.0, parent=parent,
        )
        if sigma is None: return None
        return n_octaves, n_scales, sigma

    def _choose_parameters(self, parent: tk.Misc, algorithm: str) -> dict | None:
        if algorithm == self.algorithms[0]:
            n_keypoints = self._choose_n_keypoints(parent)
            return None if n_keypoints is None else {"n_keypoints": n_keypoints}
        parameters = self._choose_sift_parameters(parent)
        if parameters is None: return None
        n_octaves, n_scales, sigma = parameters
        return {"n_octaves": n_octaves, "n_scales": n_scales, "sigma_min": sigma}

    def _generate_keypoints_and_descriptors(self, image: np.ndarray, algorithm: str, parameters: dict) -> tuple:
        if algorithm == self.algorithms[0]:
            extractor = ORB(**parameters)
        else:
            sigma = parameters.get("sigma_min", 1.6)
            if not np.isfinite(sigma) or sigma < 1.0:
                raise ValueError("SIFT sigma must be finite and at least 1.0.")
            extractor = SIFT(**parameters)
        try:
            extractor.detect_and_extract(image)
        except RuntimeError as error:
            if not str(error).startswith(f"{algorithm} found no features"):
                raise
            size, dtype = (256, bool) if algorithm == "ORB" else (128, np.uint8)
            return np.empty((0, 2), dtype=float), np.empty((0, size), dtype=dtype)
        return extractor.keypoints, extractor.descriptors


class KeypointMatchingTool(KeypointTool):
    """Common scoring and visualization for image matching and folder search."""

    matching_scores = ["Default", "RANSAC"]

    def _choose_scoring(self, parent: tk.Misc) -> str | None:
        return dialog_options(parent, "Matching score", "Choose how to score matches", self.matching_scores)

    def _compare_descriptors(
        self, descriptors_1: np.ndarray, descriptors_2: np.ndarray,
        metric: str = "euclidean", keypoints: tuple | None = None,
    ) -> tuple[np.ndarray, float]:
        if len(descriptors_1) == 0 or len(descriptors_2) == 0:
            return np.empty((0, 2), dtype=int), 0.0
        matches = match_descriptors(
            descriptors_1, descriptors_2, metric=metric, cross_check=True, max_ratio=0.8,
        )
        if keypoints is not None:
            if len(matches) < 4:
                return matches[:0], 0.0
            coordinates = tuple(points[matches[:, i]][:, ::-1] for i, points in enumerate(keypoints))
            _, inliers = ransac(
                coordinates, ProjectiveTransform, min_samples=4,
                residual_threshold=3, max_trials=100, rng=0,
            )
            matches = matches[inliers] if inliers is not None else matches[:0]
        score = len(matches) / min(len(descriptors_1), len(descriptors_2))
        return matches, score

    def _draw_matches(
        self, image_1: Image.Image, image_2: Image.Image,
        keypoints_1: np.ndarray, keypoints_2: np.ndarray, matches: np.ndarray,
    ) -> Image.Image:
        output = Image.new("RGB", (image_1.width + image_2.width, max(image_1.height, image_2.height)))
        output.paste(image_1.convert("RGB"), (0, 0))
        output.paste(image_2.convert("RGB"), (image_1.width, 0))
        draw = ImageDraw.Draw(output)
        for index_1, index_2 in matches:
            y_1, x_1 = keypoints_1[index_1]
            y_2, x_2 = keypoints_2[index_2]
            x_1, y_1 = float(x_1), float(y_1)
            x_2, y_2 = float(x_2) + image_1.width, float(y_2)
            draw.line((x_1, y_1, x_2, y_2), fill="yellow", width=1)
            for x, y in ((x_1, y_1), (x_2, y_2)):
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="red", width=1)
        return output

    def _make_match_preview(
        self, original_image: Image.Image, ref_image: Image.Image,
        source_keypoints: np.ndarray, reference_keypoints: np.ndarray,
        matches: np.ndarray, ref_path: str | Path,
    ) -> MatchPreview:
        output = self._draw_matches(
            ref_image, original_image, reference_keypoints, source_keypoints, matches[:, ::-1],
        )
        point_pairs = tuple(
            (
                (float(reference_keypoints[j, 1]), float(reference_keypoints[j, 0])),
                (float(source_keypoints[i, 1]), float(source_keypoints[i, 0])),
            )
            for i, j in matches
        )
        return MatchPreview(
            left_image=ref_image, right_image=original_image.copy(),
            point_pairs=point_pairs, export_image=output, reference_path=Path(ref_path),
        )
