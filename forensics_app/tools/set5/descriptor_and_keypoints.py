from __future__ import annotations
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage import feature
from skimage.feature import hog
from forensics_app.core import ImageDocument
from ..base import ForensicsTool, MatchPreview, ToolResult
from ..utilities import dialog_options
import numpy as np
from tkinter import filedialog, simpledialog
from skimage.feature import SIFT, ORB
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from skimage.color import gray2rgb
from skimage.feature import match_descriptors

class DescriptorAndKeypointsTool(ForensicsTool):
    def __init__(self):
        self.tool_id = "descriptor_and_keypoints"
        self.title = "Descriptor and Keypoints detection"
        self.category = "Set5"
        self.description = "Detect keypoints and descriptors."
        self.requires_image = True
        self.options = ["Highlight keypoints of individual image", "Matching between two images", "Find closest image"]
        self.matching_scores = ["Default", "RANSAC"]
        self.algorithms = ["ORB", "SIFT"]


    def _choose_action(self, parent: tk.Misc):
        return dialog_options(parent, "Tool selection", "Choose what do you want to do", self.options)

    def _choose_algorithm(self, parent: tk.Misc):
        return dialog_options(parent, "Algorithm selection", "Choose between ORB or SIFT", self.algorithms)

    def _choose_n_keypoitns(self, parent: tk.Misc):
        return simpledialog.askinteger("Number of keypoints", "Choose the number of keypoints to consider for the ORB algorithm", minvalue = 1, parent = parent)

    def _choose_sift_parameters(self, parent: tk.Misc):
        n_octaves = simpledialog.askinteger("Number of octaves", "Choose the number of octaves for the SIFT algorithm", minvalue = 1, parent = parent)
        if n_octaves is None: return None
        
        n_scales  = simpledialog.askinteger("Number of scales", "Choose the number of scales for the SIFT algorithm", minvalue = 1, parent = parent)
        if n_scales is None: return None

        sigma = simpledialog.askfloat(
            "Sigma", "Choose the initial Gaussian blur sigma for SIFT (at least 1.0)",
            initialvalue=1.6, minvalue=1.0, parent=parent,
        )
        if sigma is None: return None

        return (n_octaves, n_scales, sigma)

    def _choose_parameters(self, parent: tk.Misc, algorithm: str) -> dict | None:
        if algorithm == self.algorithms[0]:
            n_keypoints = self._choose_n_keypoitns(parent)
            if n_keypoints is None: return None
            return {"n_keypoints": n_keypoints}
        sift_parameters = self._choose_sift_parameters(parent)
        if sift_parameters is None: return None
        n_octaves, n_scales, sigma = sift_parameters
        return {"n_octaves": n_octaves, "n_scales": n_scales, "sigma_min": sigma}

    def _generate_keypoints_and_descriptors(self, image_array: np.ndarray, algorithm: str, parameters: dict) -> tuple:
        if algorithm == self.algorithms[0]:
            extractor = ORB(**parameters)
        else:
            sigma = parameters.get("sigma_min", 1.6)
            if not np.isfinite(sigma) or sigma < 1.0:
                raise ValueError("SIFT sigma must be finite and at least 1.0.")
            extractor = SIFT(**parameters)
        try:
            extractor.detect_and_extract(image_array)
        except RuntimeError as error:
            if not str(error).startswith(f"{algorithm} found no features"):
                raise
            size, dtype = (256, bool) if algorithm == "ORB" else (128, np.uint8)
            return np.empty((0, 2), dtype=float), np.empty((0, size), dtype=dtype)
        return extractor.keypoints, extractor.descriptors


    def _compare_descriptors(self, descriptors_1: np.array, descriptors_2: np.array, metric: str = "euclidean", keypoints: tuple | None = None) -> tuple:
        if len(descriptors_1) == 0 or len(descriptors_2) == 0:
            return np.empty((0, 2), dtype=int), 0.0
        matches = match_descriptors(
            descriptors_1, descriptors_2,
            metric=metric,
            cross_check=True,
            max_ratio=0.8
            )

        if keypoints is not None:
            if len(matches) < 4:
                return matches[:0], 0.0
            coordinates = tuple(points[matches[:, i]][:, ::-1] for i, points in enumerate(keypoints))
            _, inliers = ransac(
                coordinates, ProjectiveTransform, min_samples=4, residual_threshold=3, max_trials=100, rng=0,
            )
            matches = matches[inliers] if inliers is not None else matches[:0]
        score = len(matches) / (min(len(descriptors_1), len(descriptors_2)))
        return matches, score

    def _draw_matches(
        self,
        image_1: Image.Image,
        image_2: Image.Image,
        keypoints_1: np.ndarray,
        keypoints_2: np.ndarray,
        matches: np.ndarray,
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
        self,
        original_image: Image.Image,
        ref_image: Image.Image,
        source_keypoints: np.ndarray,
        reference_keypoints: np.ndarray,
        matches: np.ndarray,
        ref_path: str | Path,
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
            left_image=ref_image,
            right_image=original_image.copy(),
            point_pairs=point_pairs,
            export_image=output,
            reference_path=Path(ref_path),
        )

    def _find_closest_image(
        self, parent: tk.Misc, document: ImageDocument, algorithm: str, parameters: dict, scoring: str,
    ) -> ToolResult | None:
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
        original_image = document.current
        source_pixels = np.asarray(original_image.convert("RGB"))
        source_keypoints, source_descriptors = self._generate_keypoints_and_descriptors(
            np.asarray(original_image.convert("L")), algorithm, parameters,
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
                matches, score = self._compare_descriptors(source_descriptors, descriptors, metric=metric, keypoints=points)
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
            details={"Method": algorithm, "Scoring": scoring, "Closest image": str(path), "Matching score": best_rank[0], "Matches": len(matches)},
            match_preview=self._make_match_preview(
                original_image, candidate_image, source_keypoints, keypoints, matches, path,
            ),
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        original_image = document.current
        src_image = np.asarray(original_image.convert("L"))
        

        algorithm = self._choose_algorithm(parent)
        if algorithm is None: return None
        tool = self._choose_action(parent)
        if tool is None: return None
        parameters = self._choose_parameters(parent, algorithm)
        if parameters is None: return None
        scoring = "Default"
        if tool in self.options[1:]:
            scoring = dialog_options(parent, "Matching score", "Choose how to score matches", self.matching_scores)
            if scoring is None: return None

        if tool == self.options[0]:
            keypoints, _ = self._generate_keypoints_and_descriptors(src_image, algorithm, parameters)
            if len(keypoints) == 0:
                return ToolResult(
                    message="No keypoints found in the working image.",
                    details={"Method": algorithm, "Number of keypoints": 0},
                )
            rgb_image = gray2rgb(src_image)
            coords = np.round(keypoints).astype(int)
            rgb_image[coords[:, 0], coords[:, 1]] = [255, 0, 0]
            new_image = Image.fromarray(rgb_image)
            return ToolResult(
                        image=new_image,
                        message="Keypoints detected",
                        details={"Method": algorithm, "Number of keypoints": len(keypoints)},
                    )

        elif tool == self.options[1]:
            ref_path = filedialog.askopenfilename(
                        title="Select a reference image.",
                        filetypes=[
                            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                            ("All files", "*.*"),
                        ],
                        parent=parent,
                    )
            if not ref_path: return None
    
            with Image.open(ref_path) as ref_file:
                ref_image = ref_file.convert("RGB")
            ref_gray = np.asarray(ref_image.convert("L"))
            source_features = self._generate_keypoints_and_descriptors(src_image, algorithm, parameters)
            reference_features = self._generate_keypoints_and_descriptors(ref_gray, algorithm, parameters)
            keypoints_1, descriptors_1 = source_features
            keypoints_2, descriptors_2 = reference_features
            metric = "hamming" if algorithm == "ORB" else "euclidean"
            points = (keypoints_1, keypoints_2) if scoring == "RANSAC" else None
            matches, score = self._compare_descriptors(descriptors_1, descriptors_2, metric=metric, keypoints=points)
            return ToolResult(
                message=f"Found {len(matches)} keypoint matches.",
                details={"Method": algorithm, "Scoring": scoring, "Matches": len(matches), "Matching score": score},
                match_preview=self._make_match_preview(
                    original_image, ref_image, keypoints_1, keypoints_2, matches, ref_path,
                ),
            )

        elif tool == self.options[2]:
            return self._find_closest_image(parent, document, algorithm, parameters, scoring)
