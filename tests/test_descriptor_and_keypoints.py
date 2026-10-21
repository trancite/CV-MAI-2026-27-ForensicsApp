"""Match geometry and shared settings, with real ORB/SIFT extraction."""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.set5.descriptor_and_keypoints import DescriptorAndKeypointsTool


MODULE = "forensics_app.tools.set5.descriptor_and_keypoints"


class DescriptorAndKeypointsTests(unittest.TestCase):
    def setUp(self):
        self.tool = DescriptorAndKeypointsTool()
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.reference = Path(self.folder.name) / "reference.png"
        self.pixels = np.random.default_rng(42).integers(0, 256, (128, 192), dtype=np.uint8)
        self.similar_pixels = self.pixels.copy()
        self.similar_pixels[0, 0] ^= 1
        Image.fromarray(self.pixels).save(self.reference)
        self.document = ImageDocument()
        self.document.load(self.reference)

    def test_matching_prompts_once_and_preserves_working_image(self):
        for algorithm, integers, integer_count, float_count, scoring in (
            ("ORB", [100], 1, 0, "Default"),
            ("ORB", [100], 1, 0, "RANSAC"),
            ("SIFT", [2, 3], 2, 1, "Default"),
            ("SIFT", [2, 3], 2, 1, "RANSAC"),
        ):
            with self.subTest(algorithm=algorithm, scoring=scoring), \
                 patch(f"{MODULE}.dialog_options", side_effect=[algorithm, self.tool.options[1], scoring]), \
                 patch(f"{MODULE}.filedialog.askopenfilename", return_value=str(self.reference)), \
                 patch(f"{MODULE}.simpledialog.askinteger", side_effect=integers) as askinteger, \
                 patch(f"{MODULE}.simpledialog.askfloat", return_value=1.6) as askfloat:
                result = self.tool.run(None, self.document)
            self.assertEqual(askinteger.call_count, integer_count)
            self.assertEqual(askfloat.call_count, float_count)
            self.assertIsNone(result.image)
            self.assertGreater(result.details["Matches"], 0)
            self.assertEqual(result.details["Matching score"], 1.0)
            self.assertEqual(result.details["Scoring"], scoring)
            preview = result.match_preview
            self.assertEqual(len(preview.point_pairs), result.details["Matches"])
            self.assertEqual(preview.export_image.size, (384, 128))
            self.assertEqual(preview.reference_path, self.reference)
            for left_point, right_point in preview.point_pairs:
                self.assertEqual(left_point, right_point)
            np.testing.assert_array_equal(np.asarray(self.document.current), self.pixels)
            self.assertFalse(self.document.can_undo)

    def test_cancelling_shared_parameters_skips_both_extractions(self):
        for algorithm, integers in (("ORB", [None]), ("SIFT", [None]), ("SIFT", [2, None]), ("SIFT", [2, 3])):
            with self.subTest(algorithm=algorithm, integers=integers), \
                 patch(f"{MODULE}.dialog_options", side_effect=[algorithm, self.tool.options[1], "Default"]), \
                 patch(f"{MODULE}.filedialog.askopenfilename", return_value=str(self.reference)), \
                 patch(f"{MODULE}.simpledialog.askinteger", side_effect=integers), \
                 patch(f"{MODULE}.simpledialog.askfloat", return_value=None), \
                 patch.object(self.tool, "_generate_keypoints_and_descriptors") as extract:
                self.assertIsNone(self.tool.run(None, self.document))
            extract.assert_not_called()

    def test_export_connects_selected_points_with_reference_offset(self):
        first = Image.new("RGB", (12, 8), "black")
        second = Image.new("RGB", (9, 11), "blue")
        points_1 = np.array([[5., 7.], [3., 2.]])
        points_2 = np.array([[3., 4.], [8., 6.]])
        before = first.tobytes(), second.tobytes()
        output = self.tool._draw_matches(first, second, points_1, points_2, np.array([[1, 0]]))
        self.assertEqual(output.size, (21, 11))
        for point in ((2, 3), (9, 3), (16, 3)):
            self.assertEqual(output.getpixel(point), (255, 255, 0))
        self.assertEqual(output.getpixel((7, 5)), (0, 0, 0))
        self.assertEqual(output.getpixel((18, 8)), (0, 0, 255))
        self.assertEqual((first.tobytes(), second.tobytes()), before)

    def test_preview_uses_reference_points_on_left_and_source_points_on_right(self):
        source_points = np.array([[8., 9.], [3., 2.]])
        reference_points = np.array([[6., 5.], [10., 11.]])
        descriptors = np.ones((2, 8), dtype=bool)
        with patch(f"{MODULE}.dialog_options", side_effect=["ORB", self.tool.options[1], "Default"]), \
             patch(f"{MODULE}.filedialog.askopenfilename", return_value=str(self.reference)), \
             patch(f"{MODULE}.simpledialog.askinteger", return_value=2), \
             patch.object(self.tool, "_generate_keypoints_and_descriptors", side_effect=[
                 (source_points, descriptors), (reference_points, descriptors),
             ]), \
             patch.object(self.tool, "_compare_descriptors", return_value=(np.array([[1, 0]]), 0.5)):
            result = self.tool.run(None, self.document)
        self.assertEqual(result.match_preview.point_pairs, (((5., 6.), (2., 3.)),))

    def test_find_closest_searches_only_selected_folder_with_shared_settings(self):
        selected = Path(self.folder.name) / "selected"
        selected.mkdir()
        closest = selected / "closest.PNG"
        Image.fromarray(self.similar_pixels).save(closest)
        other = np.random.default_rng(99).integers(0, 256, self.pixels.shape, dtype=np.uint8)
        Image.fromarray(other).save(selected / "other.png")
        Image.new("L", (192, 128), 50).save(selected / "blank.png")
        (selected / "broken.png").write_bytes(b"not an image")
        (selected / "notes.txt").write_text("Not a candidate.")
        nested = selected / "a_subfolder"
        nested.mkdir()
        Image.fromarray(self.similar_pixels).save(nested / "ignored.png")
        Image.fromarray(self.similar_pixels).save(Path(self.folder.name) / "outside.png")
        for algorithm, integers, scoring in (
            ("ORB", [100], "Default"), ("ORB", [100], "RANSAC"),
            ("SIFT", [2, 3], "Default"), ("SIFT", [2, 3], "RANSAC"),
        ):
            with self.subTest(algorithm=algorithm, scoring=scoring), \
                 patch(f"{MODULE}.dialog_options", side_effect=[algorithm, self.tool.options[2], scoring]), \
                 patch(f"{MODULE}.filedialog.askdirectory", return_value=str(selected)) as directory, \
                 patch(f"{MODULE}.simpledialog.askinteger", side_effect=integers) as askinteger, \
                 patch(f"{MODULE}.simpledialog.askfloat", return_value=1.6) as askfloat:
                result = self.tool.run(None, self.document)
            self.assertEqual(askinteger.call_count, len(integers))
            self.assertEqual(askfloat.call_count, int(algorithm == "SIFT"))
            self.assertEqual(directory.call_args.kwargs["initialdir"], str(self.reference.parent))
            self.assertEqual(result.match_preview.reference_path, closest)
            self.assertEqual(result.details["Matching score"], 1.0)
            self.assertIsNone(result.image)
            np.testing.assert_array_equal(np.asarray(self.document.current), self.pixels)
            self.assertFalse(self.document.can_undo)

    def test_find_closest_ranks_by_score_then_match_count_then_path(self):
        candidates = Path(self.folder.name) / "candidates"
        candidates.mkdir()
        for name in ("d.png", "c.png", "b.png", "a.png"):
            Image.fromarray(self.similar_pixels).save(candidates / name)
        points = np.array([[20., 30.], [30., 40.], [40., 50.], [50., 60.]])
        descriptors = np.ones((4, 8), dtype=bool)
        matches = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        with patch(f"{MODULE}.dialog_options", side_effect=["ORB", self.tool.options[2], "Default"]), \
             patch(f"{MODULE}.filedialog.askdirectory", return_value=str(candidates)), \
             patch.object(self.tool, "_choose_parameters", return_value={"n_keypoints": 100}) as choose, \
             patch.object(self.tool, "_generate_keypoints_and_descriptors", return_value=(points, descriptors)) as extract, \
             patch.object(self.tool, "_compare_descriptors", side_effect=[
                 (matches[:3], .75), (matches, .5), (matches, .75), (matches, .75),
             ]):
            result = self.tool.run(None, self.document)
        choose.assert_called_once()
        self.assertEqual(extract.call_count, 5)  # Source extracted once plus four candidates.
        self.assertEqual(result.match_preview.reference_path, candidates / "c.png")
        self.assertEqual(result.details["Matches"], 4)
        self.assertEqual(result.details["Matching score"], .75)

    def test_find_closest_cancellation_stops_before_extraction(self):
        candidates = Path(self.folder.name) / "candidates"
        candidates.mkdir()
        Image.fromarray(self.pixels).save(candidates / "image.png")
        for parameters, folder in ((None, str(candidates)), ({"n_keypoints": 100}, "")):
            with self.subTest(parameters=parameters, folder=folder), \
                 patch(f"{MODULE}.dialog_options", side_effect=["ORB", self.tool.options[2], "Default"]), \
                 patch(f"{MODULE}.filedialog.askdirectory", return_value=folder) as directory, \
                 patch.object(self.tool, "_choose_parameters", return_value=parameters) as choose, \
                 patch.object(self.tool, "_generate_keypoints_and_descriptors") as extract:
                self.assertIsNone(self.tool.run(None, self.document))
            choose.assert_called_once()
            self.assertEqual(directory.call_count, int(parameters is not None))
            extract.assert_not_called()

    def test_find_closest_excludes_the_current_file_and_reports_empty_folder(self):
        with patch(f"{MODULE}.dialog_options", side_effect=["ORB", self.tool.options[2], "Default"]), \
             patch(f"{MODULE}.filedialog.askdirectory", return_value=self.folder.name), \
             patch.object(self.tool, "_choose_parameters", return_value={"n_keypoints": 100}) as choose:
            result = self.tool.run(None, self.document)
        self.assertIsNone(result.match_preview)
        self.assertIn("No other supported images", result.message)
        choose.assert_called_once()

    def test_find_closest_excludes_copies_but_keeps_similar_images(self):
        Image.fromarray(self.pixels).save(Path(self.folder.name) / "copy.png")
        Image.fromarray(self.pixels).convert("RGB").save(Path(self.folder.name) / "copy.bmp")
        closest = Path(self.folder.name) / "similar.png"
        Image.fromarray(self.similar_pixels).save(closest)
        for source_path in (self.reference, None):
            self.document.path = source_path
            with self.subTest(source_path=source_path), \
                 patch(f"{MODULE}.dialog_options", side_effect=["ORB", self.tool.options[2], "Default"]), \
                 patch(f"{MODULE}.filedialog.askdirectory", return_value=self.folder.name), \
                 patch(f"{MODULE}.simpledialog.askinteger", return_value=100), \
                 patch.object(self.tool, "_generate_keypoints_and_descriptors", wraps=self.tool._generate_keypoints_and_descriptors) as extract:
                result = self.tool.run(None, self.document)
            self.assertEqual(result.match_preview.reference_path, closest)
            self.assertEqual(extract.call_count, 2)  # Source and distinct candidate only.
            self.assertEqual(result.details["Matching score"], 1.0)

    def test_find_closest_reports_unusable_images_or_zero_matches(self):
        candidates = Path(self.folder.name) / "candidates"
        candidates.mkdir()
        image_path = candidates / "candidate.png"
        points = np.array([[20., 30.], [40., 50.]])
        descriptors = np.ones((2, 8), dtype=bool)
        for corrupt in (True, False):
            if corrupt:
                image_path.write_bytes(b"invalid image")
            else:
                Image.fromarray(self.similar_pixels).save(image_path)
            with self.subTest(corrupt=corrupt), \
                 patch(f"{MODULE}.dialog_options", side_effect=["ORB", self.tool.options[2], "Default"]), \
                 patch(f"{MODULE}.filedialog.askdirectory", return_value=str(candidates)), \
                 patch.object(self.tool, "_choose_parameters", return_value={"n_keypoints": 100}), \
                 patch.object(self.tool, "_generate_keypoints_and_descriptors", return_value=(points, descriptors)), \
                 patch.object(self.tool, "_compare_descriptors", return_value=(np.empty((0, 2), dtype=int), 0.0)):
                result = self.tool.run(None, self.document)
            self.assertIsNone(result.match_preview)
            self.assertIn("No matching", result.message)

    def test_comparing_empty_descriptors_has_zero_score(self):
        for first, second in ((0, 2), (2, 0), (0, 0)):
            matches, score = self.tool._compare_descriptors(np.empty((first, 8)), np.empty((second, 8)))
            self.assertEqual(matches.shape, (0, 2))
            self.assertEqual(score, 0.0)

    def test_ransac_removes_geometric_outliers_from_matches_and_score(self):
        rng = np.random.default_rng(7)
        source_xy = rng.uniform(20, 200, (30, 2))
        # A projective mapping with 22 correct correspondences and 8 outliers.
        matrix = np.array([[1.1, .1, 12], [-.05, .9, 8], [.0005, .0002, 1]])
        homogeneous = np.column_stack((source_xy, np.ones(30))) @ matrix.T
        reference_xy = homogeneous[:, :2] / homogeneous[:, 2:]
        reference_xy[22:] = rng.uniform(300, 500, (8, 2))
        descriptors = np.eye(30)
        matches, score = self.tool._compare_descriptors(
            descriptors, descriptors, keypoints=(source_xy[:, ::-1], reference_xy[:, ::-1]),
        )
        np.testing.assert_array_equal(matches, np.column_stack((np.arange(22), np.arange(22))))
        self.assertAlmostEqual(score, 22 / 30)

    def test_ransac_needs_four_matches_and_handles_failed_estimation(self):
        descriptors = np.eye(3)
        points = np.zeros((3, 2))
        with patch(f"{MODULE}.ransac") as estimate:
            matches, score = self.tool._compare_descriptors(descriptors, descriptors, keypoints=(points, points))
        self.assertEqual(matches.shape, (0, 2))
        self.assertEqual(score, 0.0)
        estimate.assert_not_called()
        with patch(f"{MODULE}.ransac", return_value=(None, None)):
            matches, score = self.tool._compare_descriptors(
                np.eye(4), np.eye(4), keypoints=(np.zeros((4, 2)), np.zeros((4, 2))),
            )
        self.assertEqual(matches.shape, (0, 2))
        self.assertEqual(score, 0.0)

    def test_cancelling_ransac_selection_stops_before_opening_folder(self):
        with patch(f"{MODULE}.dialog_options", side_effect=["ORB", self.tool.options[2], None]), \
             patch(f"{MODULE}.simpledialog.askinteger", return_value=100) as parameters, \
             patch(f"{MODULE}.filedialog.askdirectory") as directory, \
             patch.object(self.tool, "_generate_keypoints_and_descriptors") as extract:
            self.assertIsNone(self.tool.run(None, self.document))
        parameters.assert_called_once()
        directory.assert_not_called()
        extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
