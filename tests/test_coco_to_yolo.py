"""COCO conversion must preserve images and produce aligned YOLO labels."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.set9.coco_to_yolo import COCOToYOLO, coco_to_yolo


class COCOToYOLOTests(unittest.TestCase):
    module = "forensics_app.tools.set9.coco_to_yolo"

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "coco"
        self.output = self.root / "yolo"
        self.source.mkdir()
        (self.source / "nested").mkdir()
        Image.new("RGB", (100, 50), "red").save(self.source / "blood.png")
        Image.new("RGB", (100, 50), "white").save(self.source / "nested" / "negative.jpg")
        self.dataset = {
            "categories": [{"id": 42, "name": "RBC"}, {"id": 7, "name": "PC"}],
            "images": [
                {"id": 10, "file_name": "blood.png", "width": 100, "height": 50},
                {"id": 20, "file_name": "nested/negative.jpg", "width": 100, "height": 50},
            ],
            "annotations": [
                {"id": 1, "image_id": 10, "category_id": 42, "bbox": [10, 5, 40, 20]},
                {"id": 2, "image_id": 10, "category_id": 7, "bbox": [0, 0, 100, 50]},
            ],
        }
        self._write_annotations(self.dataset)

    def _write_annotations(self, dataset):
        (self.source / "_annotations.coco.json").write_text(json.dumps(dataset), encoding="utf-8")

    def _config(self, output=None):
        text = ((output or self.output) / "data.yaml").read_text(encoding="utf-8")
        return json.loads("\n".join(line for line in text.splitlines() if not line.startswith("#")))

    def test_maps_noncontiguous_categories_and_converts_xywh_to_normalized_centers(self):
        summary = coco_to_yolo(self.source, self.output)
        lines = (self.output / "labels" / "blood.txt").read_text().splitlines()
        self.assertEqual([int(line.split()[0]) for line in lines], [1, 0])
        self.assertEqual([float(value) for value in lines[0].split()[1:]], [0.3, 0.3, 0.4, 0.4])
        self.assertEqual([float(value) for value in lines[1].split()[1:]], [0.5, 0.5, 1.0, 1.0])
        self.assertEqual(summary["Images"], 2)
        self.assertEqual(summary["Annotations"], 2)
        self.assertEqual(summary["Class names"], ["PC", "RBC"])
        self.assertEqual(summary["Clipped boxes"], 0)

    def test_copies_images_unchanged_and_creates_empty_labels_for_background_images(self):
        original_json = (self.source / "_annotations.coco.json").read_bytes()
        originals = {name: (self.source / name).read_bytes()
                     for name in ("blood.png", "nested/negative.jpg")}
        coco_to_yolo(self.source, self.output)
        for name, contents in originals.items():
            self.assertEqual((self.output / "images" / name).read_bytes(), contents)
            self.assertEqual((self.source / name).read_bytes(), contents)
        self.assertEqual((self.output / "labels" / "nested" / "negative.txt").read_text(), "")
        self.assertEqual((self.source / "_annotations.coco.json").read_bytes(), original_json)

    def test_omits_unannotated_supercategory_and_keeps_unused_leaf_classes(self):
        self.dataset["categories"] = [
            {"id": 0, "name": "1-2-3-SmhR", "supercategory": "none"},
            {"id": 1, "name": "PC", "supercategory": "1-2-3-SmhR"},
            {"id": 2, "name": "RBC", "supercategory": "1-2-3-SmhR"},
            {"id": 3, "name": "WBC", "supercategory": "1-2-3-SmhR"},
        ]
        self.dataset["annotations"] = [
            {"image_id": 10, "category_id": 2, "bbox": [10, 5, 40, 20]},
        ]
        self._write_annotations(self.dataset)
        summary = coco_to_yolo(self.source, self.output)
        self.assertEqual(self._config()["names"], ["PC", "RBC", "WBC"])
        self.assertEqual(summary["Category ID mapping"], {1: 0, 2: 1, 3: 2})
        self.assertEqual(summary["Omitted parent categories"], {0: "1-2-3-SmhR"})
        self.assertEqual((self.output / "labels" / "blood.txt").read_text().split()[0], "1")

    def test_retains_supercategory_when_it_has_real_annotations(self):
        self.dataset["categories"] = [
            {"id": 0, "name": "Blood cell"},
            {"id": 7, "name": "PC", "supercategory": "Blood cell"},
            {"id": 42, "name": "RBC", "supercategory": "Blood cell"},
        ]
        self.dataset["annotations"].append(
            {"image_id": 10, "category_id": 0, "bbox": [0, 0, 10, 10]}
        )
        self._write_annotations(self.dataset)
        summary = coco_to_yolo(self.source, self.output)
        self.assertEqual(summary["Class names"], ["Blood cell", "PC", "RBC"])
        self.assertEqual(summary["Category ID mapping"], {0: 0, 7: 1, 42: 2})
        self.assertEqual(summary["Omitted parent categories"], {})
        self.assertEqual((self.output / "labels" / "blood.txt").read_text().splitlines()[-1].split()[0], "0")

    def test_retains_unused_standalone_category_and_self_referencing_category(self):
        self.dataset["categories"].extend([
            {"id": 50, "name": "WBC"},
            {"id": 60, "name": "Other", "supercategory": "Other"},
        ])
        self._write_annotations(self.dataset)
        summary = coco_to_yolo(self.source, self.output)
        self.assertEqual(summary["Class names"], ["PC", "RBC", "WBC", "Other"])
        self.assertEqual(summary["Omitted parent categories"], {})

    def test_clips_partial_boxes_before_normalization(self):
        self.dataset["annotations"] = [
            {"image_id": 10, "category_id": 7, "bbox": [-10, -5, 40, 20]},
        ]
        self._write_annotations(self.dataset)
        summary = coco_to_yolo(self.source, self.output)
        values = (self.output / "labels" / "blood.txt").read_text().split()
        self.assertEqual(values[0], "0")
        self.assertEqual([float(value) for value in values[1:]], [0.15, 0.15, 0.3, 0.3])
        self.assertEqual(summary["Clipped boxes"], 1)

    def test_dataset_yaml_preserves_quoted_names_without_inventing_validation_split(self):
        self.dataset["categories"][0]["name"] = 'Glóbulo: "rojo"\nsegunda línea'
        self._write_annotations(self.dataset)
        coco_to_yolo(self.source, self.output)
        config = self._config()
        self.assertEqual(config["path"], str(self.output))
        self.assertEqual(config["train"], "images")
        self.assertEqual(config["names"], ["PC", 'Glóbulo: "rojo"\nsegunda línea'])
        self.assertEqual(config["nc"], 2)
        self.assertNotIn("val", config)
        self.assertNotIn("test", config)

    def test_accepts_an_existing_empty_output_folder(self):
        self.output.mkdir()
        coco_to_yolo(self.source, self.output)
        self.assertTrue((self.output / "data.yaml").is_file())

    def test_preserves_test_split_without_training_or_validation_paths(self):
        test_source = self.root / "test"
        self.source.rename(test_source)
        summary = coco_to_yolo(test_source, self.output)
        config = self._config()
        self.assertEqual(config["test"], "images")
        self.assertNotIn("train", config)
        self.assertNotIn("val", config)
        self.assertEqual(summary["Split"], "test")

    def test_preserves_training_role_and_validation_folder_aliases(self):
        for folder, role in (("train", "train"), ("valid", "val"),
                             ("val", "val"), ("validation", "val")):
            with self.subTest(folder=folder):
                source = self.root / folder
                output = self.root / f"yolo_{folder}"
                shutil.copytree(self.source, source)
                summary = coco_to_yolo(source, output)
                config = self._config(output)
                self.assertEqual(config[role], "images")
                self.assertEqual(summary["Split"], role)
                for other in {"train", "val", "test"} - {role}:
                    self.assertNotIn(other, config)

    def test_rejects_existing_output_content_without_changing_it(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("preserve me")
        with self.assertRaisesRegex(FileExistsError, "(?i)empty|contains"):
            coco_to_yolo(self.source, self.output)
        self.assertEqual(sentinel.read_text(), "preserve me")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_rejects_overlapping_source_and_output_folders(self):
        for output in (self.source, self.source / "export", self.root):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "(?i)overlap|separate"):
                coco_to_yolo(self.source, output)
        self.assertFalse((self.source / "export").exists())

    def test_rejects_unsafe_image_paths_before_creating_output(self):
        for name in ("../outside.png", "/tmp/outside.png", "nested/../../outside.png",
                     "..\\outside.png", "C:\\outside.png"):
            with self.subTest(name=name):
                dataset = deepcopy(self.dataset)
                dataset["images"][0]["file_name"] = name
                self._write_annotations(dataset)
                with self.assertRaisesRegex(ValueError, "(?i)unsafe|escapes"):
                    coco_to_yolo(self.source, self.output)
                self.assertFalse(self.output.exists())

    def test_rejects_invalid_boxes_without_partial_output(self):
        for bbox in ([0, 0, 0, 10], [0, 0, -1, 10], [1, 2, 3],
                     [0, 0, float("inf"), 5], [float("nan"), 0, 5, 5],
                     [200, 0, 10, 10], [0, 0, True, 10]):
            with self.subTest(bbox=bbox):
                dataset = deepcopy(self.dataset)
                dataset["annotations"][0]["bbox"] = bbox
                self._write_annotations(dataset)
                with self.assertRaisesRegex(ValueError, "(?i)bbox"):
                    coco_to_yolo(self.source, self.output)
                self.assertFalse(self.output.exists())

    def test_rejects_annotations_referring_to_unknown_image_or_category(self):
        for key in ("image_id", "category_id"):
            with self.subTest(key=key):
                dataset = deepcopy(self.dataset)
                dataset["annotations"][0][key] = 999
                self._write_annotations(dataset)
                with self.assertRaisesRegex(ValueError, "(?i)unknown"):
                    coco_to_yolo(self.source, self.output)
                self.assertFalse(self.output.exists())

    def test_rejects_colliding_label_filenames(self):
        Image.new("RGB", (100, 50)).save(self.source / "blood.jpg")
        self.dataset["images"].append(
            {"id": 30, "file_name": "blood.jpg", "width": 100, "height": 50}
        )
        self._write_annotations(self.dataset)
        with self.assertRaisesRegex(ValueError, "(?i)duplicate|label"):
            coco_to_yolo(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_incorrect_image_dimensions_or_missing_files(self):
        self.dataset["images"][0]["width"] = 101
        self._write_annotations(self.dataset)
        with self.assertRaisesRegex(ValueError, "(?i)dimensions"):
            coco_to_yolo(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.dataset["images"][0]["width"] = 100
        self._write_annotations(self.dataset)
        (self.source / "blood.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "(?i)image"):
            coco_to_yolo(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_ui_reports_conversion_without_requiring_an_open_image(self):
        tool = COCOToYOLO()
        self.assertFalse(tool.requires_image)
        self.assertEqual(tool.tool_id, "cocotoyolo")
        self.assertEqual(tool.category, "Set9")
        with patch(f"{self.module}.filedialog.askdirectory", side_effect=[str(self.source), str(self.output)]):
            result = tool.run(None, ImageDocument())
        self.assertEqual(result.details["Images"], 2)
        self.assertEqual(result.details["Split"], "single")
        self.assertIsNone(result.image)

    def test_cancelling_either_folder_dialog_does_not_convert(self):
        for folders in ([""], [str(self.source), ""]):
            with self.subTest(folders=folders), \
                 patch(f"{self.module}.filedialog.askdirectory", side_effect=folders), \
                 patch(f"{self.module}.coco_to_yolo") as convert:
                self.assertIsNone(COCOToYOLO().run(None, ImageDocument()))
            convert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
