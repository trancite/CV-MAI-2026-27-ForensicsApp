"""Convert one COCO dataset split into YOLO images, labels and class metadata."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from tkinter import filedialog

from PIL import Image

from ..base import ForensicsTool, ToolResult


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _integer_id(value, description):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{description} must be an integer.")
    return value


def _image_path(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(ord(character) < 32 for character in name)):
        raise ValueError(f"Unsafe COCO image filename: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsafe or unsupported COCO image filename: {name!r}")
    return Path(*path.parts)


def _normalized_box(bbox, width, height, description):
    if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in bbox)):
        raise ValueError(f"{description} must have a finite [x, y, width, height] bbox.")
    x, y, box_width, box_height = bbox
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"{description} bbox width and height must be positive.")
    # COCO boxes can cross image boundaries; YOLO coordinates must stay in range.
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(float(width), x + box_width), min(float(height), y + box_height)
    if right <= left or bottom <= top:
        raise ValueError(f"{description} bbox does not overlap its image.")
    normalized = ((left + right) / (2 * width), (top + bottom) / (2 * height),
                  (right - left) / width, (bottom - top) / height)
    clipped = (left, top, right, bottom) != (x, y, x + box_width, y + box_height)
    return normalized, clipped


def coco_to_yolo(input_folder, output_folder):
    """Convert ``_annotations.coco.json`` and its images for one dataset split.

    Class IDs are remapped in ascending COCO category-ID order. Unannotated
    container categories referenced as another category's supercategory are
    omitted; unused leaf classes remain to preserve class IDs between splits.
    Image-relative subdirectories are preserved, and every image receives a
    label file, even when it contains no objects. Partly out-of-frame boxes are clipped. The
    destination must be empty and separate from the source. Conventional source
    folder roles (train, valid/val/validation, test) are preserved; images are
    never repartitioned into additional splits.
    """
    source = Path(input_folder).expanduser().resolve()
    output = Path(output_folder).expanduser().resolve()
    split = {"train": "train", "valid": "val", "val": "val", "validation": "val",
             "test": "test"}.get(source.name.lower())
    if not source.is_dir():
        raise FileNotFoundError(f"COCO input folder not found: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Choose separate, non-overlapping input and output folders.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose an empty output folder; this destination contains data: {output}")
    annotations_path = source / "_annotations.coco.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"COCO annotations file not found: {annotations_path}")
    try:
        dataset = json.loads(annotations_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"Invalid COCO annotations JSON: {annotations_path}") from error
    if not isinstance(dataset, dict) or any(
            not isinstance(dataset.get(key), list) for key in ("images", "annotations", "categories")):
        raise ValueError("COCO JSON must contain images, annotations and categories lists.")
    if not dataset["images"] or not dataset["categories"]:
        raise ValueError("COCO JSON must contain at least one image and one category.")

    categories, supercategories = {}, {}
    for category in dataset["categories"]:
        if not isinstance(category, dict):
            raise ValueError("Each COCO category must be an object with id and name.")
        category_id = _integer_id(category.get("id"), "COCO category id")
        name = category.get("name")
        if category_id in categories or not isinstance(name, str) or not name.strip():
            raise ValueError("COCO categories must have unique IDs and non-empty names.")
        categories[category_id] = name
        supercategory = category.get("supercategory")
        if supercategory is not None and not isinstance(supercategory, str):
            raise ValueError("COCO category supercategory must be a name string.")
        supercategories[category_id] = supercategory
    if len(set(categories.values())) != len(categories):
        raise ValueError("COCO categories must have distinct class names.")

    annotated_ids = set()
    for index, annotation in enumerate(dataset["annotations"]):
        if not isinstance(annotation, dict):
            raise ValueError(f"COCO annotation {index + 1} must be an object.")
        category_id = _integer_id(annotation.get("category_id"), f"COCO annotation {index + 1} category_id")
        if category_id not in categories:
            raise ValueError(f"COCO annotation {index + 1} refers to an unknown category ID.")
        annotated_ids.add(category_id)
    omitted_parents = {
        category_id: name for category_id, name in categories.items()
        if category_id not in annotated_ids
        and any(other_id != category_id and parent_name == name
                for other_id, parent_name in supercategories.items())
    }
    ordered_ids = sorted(category_id for category_id in categories if category_id not in omitted_parents)
    class_ids = {category_id: index for index, category_id in enumerate(ordered_ids)}
    names = [categories[category_id] for category_id in ordered_ids]
    if not names:
        raise ValueError("COCO categories must include at least one concrete class.")

    images, image_paths, label_paths = {}, set(), set()
    for entry in dataset["images"]:
        if not isinstance(entry, dict):
            raise ValueError("Each COCO image must describe its id, filename, width and height.")
        image_id = _integer_id(entry.get("id"), "COCO image id")
        relative = _image_path(entry.get("file_name"))
        label_relative = relative.with_suffix(".txt")
        if image_id in images or relative in image_paths or label_relative in label_paths:
            raise ValueError("Duplicate COCO image IDs, image paths or YOLO label filenames.")
        width, height = entry.get("width"), entry.get("height")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in (width, height)):
            raise ValueError(f"COCO image {relative} needs positive integer width and height.")
        image_source = (source / relative).resolve()
        if source not in image_source.parents:
            raise ValueError(f"COCO image escapes the input folder: {relative}")
        if not image_source.is_file():
            raise FileNotFoundError(f"COCO image not found: {relative}")
        try:
            with Image.open(image_source) as image:
                if image.size != (width, height):
                    raise ValueError(f"COCO image dimensions do not match the file: {relative}")
                image.verify()
        except OSError as error:
            raise ValueError(f"Cannot read COCO image: {relative}") from error
        images[image_id] = {"path": relative, "source": image_source,
                            "width": width, "height": height, "labels": []}
        image_paths.add(relative)
        label_paths.add(label_relative)

    clipped_count = 0
    for index, annotation in enumerate(dataset["annotations"]):
        description = f"COCO annotation {index + 1}"
        if not isinstance(annotation, dict):
            raise ValueError(f"{description} must be an object.")
        image_id = _integer_id(annotation.get("image_id"), f"{description} image_id")
        category_id = _integer_id(annotation.get("category_id"), f"{description} category_id")
        if image_id not in images or category_id not in class_ids:
            raise ValueError(f"{description} refers to an unknown image or category ID.")
        image = images[image_id]
        box, clipped = _normalized_box(annotation.get("bbox"), image["width"], image["height"], description)
        image["labels"].append(f"{class_ids[category_id]} " + " ".join(f"{value:.10g}" for value in box))
        clipped_count += int(clipped)

    # Publish only a completely converted split. Existing content is never replaced.
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-coco-", dir=output.parent) as directory:
        staging = Path(directory)
        (staging / "images").mkdir()
        (staging / "labels").mkdir()
        for image in images.values():
            target = staging / "images" / image["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image["source"], target)
            label = (staging / "labels" / image["path"]).with_suffix(".txt")
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(image["labels"]) + ("\n" if image["labels"] else ""), encoding="utf-8")
        config = {"path": str(output), split or "train": "images", "nc": len(names), "names": names}
        # JSON syntax is valid YAML; quoted names retain punctuation and Unicode.
        comment = (f"# Converted {split} split. Keep independent train/validation/test data.\n"
                   if split else
                   "# Single converted split with unknown role; configure its role before training.\n")
        (staging / "data.yaml").write_text(
            comment + "# No additional dataset split is inferred or created.\n"
            + json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if output.exists():
            if not output.is_dir() or any(output.iterdir()):
                raise FileExistsError(f"Output folder now contains data: {output}")
            output.rmdir()
        staging.rename(output)

    return {"Images": len(images), "Annotations": len(dataset["annotations"]),
            "Classes": len(names), "Class names": names, "Clipped boxes": clipped_count,
            "Omitted parent categories": omitted_parents, "Category ID mapping": class_ids,
            "Split": split or "single", "Output": str(output), "Dataset config": str(output / "data.yaml")}


class COCOToYOLO(ForensicsTool):
    tool_id = "cocotoyolo"
    title = "Convert COCO dataset to YOLO"
    category = "Set9"
    description = "Convert one COCO split into YOLO images, labels and class metadata."
    requires_image = False

    def run(self, parent, document):
        source = filedialog.askdirectory(title="Select COCO split with _annotations.coco.json",
                                         parent=parent, mustexist=True)
        if not source:
            return None
        output = filedialog.askdirectory(title="Select an empty YOLO output folder", parent=parent)
        if not output:
            return None
        summary = coco_to_yolo(source, output)
        return ToolResult(message=f"Converted {summary['Images']} images and {summary['Annotations']} boxes to YOLO.",
                          details=summary)
