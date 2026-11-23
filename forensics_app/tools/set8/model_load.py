"""Inspect a trained model's passport or classify the current image."""

from pathlib import Path
from tempfile import TemporaryDirectory
from tkinter import filedialog, scrolledtext
import json
import shutil
import tkinter as tk
import zipfile

import numpy as np

from ..base import ForensicsTool, ToolResult
from ..utilities import dialog_options


class ClassifyWithPretrainedModel(ForensicsTool):
    tool_id = "classifywithpretrainedmodel"
    title = "Classification with pretrained model."
    category = "Set8"
    description = "Load a pretrained model and use it to classify images."
    requires_image = False

    def __init__(self):
        self.options = ["Inspect passport", "Classify model"]
        self.model = None
        self.passport = None

    @staticmethod
    def _archive_model_name(archive):
        models = [name for name in archive.namelist()
                  if name.endswith(".keras") and "__MACOSX/" not in name]
        if len(models) != 1:
            raise ValueError("Select a ZIP containing exactly one .keras model and its passport JSON.")
        return models[0]

    def _load_passport(self, path):
        """Read metadata without importing TensorFlow or loading any weights."""
        path = Path(path)
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                model_name = self._archive_model_name(archive)
                passport_name = str(Path(model_name).with_suffix(".json"))
                if passport_name not in archive.namelist():
                    raise FileNotFoundError(f"Model passport missing from ZIP: {passport_name}")
                passport = json.loads(archive.read(passport_name))
        elif path.suffix.lower() in {".keras", ".json"}:
            passport_path = path.with_suffix(".json")
            if not passport_path.is_file():
                raise FileNotFoundError(f"Model passport not found: {passport_path}")
            passport = json.loads(passport_path.read_text(encoding="utf-8"))
        else:
            raise ValueError("Select a model ZIP, a .keras model, or its passport JSON.")

        if not isinstance(passport, dict):
            raise ValueError("The model passport must contain a JSON object.")
        classes = passport.get("classes")
        if (not isinstance(classes, list) or not classes
                or any(not isinstance(name, str) or not name for name in classes)
                or len(set(classes)) != len(classes)):
            raise ValueError("The passport must list the distinct class names in model output order.")
        preprocessing = passport.get("preprocessing", {})
        if not isinstance(preprocessing, dict):
            raise ValueError("The passport must describe the model preprocessing.")
        size = preprocessing.get("size")
        if (not isinstance(size, (list, tuple)) or len(size) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in size)):
            raise ValueError("The passport preprocessing size must be [height, width].")
        if preprocessing.get("color", "RGB") != "RGB":
            raise ValueError("Set 8 currently supports models with RGB input.")

        normalization = preprocessing.get("normalization")
        if normalization is None:
            instructions = str(preprocessing.get("instructions", "")).replace(" ", "")
            if "[-1,1]" in instructions:
                normalization = "minus_one_one"
            elif "[0,1]" in instructions:
                normalization = "zero_one"
        if normalization not in {"zero_one", "minus_one_one", "none"}:
            raise ValueError("Passport normalization must be zero_one, minus_one_one, or none.")
        preprocessing["normalization"] = normalization
        self.passport = passport
        return passport

    @staticmethod
    def _load_keras_model(model_path):
        try:
            from tensorflow.keras.models import load_model, model_from_json
        except ImportError as error:
            raise RuntimeError("Install the CNN dependencies with: python -m pip install '.[cnn]'") from error
        try:
            return load_model(model_path, compile=False)
        except TypeError as error:
            if "quantization_config" not in str(error):
                raise
            # Older Keras cannot read the empty quantization option saved by Colab.
            with zipfile.ZipFile(model_path) as archive:
                config = json.loads(archive.read("config.json"), object_hook=lambda d: {
                    key: value for key, value in d.items()
                    if key != "quantization_config" or value is not None
                })
            config.pop("compile_config", None)
            model = model_from_json(json.dumps(config))
            model.load_weights(model_path)
            return model

    def _validate_model(self):
        shape = self.model.input_shape
        if isinstance(shape, list) or len(shape) != 4 or shape[-1] != 3:
            raise ValueError("The loaded model must have a single RGB image input.")
        size = self.passport["preprocessing"]["size"]
        if any(actual is not None and actual != expected for actual, expected in zip(shape[1:3], size)):
            raise ValueError("Passport preprocessing size does not match the model input shape.")
        output_shape = self.model.output_shape
        if (isinstance(output_shape, list) or len(output_shape) != 2
                or output_shape[-1] != len(self.passport["classes"])):
            raise ValueError("Passport class count does not match the model output shape.")

    def _load_image(self, path: str):
        """Load a model bundle without extracting files beside the source ZIP."""
        path = Path(path)
        self._load_passport(path)
        if path.suffix.lower() == ".zip":
            with TemporaryDirectory(prefix="forensics-set8-") as directory:
                model_path = Path(directory) / "model.keras"
                with zipfile.ZipFile(path) as archive:
                    with archive.open(self._archive_model_name(archive)) as source, model_path.open("wb") as target:
                        shutil.copyfileobj(source, target)
                self.model = self._load_keras_model(model_path)
        else:
            self.model = self._load_keras_model(path.with_suffix(".keras"))
        self._validate_model()
        return self.model

    def _show_passport(self, passport, parent=None):
        text = json.dumps(passport, indent=2, ensure_ascii=False)
        print(text)
        if parent is not None:
            window = tk.Toplevel(parent)
            window.title("Model passport")
            window.geometry("720x600")
            viewer = scrolledtext.ScrolledText(window, wrap="word", padx=12, pady=12)
            viewer.pack(fill="both", expand=True)
            viewer.insert("1.0", text)
            viewer.configure(state="disabled")

    def _preprocess_image(self, image: np.ndarray):
        if self.passport is None:
            raise ValueError("Load a model passport before preparing images.")
        x = np.asarray(image, dtype=np.float32)
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError("Expected an RGB image with three channels.")
        preprocessing = self.passport["preprocessing"]
        size = tuple(preprocessing["size"])
        if x.shape[:2] != size:
            import tensorflow as tf
            x = tf.image.resize(x, size).numpy()
        normalization = preprocessing["normalization"]
        if normalization == "zero_one":
            x = x / 255.0
        elif normalization == "minus_one_one":
            x = x / 127.5 - 1.0
        elif normalization != "none":
            raise ValueError("Unknown passport normalization.")
        return x.astype(np.float32, copy=False)

    def _classify_image(self, image: np.ndarray):
        x = self._preprocess_image(image)
        probabilities = np.asarray(self.model.predict(x[None, ...], verbose=0))[0]
        if probabilities.shape != (len(self.passport["classes"]),) or not np.isfinite(probabilities).all():
            raise ValueError("The model returned invalid class probabilities.")
        index = int(np.argmax(probabilities))
        return self.passport["classes"][index], float(probabilities[index])

    def run(self, parent, document):
        option = dialog_options(parent, self.title, "Choose an option", self.options)
        if option is None:
            return None
        if option == self.options[1] and (document is None or not document.is_loaded):
            raise ValueError("Open an image before classifying it.")
        path = filedialog.askopenfilename(
            title="Select model or passport", parent=parent,
            filetypes=[("Model or passport", "*.zip *.keras *.json"), ("All files", "*.*")],
        )
        if not path:
            return None
        if option == self.options[0]:
            self._load_passport(path)
            self._show_passport(self.passport, parent)
            return ToolResult(message="Model passport displayed.", details=self.passport)

        self._load_image(path)
        image = np.asarray(document.current.convert("RGB"))
        label, probability = self._classify_image(image)
        return ToolResult(
            message=f"{label}: {probability:.1%}",
            details={"Prediction": label, "Probability": f"{probability:.1%}"},
        )
