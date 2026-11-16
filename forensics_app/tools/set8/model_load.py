"""Inspect a trained model's passport or classify the current image."""

from pathlib import Path
from tkinter import filedialog
import json
import zipfile

import numpy as np

from ..base import ForensicsTool, ToolResult
from ..utilities import dialog_options


class ClassifyWithPretrainedModel(ForensicsTool):
    tool_id = "classifywithpretrainedmodel"
    title = "Classification with pretrained model."
    category = "Set8"
    description = "Load a pretrained model and use it to classify images."
    requires_image = True

    def __init__(self):
        self.options = ["Inspect passport", "Classify model"]

    def _load_image(self, path: str):
        from tensorflow.keras.models import load_model, model_from_json

        folder = Path(path).with_suffix("")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(folder)
        model_path = next(folder.glob("*.keras"))
        try:
            self.model = load_model(model_path, compile=False)
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
            self.model = model_from_json(json.dumps(config))
            self.model.load_weights(model_path)
        with model_path.with_suffix(".json").open() as f:
            self.passport = json.load(f)

    def _show_passport(self, passport):
        print(json.dumps(passport, indent=2, ensure_ascii=False))

    def _classify_image(self, image: np.ndarray):
        import tensorflow as tf

        x = tf.image.resize(image.astype(np.float32), (128, 128)).numpy()
        x = x / 127.5 - 1
        probabilities = self.model.predict(x[None, ...], verbose=0)[0]
        index = int(np.argmax(probabilities))
        return self.passport["classes"][index], float(probabilities[index])

    def run(self, parent, document):
        option = dialog_options(parent, self.title, "Choose an option", self.options)
        if option is None:
            return None
        path = filedialog.askopenfilename(
            title="Select model ZIP", parent=parent, filetypes=[("Model ZIP", "*.zip")],
        )
        if not path:
            return None
        self._load_image(path)
        if option == self.options[0]:
            self._show_passport(self.passport)
            return ToolResult(message="Model passport displayed.", details=self.passport)

        image = np.asarray(document.current.convert("RGB"))
        label, probability = self._classify_image(image)
        return ToolResult(
            message=f"{label}: {probability:.1%}",
            details={"Prediction": label, "Probability": f"{probability:.1%}"},
        )
