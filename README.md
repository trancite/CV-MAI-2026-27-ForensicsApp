# ForensicsApp course starter

ForensicsApp is the semester project shell for the MSc Computer Vision course.  
Students add a small, demonstrable image-forensics feature as the course moves  
from classical image processing to deep learning and interpretability.

The starter already provides:

- a Tkinter desktop interface with tool, image, and results areas;
- opening and exporting common image formats;
- side-by-side, fit-to-window previews of the original and current result;
- an optional comparison image and a scrollable tools sidebar;
- undo, redo, and reset-to-original behavior;
- a tiny plug-in-style contract for adding course functionality;
- one analysis example and one image-processing example.

## Quick start

Python 3.10 or newer is recommended.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

On some Linux distributions, Tkinter is a separate system package. For example,  
Ubuntu/Debian users may first need `sudo apt install python3-tk`.

The complete student handout is [docs/STUDENT_GUIDE.md](docs/STUDENT_GUIDE.md).

Open an image to see its original on the left and its current result on the right.
Tools, undo/redo, reset, and saving apply to the current result. Use **Open
comparison…** to show a different image on the left, or **Use original** to
restore the original preview. Drag the divider between the previews to resize
them. Scroll the tools sidebar with its scrollbar or the mouse wheel.

Set5 has separate tools for **Descriptor and Keypoints detection**, **Match with
second image**, and **Find closest image**. Each tool opens its own workflow
directly.

Keypoint matching asks for one set of ORB or SIFT settings for both images.
Choose **RANSAC** under **Matching score** to draw and score only matches that
agree with a projective transformation; this requires at least four initial matches.
Its match map appears directly in the interface and follows window resizing.
Toggle **Show matches** to return to the regular image panes. **Save result**
exports the map while it is shown, and otherwise saves the working image.
Matching leaves the working image and its edit history intact.

**Find closest image** searches only the images directly inside the folder you
select, excludes the current file and pixel-identical copies, and shows the best
candidate with its match map. It ranks by matching fraction, breaking ties by the
number of matched points.

Set6 includes **Face Detection**, using the pretrained
[YuNet model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
It marks face boxes, confidence scores, and five facial landmarks. The dialogs
default to a confidence threshold of 0.6, an NMS threshold of 0.3, and 5000
candidates before NMS. It reuses `Set6_materials/face_detection_yunet_2023mar.onnx`
when present; otherwise, the first run downloads the model to the ignored
`models/` directory. Both paths are relative to the project, regardless of the
working directory. To supply a model in code, use `FaceDetection(model_path=...)`.

**Set6 → Eigenfaces** loads a folder of training images, computes PCA with a
selectable explained-variance target (POV), and displays the first K eigenfaces.
Images can be directly in the selected folder or in subfolders.
See [docs/EIGENFACES.md](docs/EIGENFACES.md) for the workflow.

Set7 has three separate tools; each opens its workflow directly:

- **Codebook Construction:** builds and displays the visual vocabulary.
- **Histogram Construction:** displays the open test image's histogram.
- **Classification:** trains the SVM and displays its classification controls.

**Codebook Construction** asks for the number of visual words per class
(default: 10), extracts grayscale ORB/SIFT descriptors and fits k-means separately
for each class. Each gallery displays exactly that class's selected number of
centroids as representative 32 × 32 patches. Images keep their original sizes. With
Caltech, select `Caltech_101_subset/Caltech_101_subset` as the training folder.
If a class has too few distinct descriptors, choose a smaller number of words.

For **Histogram Construction**, open a test image in the app, then choose the
extractor, training folder and number of words per class (default: 10). It combines the class vocabularies
and displays the histogram of the current image directly. Each descriptor
votes for its nearest centroid (3 classes give 30 bins when enough descriptors
exist).

**Classification** trains an SVM with a chi-squared kernel on L1-normalized
training histograms, then shows the current image's histogram. Click **Classify**
in that window to display the most probable class and its model probability.
The selected test file is excluded from BoW training. Face Recognition also
excludes pixel-identical copies of its selected test image before fitting PCA.

Set8 includes **Classification with pretrained model** and **Feature projection**.
Install TensorFlow for model inference and feature extraction:

```sh
python -m pip install -e '.[cnn]'
```

Two trained CIFAR-10 models are prepared locally:

- `cnn_small/cnn_small.keras`, with `cnn_small/cnn_small.json`: a small CNN trained
  for 50 epochs, RGB 32 × 32 input, and pixel values divided by 255.
- `inception_cifar10/inception_cifar10.keras`, with its adjacent JSON passport:
  a frozen InceptionV3 base and trained classification head, RGB 128 × 128 input,
  and pixel values scaled to [-1, 1].

Passports record the dataset, training/validation sample counts, output class
order, preprocessing, and training history. Weights and datasets stay local and
are ignored by Git. To prepare another checkout, obtain the trained model exports
from Colab, put each `.keras` beside its versioned passport, and supply a CIFAR-10
test folder. A model ZIP may also be selected; it must contain one `.keras` and
its matching `<model_name>.json` passport. The prepared `cnn_small.zip` includes
both files; a raw export containing only `cnn_small_history.json` needs the
versioned `cnn_small/cnn_small.json` added to it.

Choose **Inspect passport** to open a scrollable metadata window; this requires
neither an open image nor TensorFlow. To classify, open an image, choose
**Classify model**, and select either model. Classification and feature extraction
share the model passport's resizing and normalization settings.

For **Feature projection**, choose **Extract and save features**, select a model,
then select `datasets/set8/cifar10_test` and save the vectors as `.npz`.
The provided sample contains 100 original test images, ten per class, organized
in class folders. Choose **Load and visualize features** to reopen the saved
vectors and display both t-SNE and UMAP in 2D, colored by the true folder labels.
This step reuses the saved vectors and does not load the CNN again.
Local verification artifacts are in `outputs/set8/`: vectors, projection plots,
and `evaluation.json` with accuracy on this 100-image sample, not the full test
set. To check the Set8 workflows automatically, run:

```sh
python -m unittest discover -s tests -p 'test_model_load.py' -v
python -m unittest discover -s tests -p 'test_feature_projection.py' -v
```

Set9 includes **Object detection with YOLO**, **Convert COCO dataset to YOLO**,
and **YOLO feature projection**. Install the detector dependencies:

```sh
python -m pip install -e '.[yolo]'
```

For the pretrained-detector exercise (9.1), the local checkpoint
`models/set9/yolo11n.pt` and example `datasets/set9/yolo_examples/bus.jpg` are
prepared. On another checkout, `YOLO("yolo11n.pt")` from the `ultralytics`
package downloads this COCO-pretrained model on first use. The supplied
`forensics_app/tools/set9/best.pt` is the blood-cell detector, with class IDs
`0: PC`, `1: RBC`, and `2: WBC`.

The local blood-cell exports are prepared in `datasets/set9/blood_cell_yolo/`
(3,027 training, 288 validation, and 144 test images) and
`datasets/set9/blood_cell_coco/test/` (144 test images). The original archives are
`blood cell.v1i.yolov11.zip` and `blood_cell.v1i.coco.zip` in Downloads. YOLO uses
one `.txt` annotation file per image, containing normalized
`class_id center_x center_y width height` rows, together with `data.yaml` for
class names and dataset splits. COCO stores image records, category IDs, and
pixel bounding boxes in a JSON file such as `_annotations.coco.json`. The class
order in the selected model must match the annotation class IDs.

**Object detection with YOLO** offers three options:

- **Inspect model:** select a `.pt` checkpoint to see its task and class names.
- **Detect objects:** open an image, select the checkpoint, and view bounding
  boxes, class names, and confidence scores. **Save result** exports the annotated
  image. Prediction follows the [Ultralytics PIL RGB input convention](https://docs.ultralytics.com/modes/predict/#inference-sources).
- **Compare ground truth and predictions:** open a test image, select the
  checkpoint and its matching YOLO `.txt` annotation. The file picker suggests
  the corresponding file in `labels/`. The original photo appears in panels
  labeled **Ground truth** and **Predictions**; **Save result** exports this
  comparison. Use **Undo** or **Reset** to return to the photo before another
  detection. This visual comparison does not calculate mAP.

**Convert COCO dataset to YOLO** asks for a folder containing one split's
`_annotations.coco.json` and images, then an empty output folder. It writes
`images/`, `labels/`, and `data.yaml`,
preserving the selected split rather than inventing training/validation/test
partitions. Folders named `train`, `valid`/`val`, or `test` keep that role in the
generated config; combine the independent splits when configuring training.
For this blood-cell export, it omits the unused Roboflow parent
category and maps the annotated classes to `PC: 0`, `RBC: 1`, and `WBC: 2`.
Select the resulting images and labels when comparing against `best.pt`.
The verified conversion of the 144 COCO test images is prepared in
`datasets/set9/blood_cell_coco_converted_test/`. The two exports use slightly
different coordinate rounding (at most half a pixel in box centers); their
images, object counts and class mappings agree.

**YOLO feature projection** is an additional visualization beyond the required
Set9 exercises. Select **Extract and save features**, the blood-cell checkpoint,
and `datasets/set9/blood_cell_yolo/test` (or its `images/` folder), then save a
`.npz` file. Feature vectors combine pooled detection-head maps at multiple
scales; their labels come from the real YOLO `.txt` files. Choose **Load and
visualize features** to display t-SNE and UMAP colored by those labels. Images
containing several classes receive a combined label. This tool can run without
an image open in the main window.

## Useful commands

```
python -m forensics_app
python -m unittest discover -s tests -v
python -m compileall forensics_app
```

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.
