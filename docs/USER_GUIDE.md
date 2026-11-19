# ParkNChill — Quick User Guide

## 1. Setup

1. Install Java 17+ and Apache Maven.
2. From the project root run `mvn clean test`.
3. Check that `models/vehicle_detector.onnx` exists. This project ZIP includes the model; use [`MODEL_SETUP.md`](../MODEL_SETUP.md) only if it is missing or must be replaced.
4. Launch with `mvn javafx:run` and check that the detector status says `READY`. Then confirm real bounding boxes appear on an image containing vehicles.

## 2. Four-step workflow

### Load & Analyze
Choose **Open Image**. ParkNChill supports image enhancement via **Enhance Image** / **View IVP Stages**. When enhancement is applied, the enhanced image becomes the active working image from which all eight canonical IVP features are calculated. Vehicle inference receives the untouched original full-scene image. The active ten-feature schema is `v2.3-10feat-enhanced-v1` with canonical policy `ivp-enhance-v1`. Serialized models and datasets record these identifiers; mismatched models are rejected with `SchemaMismatchException`. If there are zero slots, load or import valid slot annotations or fix localization before continuing.

### Review Slots
Select each slot. Confirm **Occupied** only when a vehicle is in the bay, **Available** only when the bay is visibly empty, or **Needs Review** when uncertain. Enter a real reviewer ID when requested. Do not bulk-approve detector suggestions.

### Train & Check
Choose **Load Dataset (PKLot)...** for the external PKLot dataset or use genuinely reviewed examples. PKLot must be downloaded and extracted separately. Select the top-level extracted dataset root containing multiple lot/camera, weather, and date folders, not one single date folder. Read the import summary and confirm valid image/XML pairs, ten-feature records, and at least one eligible group in each TRAIN, VALIDATION, and TEST partition. TRAIN must contain both Occupied and Available labels. If the UI says more groups are needed, select the broader PKLot root and re-import; do not force training on a single group.

### Save & Use
Save or load the compatible Decision Tree model. Check the active model status before relying on predictions.

## 3. Common problems

- `MODEL_MISSING`: confirm the extracted project contains `models/vehicle_detector.onnx`; then follow `MODEL_SETUP.md` if the file is missing or invalid.
- `0 slots`: vehicle detection and slot localization are separate; use correct parking-slot annotations or fix slot localization.
- `Train` disabled: check the displayed reason, feature schema, class counts, and independent capture groups.
- Tests fail: inspect the first error from the latest `mvn clean test` run; do not rely on old Surefire reports.

## 4. Evaluation

Do not report training-set accuracy as test accuracy. Keep related images from the same camera/date group together. PKLot results do not guarantee generalization to a different camera, parking lot, or night-time scene.

## If the image shows zero parking spaces

1. Click **Open Image** and select the PKLot photograph.
2. If the app does not find a same-basename XML beside the image, click **Load Slot XML**.
3. Choose the `.xml` file with the same filename stem as the photograph. For example, `photo.jpg` must be paired with `photo.xml`.
4. Check that the orange/green/red slot outlines align with the parking bays. The displayed initial labels are the PKLot source annotations.
5. Click **Detect and Analyse** to run live vehicle inference on that image.

Slot boundary annotations and vehicle detections are separate: the XML provides the parking-space regions; the ONNX model provides vehicle boxes. If no XML or imported slot file exists, the general-purpose IVP slot localizer may not reliably identify every parking space in a perspective PKLot scene.

## If vehicle detection reports zero vehicles

Check the full status message at the bottom of the window. `Detector: READY` means only that the model/runtime initialized. The official YOLOX-s model requires the YOLOX-specific raw BGR 0–255 input profile and grid/stride output decoding. If a different model is installed at `models/vehicle_detector.onnx`, make sure the configured input profile and decoder match that model. See `MODEL_SETUP.md` and `docs/DETECTION_ZERO_BOX_FIX.md`.


## Viewing a saved Decision Tree without replacing detections

1. Load a compatible enhanced 10-feature `.pncmodel` using **Save & Use**. Loading activates the model but does not rerun analysis or change the existing detector result.
2. Return to **Review Slots**. Confirm the current image has been analysed.
3. Click **Run Tree Separately**. A new **Loaded Tree Prediction** column appears with the tree's status and leaf purity.
4. Select a slot to inspect the saved tree's rule path. The existing **Last Analysis Result**, detector suggestions, occupancy totals, and overlays remain unchanged.
5. If you deliberately want the tree to produce the primary status, choose **Feature-Based Decision Tree** in Classifier Mode and click **Detect and Analyse**. That is a separate full analysis and is expected to change the primary result.
