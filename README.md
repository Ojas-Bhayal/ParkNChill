# ParkNChill

**Smart Parking Occupancy Detection — JavaFX desktop project**

ParkNChill analyses a parking-lot image, identifies parking spaces, uses a vehicle detector to provide evidence, and supports an explainable CART Decision Tree for classifying slot occupancy.

## Start here

### 1. Requirements

- Java 17 or newer
- Apache Maven 3.9 or newer
- Windows, Linux, or macOS desktop with JavaFX support
- Internet access only if you need to re-download the bundled ONNX vehicle model or obtain PKLot

### 2. Open the project

Extract the ZIP, open a terminal in the extracted project folder, and run:

```bash
mvn clean test
mvn javafx:run
```

If the tests fail, read the first error in the terminal before continuing. Do not rely on old reports from a previous build.

### 3. Use the application

The application has four simple steps:

1. **Load & Analyze** — open a parking photo. ParkNChill creates a contrast/brightness-enhanced preview automatically. Use **Show Original** to compare, **Show Enhanced** to switch back, and **View IVP Stages** to inspect the actual processing images. Then select **Detect and Analyse**.
2. **Review Slots** — select each slot and confirm **Occupied** or **Available**. Keep unclear cases as **Needs Review**.
3. **Train & Check** — load PKLot or use reviewed examples, inspect eligible records and partitions, then separately click Train Decision Tree and inspect the evaluation results.
4. **Save & Use** — save the trained model as a `.pncmodel` file or load a compatible saved model. Loading activates the model only; it does not rerun detection or replace the visible results. Return to **Review Slots** and click **Run Tree Separately** to populate the separate **Loaded Tree Prediction** column without changing detector results or the occupancy summary.

In **Train & Check**, the panel **What Loading This Dataset Does to the Decision Tree** shows the effect of dataset loading: imported/eligible rows, class balance, source labels, and TRAIN/VALIDATION/TEST groups. **Loading only prepares data; it does not train or update the tree.** Click **Train Decision Tree** to fit rules, then inspect the results. Use **Explain Dataset → Decision Tree** for the full workflow.

The top of the application shows the vehicle-detector status. This ZIP includes `models/vehicle_detector.onnx` (about 35.9 MB). If the status says `MODEL_MISSING`, first check that the file exists under the extracted project folder.

### 4. Check or reinstall the vehicle detector

Normally, no download is needed because the model is bundled. Only run the setup script if the file is missing or corrupt. Use `-Force` on Windows or `--force` on Linux/macOS to replace the existing file:

**Windows PowerShell**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_onnx_model.ps1
```

**Linux/macOS**

```bash
bash scripts/setup_onnx_model.sh
```

Restart ParkNChill and confirm the status reads `Detector: READY`. If the download fails, see `models/README.md`.

### 5. Import PKLot

Download PKLot from the [official dataset page](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) and extract it separately. In **Train & Check**, choose **Load Dataset (PKLot)...** and select the extracted dataset root that contains the lot/weather/date folders. Do not put the full PKLot archive inside the project ZIP.

**Important:** the detector must be installed before importing PKLot for ten-feature training. Select the **top-level extracted PKLot folder** that contains multiple camera/lot, weather, and date folders—not a single date folder. Inspect the import summary and confirm that usable TRAIN, VALIDATION, and TEST groups exist and that the TRAIN group includes both Occupied and Available slots. If the message requests more independent groups, select the broader root and re-import.

## What the demo can and cannot show

- The included demo picture helps you learn the interface. It is not evidence of real-world accuracy.
- An enhanced working copy is generated and can be applied via **Enhance Image** / **View IVP Stages**. The canonical IVP features are calculated from the active enhanced image, while ONNX vehicle detection operates on the original full-scene image. The active ten-feature schema is `v2.3-10feat-enhanced-v1` under canonical policy `ivp-enhance-v1`. Models trained on incompatible schemas or unenhanced ROI values are rejected by the active enhanced pipeline.
- The Decision Tree needs compatible ten-feature records with valid labels from both occupancy classes.
- Evaluate on the held-out test data, not just the examples used for training.
- Results from PKLot do not guarantee the same accuracy on a different camera angle, parking lot, or night-time scene.

## More help

- `docs/USER_GUIDE.md` — short user walkthrough
- `docs/PKLOT_IMPORT.md` — how to prepare and import PKLot
- `docs/DATASET_TO_DECISION_TREE.md` — difference between loading a dataset, training a tree, saving it, and loading a saved model
- `docs/PROJECT_AUDIT.md` — known limitations and verification status
- `MODEL_SETUP.md` — detector setup, expected input format, checksum, and troubleshooting
- `models/README.md` — detector requirements

## If a PKLot image shows zero parking spaces

When opening a PKLot image, keep its matching same-basename `.xml` annotation in the same folder (for example, `photo.jpg` and `photo.xml`). ParkNChill now loads the XML parking-space polygons and labels automatically when the pair is present. Without a matching XML sidecar, the generic line-based slot detector may be uncertain on elevated/perspective camera views; use **Load Dataset (PKLot)...** or **Sample Annotations...** rather than treating zero slots as a result.

If the detector badge says `READY` but returns zero vehicle boxes on a photograph containing cars, check `MODEL_SETUP.md`. `READY` means the ONNX runtime initialized; the default YOLOX model must use BGR `0..255` input and top-left letterboxing, which is handled by the current source.

### If vehicle detection returns zero boxes

The default setup script installs the official YOLOX-s ONNX model. That model needs YOLOX-specific preprocessing (BGR float32 in the 0–255 range, top-left letterbox padding) and grid/stride decoding of its raw `[1, 8400, 85]` output. ParkNChill now uses that profile for the default YOLOX model. `Detector: READY` means the model loaded; it does not guarantee cars were detected. Check the status message and verify that the configured ONNX file is actually YOLOX-s.

### If a PKLot photograph shows zero parking spaces

If the selected image is not beside its matching XML annotation, click **Load Slot XML** in the image toolbar and select the XML file with the same base filename as the image. The XML provides PKLot's parking-space geometry and source labels; it is separate from live vehicle detection. If you have neither annotations nor a working automatic slot localizer for that scene, ParkNChill cannot reliably infer every empty space from the photograph alone.

## Image enhancement, IVP, and training

See [`docs/IMAGE_ENHANCEMENT_AND_TRAINING.md`](docs/IMAGE_ENHANCEMENT_AND_TRAINING.md) for the plain-language workflow, before/after enhancement behavior, IVP previews, and PKLot group requirements.
