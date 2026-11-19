# ParkNChill v2.3 — AI/IVP Visibility Update

## Changes
- Added a visible **View IVP Stages** action in Load & Analyze, Review Slots, and Train & Check / Save & Use workspaces.
- Added `IvpStagesDialog`, which calls the existing `FeatureExtractor.processWithStages` implementation and displays eight real stages with eight numerical IVP feature values.
- Supports a slot-specific ROI preview when a slot is selected; otherwise uses a downscaled full-image overview and explicitly warns that those scene-level values are not slot training records.
- Added plain-English explanation of the roles of vehicle AI, IVP, and the CART Decision Tree.
- Removed the implicit `OPERATOR_1` value from the advanced slot inspector.
- Updated setup/user docs to match the included ONNX model and made model-download scripts preserve an existing model unless force is requested.
- Removed generated `target/` classes and stale Surefire reports from the release archive.

## Contents / data requirements
- The ONNX vehicle model is bundled at `models/vehicle_detector.onnx`.
- The complete PKLot dataset is not bundled and must be downloaded separately.

## Verification status
- The new `IvpStagesDialog.java` passed a source/type compilation check against the existing compiled project classes using temporary JavaFX API stubs.
- A separate runtime smoke check executed `FeatureExtractor.processWithStages` on the bundled demo image and confirmed all eight IVP stage images and eight feature values were produced.
- The complete Maven test suite and interactive JavaFX launch could not be run in the editing environment because Maven is unavailable and package-network access failed. Run `mvn clean test` and `mvn javafx:run` on your Windows machine before treating this as fully verified.

## Image Enhancement and PKLot Training Workflow Update

- Added a dedicated, deterministic canonical image enhancement policy (`ivp-enhance-v1`) that produces an enhanced working copy without mutating the original image.
- The main image workspace previews the enhanced copy by default and offers a toggle to show the original; ONNX full-scene detection continues to use the original image.
- Slot localization and canonical IVP feature extraction use the active enhanced working copy for interactive records and PKLot imports.
- Added the enhanced intermediate image to View IVP Stages.
- The active authoritative ten-feature schema is `v2.3-10feat-enhanced-v1`. Serialized models and datasets record the schema ID and enhancement policy ID; incompatible models are rejected with `SchemaMismatchException`.
- Saved-model metadata now records the enhancement preprocessing version.
- PKLot import summaries show camera/date group counts and partition readiness; the guidance asks users to select the top-level extracted PKLot root and include multiple camera/date groups.
- The model is bundled in this release; the complete PKLot dataset remains external.

### Verification limitation

The standalone Java ImageEnhancer smoke test passed (new enhanced copy, same dimensions, source pixels unchanged, and visible pixel change). `ModelSerializer.java` compiled against the available project classes. Maven is not installed in the build environment, so the full Maven/JUnit suite and JavaFX GUI could not be verified here.
