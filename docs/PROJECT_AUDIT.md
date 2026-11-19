# ParkNChill Simplification Audit

## What was changed in this simplified release

- Replaced the six visible navigation buttons with four user-facing steps: **Load & Analyze**, **Review Slots**, **Train & Check**, and **Save & Use**. Internal states remain separate so analysis and evaluation logic can remain intact.
- Shortened training/status copy and moved secondary dataset provenance, feature schema, and tree-rule information into the collapsed Advanced Settings section.
- Simplified the image toolbar by removing duplicate 100% and Fit Image controls; zoom and Reset View remain available.
- Made the normal app startup open to a blank workspace instead of automatically loading the demo and running analysis before the user chooses what to do.
- Changed the default reviewer identity to blank; the user must enter a real ID before making human-reviewed labels.
- Replaced the long root implementation report with a short start guide and separate focused guides.

## Package structure

- `src/main/java/` — JavaFX app, computer vision, ONNX integration, Decision Tree, and PKLot import.
- `src/test/java/` — unit and integration test sources.
- `data/` — small demo image, configuration, and bundled example labels.
- `models/` and `scripts/` — instructions and scripts for installing the vehicle model.
- `docs/` — end-user guides and this audit.

## Known limitations

1. **The pretrained ONNX model is not bundled.** Run `scripts/setup_onnx_model.ps1` on Windows or `scripts/setup_onnx_model.sh` on Linux/macOS, then confirm `Detector: READY`.
2. **PKLot itself is not bundled.** Download and extract it separately, then use `Load Dataset (PKLot)...`.
3. **The default demo image is a UI/processing example, not an independent real-camera validation set.** Do not use it to claim real-world accuracy.
4. **The codebase still contains multiple internal computer-vision and dataset classes.** They were retained to avoid destructive removal of functions; the simplification focuses on the user workflow, packaging, and instructions.
5. **Build and GUI verification limitation for this audit:** Java 21 is available in the editing environment, but Maven was not installed and package-network access failed. Therefore `mvn clean test` and an interactive JavaFX launch could not be run here. Test source updates were made for the new four-step UI, but a fresh local build is required before treating this ZIP as verified.
6. **Model/data prerequisite:** ten-feature PKLot training requires a compatible ONNX model and valid image/XML pairs. If either is missing or inference fails, those records must not be treated as complete ten-feature samples.

## Local verification commands

From the extracted project directory:

```bash
mvn clean test
mvn javafx:run
```

On first run, install the ONNX model using the setup script described in `README.md`.
