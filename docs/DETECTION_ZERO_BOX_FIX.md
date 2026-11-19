# ParkNChill: Zero Vehicle Boxes / Zero Slot Fix

## Source changes in this repair

1. The default model setup script downloads the official YOLOX-s ONNX artifact. YOLOX uses a different pre/post-processing convention from YOLOv5.
2. `DetectionPreprocessor.preprocessYoloX()` supplies BGR CHW float32 values in the 0–255 range, with a 114-valued pad and top-left letterboxing, matching the official YOLOX inference path.
3. `DetectionFilter.decodeYoloX()` decodes raw YOLOX box outputs using the 8/16/32 strides and exponentiated width/height. The old generic YOLOv5 decoder must not be used for raw YOLOX output.
4. `OnnxVehicleDetectionService` chooses a preprocessing/decoder profile. For the generic `vehicle_detector.onnx` filename, it attempts to inspect ONNX output metadata so an older YOLOv5 model copied into that name can use the RGB-normalized decoder profile instead. An explicit override is also available:
   - `-Dparknchill.onnx.inputProfile=YOLOX_BGR_255`
   - `-Dparknchill.onnx.inputProfile=RGB_NORMALIZED`
5. `ParkingView` provides a **Load Slot XML** button. If a PKLot image has no same-basename XML beside it, select the matching `.xml` file manually. The app validates that the image and XML basenames match, then loads the original PKLot slot polygons and source labels.
6. The status message remains explicit: a detector can be `READY` while producing zero boxes. A loaded model is not proof of successful detections.

## Verification performed in this environment

- Compiled the changed detector core classes with `javac` successfully.
- Ran a deterministic YOLOX decoder smoke test with a synthetic known-output row. It passed and returned the expected car box coordinates and confidence.
- Full Maven tests and a live JavaFX/ONNX inference were not run here. Maven and the ONNX model binary are unavailable in this environment. Verify on Windows with the actual installed model and PKLot dataset before claiming real-image accuracy.

## Windows verification

From the folder containing `pom.xml`:

```powershell
mvn clean test
mvn javafx:run
```

Check that a photograph containing visible cars produces boxes. If the detector is `READY` but still gives zero detections, read the full detection status message and verify that the actual ONNX file matches the selected profile. For PKLot slot boundaries, use a matching XML annotation (same image basename); loading a photograph alone does not create trustworthy parking-slot polygons.
