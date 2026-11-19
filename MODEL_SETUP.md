# ParkNChill Vehicle Detector Setup

This file is the single setup guide for the pretrained ONNX vehicle detector.

## Important

This project archive includes `models/vehicle_detector.onnx` (35,858,002 bytes). Its SHA-256 in this archive is `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`. The PKLot dataset is external and must be downloaded separately. The bundled model SHA-256 is listed above as a file-integrity check. If the model file is missing after extraction, use the setup script below to download it again.

**Important fix for zero detections:** the default download is YOLOX-s. It must be fed BGR pixels in the `0..255` range with top-left letterbox padding. ParkNChill now uses that default-specific preprocessing. The app reports the selected preprocessing profile if a loaded model returns zero boxes.

## Windows 10/11

1. Install Java 17 or newer and Apache Maven.
2. Extract the ParkNChill ZIP.
3. Open PowerShell in the folder containing `pom.xml`.
4. If `models\vehicle_detector.onnx` is missing or corrupt, run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\setup_onnx_model.ps1 -Force
   ```

5. The script prints the file size and SHA-256 hash. The `-Force` option replaces an existing model; without it, the script will keep an existing file.
6. Confirm the model file exists:

   ```powershell
   Test-Path .\models\vehicle_detector.onnx
   (Get-Item .\models\vehicle_detector.onnx).Length
   ```

7. Run the tests and launch the app:

   ```powershell
   mvn clean test
   mvn javafx:run
   ```

8. Confirm the application reports `Detector: READY`. If it reports `MODEL_MISSING` or `INFERENCE_FAILED`, do not assume vehicle detection is working.
9. Run a parking image that visibly contains cars. `READY` means the file/runtime initialized; confirm that actual bounding boxes appear. The bundled YOLOX setup URL requires YOLOX-specific preprocessing: BGR channel order, float32 values in the 0..255 range, top-left letterbox placement, and padding value 114. The source now uses this profile by default for `models/vehicle_detector.onnx`.

### Alternative models

If replacing the default YOLOX model with a YOLOv5/YOLOv8 model expecting RGB float32 values in `[0,1]` with centered letterboxing, launch with:

```powershell
mvn javafx:run "-Dparknchill.onnx.inputProfile=RGB_NORMALIZED"
```

Only use the profile that matches the exact ONNX export. Changing a filename does not convert the model.

## Linux / macOS

From the project root, run:

```bash
bash scripts/setup_onnx_model.sh --force  # only if model is missing/corrupt
mvn clean test
mvn javafx:run
```

## Model source and licensing

The setup scripts reference the YOLOX-s ONNX artifact published in the official Megvii-BaseDetection/YOLOX GitHub release:

`https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx`

Project code/model licensing and release-specific terms should be checked against the upstream repository and release before redistributing the downloaded binary. This setup document does not independently certify the downloaded file's contents or licence. The setup script's printed SHA-256 identifies the downloaded file; compare it with a trusted upstream checksum if one is published.

The official YOLOX demo uses a validation transform on the original OpenCV BGR image and does not use the same RGB/0..1 normalization that some YOLOv5/YOLOv8 ONNX exports require. See the upstream [YOLOX demo](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/demo.py) and [ONNX Runtime documentation](https://yolox.readthedocs.io/en/latest/demo/onnx_readme.html).

## If download fails

- Check internet access and whether GitHub downloads are blocked by your network.
- Do not rename an arbitrary `.onnx` file to `vehicle_detector.onnx`; model architecture, tensor shape, preprocessing, and class mapping must match this project's inference code.
- Keep the application honest: `MODEL_MISSING` means no model is installed, not that the detector found zero cars.

## What this does not provide

Installing the vehicle detector does not automatically solve parking-slot boundary localization. For PKLot, use the paired XML annotations and verify that the imported slot boundaries align with the original image. The full PKLot dataset is not included in this ZIP.


## Troubleshooting zero detections

For the official YOLOX-s ONNX file from the setup script, ParkNChill uses the official input convention (BGR, NCHW, float32 0–255; image resized with top-left content placement and padding value 114) and decodes YOLOX raw grid-relative boxes using strides 8, 16, and 32 before NMS. Do not substitute a YOLOv5/YOLOv8 model under the generic filename without setting its matching input profile. A `READY` status only confirms model/runtime availability; verify that a car-containing image actually yields bounding boxes.

## Image enhancement and feature compatibility

ParkNChill provides full-image and slot enhancement using canonical policy `ivp-enhance-v1` (luminance contrast stretch at 2nd/98th percentiles with BT.709 luma, adaptive gamma, and RGB Gaussian noise reduction). In the canonical pipeline, all eight IVP features are calculated from the active enhanced image, while vehicle detection uses the original full-scene image. The active authoritative ten-feature schema is `v2.3-10feat-enhanced-v1`. Serialized models and datasets record both the schema ID and enhancement policy ID; mismatched models are rejected with `SchemaMismatchException`.
