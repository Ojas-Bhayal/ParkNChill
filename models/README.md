# Vehicle detector model

This release includes `vehicle_detector.onnx` in this directory. The Java application expects this exact path by default.

To verify the file is present on Windows:

```powershell
Test-Path .\models\vehicle_detector.onnx
(Get-Item .\models\vehicle_detector.onnx).Length
```

If the file is absent or damaged, follow [`MODEL_SETUP.md`](../MODEL_SETUP.md). A loaded model showing `Detector: READY` confirms that the runtime loaded it; it does not by itself confirm that real vehicles are detected. Check the detection boxes on a sample image. Installing a vehicle detector does not provide parking-slot boundaries; use the matching PKLot XML annotations or manually verify slot boundaries.
