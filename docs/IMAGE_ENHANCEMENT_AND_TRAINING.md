# ParkNChill: Image Enhancement, IVP, and Decision Tree Training

## What each part does

- **Vehicle detector (ONNX):** finds vehicle bounding boxes in the original full-scene image. The original image is retained so detector preprocessing is not unexpectedly altered.
- **Image enhancement:** creates a separate, enhanced copy for slot-localization previews and IVP feature extraction. The original is never overwritten.
- **IVP:** processes each slot crop to calculate grayscale/intensity, saturation, foreground, edges, and connected-component features. The application shows the original crop and enhanced crop in View IVP Stages.
- **Decision Tree:** learns from slot feature vectors plus occupancy labels. It is not trained just by loading an image or by running the vehicle detector.

## Enhancement behavior

The enhancement policy is `ivp-enhance-v1`: conservatively stretches luminance between the 2nd and 98th percentiles using BT.709 luma when the image has enough contrast range, applies adaptive gamma adjustment, and performs multi-channel Gaussian noise reduction. When applied, all eight IVP features are calculated directly from the enhanced working image. Vehicle detection operates on the original full-scene photograph.

The active authoritative feature schema is `v2.3-10feat-enhanced-v1`. Serialized models and datasets record both the schema ID and enhancement policy ID; incompatible models are rejected with `SchemaMismatchException`.

## Recommended step-by-step usage

1. Extract the project ZIP and run `mvn clean test`.
2. Check `models/vehicle_detector.onnx` exists and the app reports `Detector: READY`. Test boxes on a real image; READY alone is not proof that detections are accurate.
3. Click Open Image. The image preview offers **Show Original / Show Enhanced**. The source file remains unchanged.
4. For PKLot images, load the photograph with its matching XML annotation, or use **Load Slot XML** to choose the same-basename annotation. Check that the polygons follow the actual parking bays and that XML occupied/vacant labels look correct.
5. Click Detect and Analyse. Inspect the vehicle boxes. If the count is unexpectedly zero, inspect the detector status/output instead of treating the image as empty.
6. Select a slot and click **View IVP Stages** to see the actual intermediate images.
7. In Train & Check, choose the PKLot dataset and import from the **top-level extracted PKLot folder**, not an individual weather/date folder. This gives the importer access to more camera/date groups.
8. Read the import summary. It must show at least one usable TRAIN, VALIDATION, and TEST partition, and both occupied and available examples in TRAIN. If not, import a broader selection of camera/date groups. Never bypass this warning by randomly splitting individual slots from the same capture group.
9. Train only when the button is enabled. The PKLot training route preserves group partitions, trains on TRAIN, selects settings with VALIDATION, and evaluates on held-out TEST. Check counts and metrics.
10. Save the model and test it on a different parking scene. Report the result as PKLot performance unless you also evaluate on independently labeled images from your own camera environment.

## If the project says more camera/date groups are needed

Select the root folder containing all PKLot camera folders (for example PUCPR, UFPR04, UFPR05) and multiple date/weather folders, then re-import. Hundreds of slots taken from one capture group are still one group; sample count cannot replace independent groups.

## Important limitations

- A vehicle detector is not a parking-slot detector. Parking-space boundaries must come from PKLot XML, a validated localization method, or user correction.
- The full PKLot archive is not bundled in the source ZIP.
- The exact ONNX model file and its metadata are documented in `MODEL_SETUP.md`.
- Enhancement can improve some low-contrast imagery but is not guaranteed to improve every photograph. Always inspect before/after output and confirm that the same policy is used for training and inference.

## Where to find image enhancement and IVP in the desktop app

After loading a photograph, the main image toolbar now exposes **Enhance Image** and **View IVP Stages** directly, rather than hiding IVP behind the advanced options.

- **Enhance Image** opens an Original vs Enhanced preview with Contrast, Brightness, and Gamma sliders. If a parking-slot row is selected, the dialog previews that slot crop; if no slot is selected, it previews the full photograph.
- **View IVP Stages** opens the actual pipeline stages for the selected slot, or a full-image overview if no slot is selected.
- The interactive sliders are a visual enhancement preview. To preserve training/inference consistency, the model feature pipeline continues to use the versioned `ImageEnhancer.autoEnhance()` policy. The slider values are not silently saved into the training data.
- The small **Show Original / Show Enhanced** button remains available to toggle the main workspace preview.
