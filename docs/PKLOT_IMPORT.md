# Importing PKLot into ParkNChill

## 1. Prepare the dataset

1. Download PKLot from its official source.
2. Extract the archive with 7-Zip or an equivalent tool.
3. Keep its original folders and filenames. Do not rename image/XML pairs.
4. Check that the extracted root contains the lot/camera, weather, date, image, and matching XML annotation structure.
5. Keep the dataset outside this source project. The complete dataset is large and is not bundled in the ParkNChill ZIP.

## 2. Install the vehicle detector first

Run the appropriate script from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_onnx_model.ps1
```

Or on Linux/macOS:

```bash
bash scripts/setup_onnx_model.sh
```

Restart ParkNChill. The header should display `Detector: READY`. If the detector is missing or incompatible, do not proceed as though valid vehicle-overlap features were created.

## 3. Import

1. Open **Train & Check**.
2. Select **Load Dataset (PKLot)...**.
3. Choose the top-level extracted PKLot root directory, not the `.tar.gz` file and not a single `lot/weather/date` subfolder. The root should contain several camera/lot folders (for example `PUCPR`, `UFPR04`, and `UFPR05`) and multiple acquisition dates.
4. Let the import finish; do not close the application while it is running.
5. Review the summary for paired images, usable slots, occupied/available counts, invalid annotations, and detector failures.
6. Use **Sample Annotations...** to inspect original XML labels and boundaries. This is an inspection action; it must not create fake features.
7. Confirm the import summary shows actual usable groups assigned to TRAIN, VALIDATION, and TEST. The importer preserves whole camera/date groups. If training is blocked due to missing groups, choose the broader PKLot root and import again rather than manually moving records between partitions.

## 4. Before training

- Check a sample of annotated slots against their original photographs.
- Keep all slots from one image together in a partition.
- Keep related camera/date scenes together where the importer supports it.
- Ensure the ten feature values come from actual image processing and detector results, not placeholder zeros.
- Confirm that the saved dataset manifest and feature rows match.
- Train only after the application confirms the dataset is eligible.

## 5. Report results correctly

Use the held-out test partition for final metrics. Do not use test images for training or hyperparameter selection. Keep the test partition frozen when retraining the model. In your report, distinguish PKLot test results from performance on a new local camera.
