# Enhanced 10-Feature Dataset Directory (`v2.3-10feat-enhanced-v1`)

## Status: UNGENERATED / NOT BUNDLED

In accordance with ParkNChill v2.3 core rules (**Truth over Polish**, **No Fabrication**, and **WS-1 Provenance Integrity**):

1. **No Synthetic Fabrication**: The bundled 48-sample legacy dataset in `data/labels/legacy/` contains only 8 image features (`v2.2-8feat`). The two vehicle-detection features (`vehicleOverlapScore` and `vehicleConfidenceScore`) cannot be recovered or mathematically inferred from 8-feature data without running live ONNX vehicle detection on actual ground-truth photographs.
2. **Reproduction Recipe**: To generate canonical enhanced 10-feature training data:
   - Download the real PKLot dataset or provide overhead parking lot photographs.
   - Use the **Step 1 & Step 2** review workflow in the ParkNChill UI or run the PKLot Importer (`PkLotDatasetImporter`).
   - The production feature extraction pipeline will extract the 8 enhanced IVP features (contrast/gamma stretched) and query the ONNX vehicle detector for the 2 vehicle features, producing verified records under schema `v2.3-10feat-enhanced-v1` and policy `ivp-enhance-v1`.
3. **Fail-Closed Policy**: ParkNChill strictly forbids mixing legacy 8-feature rows with canonical 10-feature rows.
