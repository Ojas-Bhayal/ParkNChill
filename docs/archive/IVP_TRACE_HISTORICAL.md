> [!WARNING]
> **HISTORICAL, pre-remediation, not current.**
> This document records the defect trace from Phase 0 before remediation. For the authoritative current implementation, see `docs/IVP_TRACE.md`.

# ParkNChill v2.3 — IVP Trace, Caller Inventory, and Claim Audit (Historical Defect Record)

**Document Version**: 1.0.0-HISTORICAL  
**Phase**: Historical Phase 0 (Trace Before Changing)  
**Baseline Test Status**: 182 tests run, 0 failures, 0 errors, 1 skipped (`RealPkLotArchiveIntegrationTest.testFullArchiveIfExtracted`)  
**Timestamp**: 2026-10-22T23:21:00+05:30  

---

## 1. Pixel-Flow Trace (Historical Defect State)

This trace details the historical defect where 8 IVP numerical features were computed from raw unenhanced pixels while the UI displayed an enhanced preview.

| Feature / Artifact | Source Image Object | Operations & Conversions | File & Line Reference |
| :--- | :--- | :--- | :--- |
| **Preview: Enhanced Image (Global)** | Original full scene (`BufferedImage fullImage`) | `ImageEnhancer.autoEnhance(fullImage)`: 2nd–98th percentile luminance stretch + gamma correction. | `ParkingView.java:626` |
| **Preview: Enhanced Crop** | Original slot crop (`roi`) | `ImageEnhancer.autoEnhance(roi)` called, stored in `ProcessedRoi.getEnhanced()`. | `FeatureExtractor.java:128` |
| **Preview: IVP Stages** | Original slot crop (`roi`) | `GrayscaleConverter.toGrayscale(roi)` and `ColorSpaceConverter.toSaturationMap(roi)` — **computed from raw unenhanced ROI!** | `FeatureExtractor.java:132-133`, `IvpStagesDialog.java:79-96` |
| **Feat 0: `meanIntensity`** | **Original crop (`roi`)** | `roi` -> `toGrayscale(roi)` -> `applyGaussianFilter(gray, 5)` -> `computeMeanIntensity(smoothed)`. | `FeatureExtractor.java:132, 136, 152` |
| **Feat 1: `foregroundRatio`** | **Original crop (`roi`)** | `roi` -> `toGrayscale(roi)` & `toSaturationMap(roi)` -> `segmentForeground(smoothed, saturation)` -> `computeForegroundRatio(binary)`. | `FeatureExtractor.java:132, 133, 139, 153` |
| **Feat 2: `darkPixelRatio`** | **Original crop (`roi`)** | `smoothed` (from raw `roi`) -> count pixels with intensity < 85 -> divided by total pixels. | `FeatureExtractor.java:136, 154` |
| **Feat 3: `edgeDensity`** | **Original crop (`roi`)** | `smoothed` (from raw `roi`) -> `applySobel(smoothed)` -> count pixels with gradient > 30 -> divided by total pixels. | `FeatureExtractor.java:149, 155` |
| **Feat 4: `objectArea`** | **Original crop (`roi`)** | `binary` (from raw `roi`) -> `MorphologyProcessor.close(binary, 3)` -> `computeForegroundRatio(closed)`. | `FeatureExtractor.java:142, 156` |
| **Feat 5: `meanSaturation`** | **Original crop (`roi`)** | `roi` -> `ColorSpaceConverter.toSaturationMap(roi)` -> `computeMeanIntensity(saturation)`. | `FeatureExtractor.java:133, 157` |
| **Feat 6: `saturationRatio`** | **Original crop (`roi`)** | `saturation` (from raw `roi`) -> count pixels with S > 90 -> divided by total pixels. | `FeatureExtractor.java:133, 158` |
| **Feat 7: `largestBlobRatio`** | **Original crop (`roi`)** | `closed` (from raw `roi`) -> `ConnectedComponentAnalyzer.analyze(closed)` -> largest 8-connected component area ÷ slot area. | `FeatureExtractor.java:145, 159` |
| **Feat 8: `vehicleOverlapRatio`** | **Original full scene + YOLOX detections** | Bounding box / rotated polygon intersection area ÷ slot area (`bestIoS()`). | `SlotOverlapAnalyzer.java:152`, `ParkingAnalysisService.java:714` |
| **Feat 9: `vehicleMatchedFlag`** | **Original full scene + YOLOX detections** | Boolean matched flag: `1.0` if vehicle IoS >= threshold; `0.0` otherwise. | `SlotOverlapAnalyzer.java:165`, `ParkingAnalysisService.java:714` |

---

## 2. Historical Caller Inventory & Persistence
- `features.csv`: 48 legacy samples (`v2.2-8feat`).
- `manifest.csv`: 22 columns with legacy metadata.
