# ParkNChill v2.3 — Authoritative IVP Trace & Feature Architecture

**Document Version**: 2.3.0-CURRENT  
**Authoritative Schema ID**: `v2.3-10feat-enhanced-v1` ([`ProcessingPolicy.CANONICAL_SCHEMA_ID`](file:///C:/Users/ojas/OneDrive/Desktop/ParkNChill_v2.3_Simplified/src/main/java/com/parknchillai/imageprocessing/ProcessingPolicy.java#L15))  
**Authoritative Policy ID**: `ivp-enhance-v1` ([`ProcessingPolicy.CANONICAL_POLICY_ID`](file:///C:/Users/ojas/OneDrive/Desktop/ParkNChill_v2.3_Simplified/src/main/java/com/parknchillai/imageprocessing/ProcessingPolicy.java#L16))  
**Core Invariant**: **What is displayed is what is computed.** The pixel data presented as "Enhanced" in previews and IVP stages is byte-identical to the pixel data fed into the eight numerical IVP feature extractors.

---

## 1. Pixel-Flow Trace: Current Ten-Feature Architecture

| Index | Feature Name | Source Image Object | Operations & Pipeline Reference |
|:---:|:---|:---|:---|
| **0** | `meanIntensity` | **Enhanced working image crop (`workingRoi`)** | Grayscale conversion (`GrayscaleConverter.toGrayscale`) -> $5\times 5$ Gaussian smoothing (`FilterProcessor.applyGaussianFilter`) -> average luminance $[0..255]$. |
| **1** | `foregroundRatio` | **Enhanced working image crop (`workingRoi`)** | Otsu/Adaptive binary thresholding (`ThresholdProcessor.segmentForeground`) from smoothed grayscale & HSV saturation map -> foreground pixel fraction $[0..1]$. |
| **2** | `darkPixelRatio` | **Enhanced working image crop (`workingRoi`)** | Fraction of pixels in smoothed grayscale crop with intensity $< 85$. |
| **3** | `edgeDensity` | **Enhanced working image crop (`workingRoi`)** | $3\times 3$ Sobel gradient filter (`EdgeDetector.applySobel`) -> fraction of edge pixels exceeding gradient threshold $> 30$. |
| **4** | `objectArea` | **Enhanced working image crop (`workingRoi`)** | Binary foreground map after $3\times 3$ morphological closing (`MorphologyProcessor.close`) -> foreground area ratio $[0..1]$. |
| **5** | `meanSaturation` | **Enhanced working image crop (`workingRoi`)** | HSV color space conversion (`ColorSpaceConverter.toSaturationMap`) -> average saturation $[0..255]$. |
| **6** | `saturationRatio` | **Enhanced working image crop (`workingRoi`)** | Fraction of pixels with HSV saturation $S > 90$ (chromatic indicator). |
| **7** | `largestBlobRatio` | **Enhanced working image crop (`workingRoi`)** | Connected-component labeling (`ConnectedComponentAnalyzer.analyze`) -> area ratio of largest 8-connected blob to slot ROI area $[0..1]$. |
| **8** | `vehicleOverlapScore` | **Original scene photograph + ONNX boxes** | Intersection-over-Slot ($\mathrm{IoS}$) ratio of detected vehicle bounding box overlapping slot quad/rect geometry $[0..1]$. Evaluated on original photograph to avoid degrading detector confidence. |
| **9** | `vehicleConfidenceScore` | **Original scene photograph + ONNX boxes** | Peak model detection confidence score ($0.00$ to $1.00$) among candidate vehicles overlapping the slot ROI. |

---

## 2. Caller Inventory: Feature Extraction Call Sites

All production and verification paths invoke the unified feature extraction pipeline adhering to `CANONICAL_SCHEMA_ID`:

1. **Interactive Live Analysis & Slot Review**:
   - Class: [`ParkingAnalysisService`](file:///C:/Users/ojas/OneDrive/Desktop/ParkNChill_v2.3_Simplified/src/main/java/com/parknchillai/app/ParkingAnalysisService.java)
   - Behavior: Preprocesses active working image (enhanced) for IVP feature extraction; runs ONNX detection on the original unenhanced photo; computes all 10 features synchronously or in background task.
2. **PKLot Dataset Importer**:
   - Class: [`PkLotDatasetImporter`](file:///C:/Users/ojas/OneDrive/Desktop/ParkNChill_v2.3_Simplified/src/main/java/com/parknchillai/ai/PkLotDatasetImporter.java)
   - Behavior: Applies `ivp-enhance-v1` enhancement to slot crops before extracting the 8 IVP features, and runs ONNX detector passes for the 2 vehicle features.
3. **Interactive IVP Stages Visualizer**:
   - Class: [`IvpStagesDialog`](file:///C:/Users/ojas/OneDrive/Desktop/ParkNChill_v2.3_Simplified/src/main/java/com/parknchillai/ui/IvpStagesDialog.java)
   - Behavior: Displays intermediate filters (gray, smoothed, binary, closed, edges, saturation, largest blob) derived from the exact same enhanced pixels used for feature calculation.

---

## 3. Strict Schema and Policy Enforcement

- **Model Serialization (`.pncmodel`)**: Serializes `schemaVersion=v2.3-10feat-enhanced-v1` and `policyId=ivp-enhance-v1`.
- **Fail-Closed Rejection**: Attempting to load an incompatible schema version or mismatched policy ID throws [`SchemaMismatchException`](file:///C:/Users/ojas/OneDrive/Desktop/ParkNChill_v2.3_Simplified/src/main/java/com/parknchillai/ai/SchemaMismatchException.java).
- **Legacy Quarantine**: Legacy 8-feature models (`v2.2-8feat`) are segregated in [`LegacyDecisionTreeContainer`](file:///C:/Users/ojas/OneDrive/Desktop/ParkNChill_v2.3_Simplified/src/main/java/com/parknchillai/ai/LegacyDecisionTreeContainer.java) with explicit UI explanations and never silently mixed with 10-feature models.
