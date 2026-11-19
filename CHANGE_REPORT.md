# ParkNChill v2.3 — Remediation Change Report

## 1. Executive Summary & Toolchain Details

A surgical performance, memory, schema, and evaluation-integrity remediation was conducted on **ParkNChill v2.3** across seven core workstreams (**WS-6 → WS-5 → WS-7 → WS-1 → WS-4 → WS-3 → WS-2**), followed by a final audit corrections pass (provenance truth, detection-cache full-pixel identity, and documentation alignment).

The four-step workflow UI (`1. Load & Analyze`, `2. Review Slots`, `3. Train & Check`, `4. Save & Use`), 10 IVP/vehicle features, CART algorithm, and ONNX detection pipeline have been preserved while eliminating performance bottlenecks, memory leaks, silent schema mixing, and non-independent evaluation claims.

### Toolchain Verification
- **JDK**: Eclipse Adoptium OpenJDK 17.0.20.1 (Temurin-17.0.20.1+1, 64-Bit Server VM)
- **Build Tool**: Apache Maven 3.9.16
- **JavaFX**: OpenJFX 21.0.2 (controls, fxml, swing, graphics)
- **Headless Execution**: Verified via `-Dglass.platform=Monocle -Dmonocle.platform=Headless -Dprism.order=sw` and JavaFX AWT/Swing interop.

---

## 2. Gate Verification Summary (G1 – G10)

| Gate | Requirement | Status | Evidence / Metrics |
|:---|:---|:---:|:---|
| **G1** | Baseline vs Final Test Pass Rate | **PASS** | Baseline: 200 run (199 pass, 0 fail, 1 skip). Intermediate: 231 run (230 pass, 0 fail, 1 skip). Final: Verified with zero regressions. |
| **G2** | CART Benchmark & Equivalence | **PASS** | N=13,000: 41.64s → 59.06 ms (**705.08x speedup**; 9-model grid search: approximately 374.8s → 0.56s, **~672x**). Discrepancies vs baseline: **0.0000%** (0 / 10,000), according to the final benchmark report. |
| **G3** | Bounded Multi-Image Memory | **PASS** | Retained heap: 4 img = 32 MB, 20 img = 62 MB, 50 img = 75 MB. Growth from 20 to 50 images is bounded at 13 MB (<0.43 MB/img). Unpinned subimage AWT rasters; on-demand IVP masks. |
| **G4** | Schema Consistency & Rejection | **PASS** | Canonical IDs: `v2.3-10feat-enhanced-v1` & `ivp-enhance-v1`. Incompatible schemas/policies throw explicit `SchemaMismatchException`. Legacy 8-feature models isolated with explicit warning. |
| **G5** | Group-Disjoint Evaluation Integrity | **PASS** | Sessions/cameras never split across partitions. When < 3 distinct groups exist, tags `INSUFFICIENT_GROUPS / NON_INDEPENDENT` and suppresses false "camera-independent" UI badges. |
| **G6** | Responsive Enhancement Preview | **PASS** | 150 ms debounce via `PauseTransition`, monotonic `AtomicLong previewGeneration` discards stale frames. 4000x3000 (12 MP): downscaled preview **36.89 ms** vs full Apply 1434.46 ms (**38.9x speedup**). |
| **G7** | Vehicle-Detection Cost Reduction | **PASS** | Reported local ONNX observation: `FAST` (1 pass, 15.09 ms), `ACCURATE` (5 passes, 51.45 ms), LRU cache hit (0 passes, 5.21 ms). These timings depend on the local model/image/hardware and are not accuracy metrics; the bundled strategy/cache unit tests use a test backend. |
| **G8** | Asset Availability Table | **PASS** | Complete table produced below. Verified real ONNX model SHA-256 and fixture paths. |
| **G9** | Concise Change Report | **PASS** | Complete documentation in `CHANGE_REPORT.md` and `AUDIT_REPORT.md`. |
| **G10** | Clean Runnable ZIP Package | **PASS** | Packaged and extracted cleanly. The supplied final execution log reports `mvn clean test` success; an interactive desktop launch is not independently evidenced by the supplied log and should be verified locally with `mvn javafx:run`. |

---

## 3. Asset Availability Table (G8)

| Asset | Present? | Path | SHA-256 Hash | Notes |
|:---|:---:|:---|:---|:---|
| **ONNX Vehicle Detector** | **YES** | `models/vehicle_detector.onnx` | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` | 35,858,002 bytes (~36 MB). Valid ONNX ModelProto format. |
| **Sample Parking Lot Image** | **YES** | `data/sample/sample_parking_lot.png` | `93db09a6864332c95429d1e8d134fd465d406e58bf42a13e5b28a516f2421255` | Synthetic 2D/3D render demo lot (12 slots) for UI walkthrough. |
| **Slot Configuration CSV** | **YES** | `data/config/slots.csv` | `6256f6c2fbe5c57b98dcf62ca010f34086ea0ec201b1a77e1dfd4681f9b36262` | 12 ground-truth slot coordinates and bays for demo lot. |
| **Legacy Features CSV** | **YES** | `data/labels/legacy/features.csv` | `e1d2afbf6aab6d30a3c8ff0bef957416ff9f4d5bc9819e15153014558bbece32` | 48 samples (5 demo-image-extracted + 43 procedural synthetic), schema `v2.2-8feat`. |
| **Legacy Manifest CSV** | **YES** | `data/labels/legacy/manifest.csv` | `1ec268316fd5852dead983699a89c03e2d5d348d1504a46b45c9b8420d7b164e` | 22-column companion manifest for legacy 48 samples. |
| **PKLot Micro-Fixtures** | **YES** | `data/fixtures/pklot/{PUCPR,UFPR04,UFPR05}` | *Multiple XML & JPG fixtures* | 3 individual XML/JPG pairs for camera-independence unit tests. |
| **Full PKLot ZIP Archive** | **NO** | `data/raw/PKLot.zip` (optional) | *N/A (External)* | External archive (~4.6 GB). Skipped in test suite when absent. |

*Evaluation Notes*:
- **Enhanced 10-Feature Evaluation**: **NOT RUN** — The complete external PKLot archive is not bundled. The 48 bundled samples are `v2.2-8feat` legacy samples and cannot be converted to 10-feature rows without fabricating the 2 vehicle features.
- **Vehicle-Detection Precision/Recall**: **UNMEASURED** — No independently annotated vehicle bounding-box ground truth dataset was supplied with the repository. Latency and ONNX pass counts were measured directly on lot imagery.

---

## 4. Workstream Details

### 4.1 WS-2: Detection-Cache Image Identity
Replaced row-sampling hash (`rowStep = max(1, h / 200)`) with a strict three-tier identity resolution policy:
1. **Verified source-file SHA-256**: Used directly if image pixels are exact unmodified decoded source (not resized, not warped).
2. **Full-pixel-content SHA-256**: Streamed row-by-row over raw decoded pixels plus width, height, and format without full-image copying. Computed once per image session and cached on `ImageSessionRecord`.
3. **No caching**: If neither identity is available.

#### Call Site Table:
| Call Site | Location | Available Identity | Decision |
|:---|:---|:---|:---|
| **Interactive Lot Analysis** | `ParkingAnalysisService.java:710` | Source-file SHA-256 or session full-pixel SHA-256 | Passed into `detectVehicles` |
| **PKLot Importer** | `PkLotDatasetImporter.java:348` | Source-file SHA-256 | Passed into `detectVehicles` |
| **Unit & Strategy Tests** | `VehicleDetectionStrategyAndCacheTest.java` | Explicit or full-pixel SHA-256 | Tested both explicit and computed |

- **Collision Elimination**: Verified that images differing only on unsampled rows produce distinct full-pixel hashes and do not share cached detections.
- **Counting Seam**: Verified that exactly one full-pixel hash is computed per session record across repeated detections.

---

### 4.2 WS-1: Enhanced-Schema Evaluation Provenance (Decision B)
- Following the literal decision tree, since external PKLot data is not bundled, **Decision Branch B** was followed:
  - Preserved legacy 48 samples under `data/labels/legacy/` (`v2.2-8feat`).
  - Zero fabrication: did not convert 8-feature rows into 10-feature rows.
  - Relabeled any historical 100% figures in `docs/EVALUATION.md` as `SYNTHETIC-DEMO / LEGACY-8-FEATURE, not evidence of enhanced-model accuracy`.
  - Added reproduction recipe for users with external PKLot data.
  - Added regression test verifying that legacy 8-feature datasets are rejected by 10-feature predictors.

---

### 4.3 WS-1 CART Trainer Optimization
- Replaced brute-force $O(F \cdot n^2)$ candidate evaluation with $O(F \cdot n \log n)$ in-place sort using primitive index arrays (`int[] indices`).
- Accumulator running counters update Gini impurity in $O(1)$ time per candidate with zero object allocations.
- Equivalence verified in `CartDifferentialEquivalenceTest`: 0 discrepancies across 10,000 predictions (0.0000%).
- Final reported benchmark: 50.44x at N=1,000; 327.72x at N=5,000; **705.08x at N=13,000** (59.06 ms vs 41.64s). Results are from the final audit run and are hardware/runtime dependent.

---

### 4.4 WS-4 Bounded Multi-Image Memory
- Decoupled AWT raster sharing in `RoiExtractor` by drawing into standalone rasters, fixing parent `DataBuffer` memory leaks.
- Made intermediate IVP masks on-demand via `SoftReference<ProcessedRoi>`.
- Added LRU session cache (capacity 3) in `AppState`.
- Retained heap: 4 img = 32 MB, 20 img = 62 MB, 50 img = 75 MB. Growth bounded at <0.43 MB/img.

---

### 4.5 WS-3 Responsive Enhancement Preview
- 150 ms debounce on sliders in `ImageEnhancementDialog`.
- Downscaled interactive preview (max 640x480).
- Monotonic generation counter background daemon worker discarding stale frames.
- 4000x3000 (12 MP): downscaled preview 36.89 ms vs full Apply 1434.46 ms (**38.9x speedup**).

---

## 5. Deliverable & Instructions

- **Deliverable Package**: `ParkNChill_v2.3_Simplified_Remediated_Final.zip`
- **Exclusions**: Cleaned of `target/`, `.git/`, and IDE metadata.

### How to Run
```bash
# 1. Unzip
unzip ParkNChill_v2.3_Simplified_Remediated_Final.zip
cd ParkNChill_v2.3_Simplified

# 2. Run full test suite
mvn clean test

# 3. Launch application
mvn javafx:run
```


## Dataset-to-Decision-Tree visibility correction

- Renamed the PKLot action to **Load Dataset (PKLot)...** and clarified that loading prepares labelled, schema-compatible slot rows; it does not train or change the active tree.
- Added a persistent dataset-impact card and **Explain Dataset → Decision Tree** dialog showing current source, eligible records, class balance, provenance, imported images/slots, invalid records, detector failures, TRAIN/VALIDATION/TEST row and group counts, active model source, and the next action.
- Renamed the main count to **Eligible Samples (all partitions)** so validation/test rows are not mistaken for training rows.
- The training button now shows an explicit in-progress state, expands the current tree-rule view after training, and reports a visible failure state.
- Loading a saved tree clears the previous in-memory evaluation report; otherwise old model metrics could appear to describe the newly loaded model. The active tree rules are shown in the advanced model section after a successful train/load.
- Added `docs/DATASET_TO_DECISION_TREE.md` with the step-by-step relationship between loading data, training, saving, loading a model, and predictions.


## Loaded Decision Tree output separation (current patch)

- Loading or training a compatible tree no longer changes the primary analysis mode or automatically reruns the detector pipeline. Existing image overlays, last-analysis status values, and occupancy summary remain untouched.
- Added **Run Tree Separately**. It applies the active compatible ten-feature tree to the feature vectors already calculated for the current image and displays the output in a distinct **Loaded Tree Prediction** column, including leaf purity.
- Renamed the former **Current Status** column to **Last Analysis Result** to make clear that it belongs to the most recent full analysis run.
- Selecting a slot shows the separate loaded-tree result and rule path. A new analysis clears the separate predictions because its input features may have changed; the user can rerun the tree explicitly.
- Added a regression test asserting that loading a model does not switch the main analysis mode or replace the current analysis summary.
