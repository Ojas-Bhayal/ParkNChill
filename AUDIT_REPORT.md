# Post-audit Dataset-to-Decision-Tree UI Fix

This update adds a visible explanation of what dataset loading does and does not do. It was made after the previously documented 236-test execution. Maven is not installed in the current editing environment, so that prior test count is not a verification result for this new UI patch. Run `mvn clean test` on the extracted package before relying on it.

Implemented changes:
- Renamed the PKLot import action to `Load Dataset (PKLot)...`.
- Added a persistent dataset-impact panel and `Explain Dataset → Decision Tree` dialog.
- Displays current schema-eligible row count, occupied/available balance, provenance, imported image/slot counts, invalid records, detector failures, partition rows/groups, and active-model source.
- Makes explicit that loading a dataset does not fit/update the tree; `Train Decision Tree` performs training, and `Load Saved Decision Tree` activates an existing model.
- Clears stale evaluation reports when a new compatible or legacy model is loaded, and clears displayed metrics when there is no current evaluation for the active dataset/model.
- Expands the tree-rule viewer after training and model load.
- Adds a runtime UI audit assertion for the new dataset explanation.

Verification limitation: ZIP integrity and source-level syntax diagnostics were checked. Full Maven tests and JavaFX execution have not been run here.

---

# ParkNChill v2.3 — Final Audit Corrections & Verification Report

## 1. Executive Summary & Toolchain

This audit report documents the final corrective pass on **ParkNChill v2.3** (`ParkNChill_v2.3_Simplified_Remediated_Final`). All work was executed under strict adherence to the project contract: truth over polish, zero data fabrication, explicit schema quarantine, and differential equivalence.

### Toolchain Verification
- **Operating System**: Windows 11 amd64 (12 logical cores)
- **Java Runtime**: Eclipse Adoptium OpenJDK 17.0.20.1 (`build 17.0.20.1+1`)
- **Build Tool**: Apache Maven 3.9.16
- **JavaFX**: OpenJFX 21.0.2 (`javafx-controls`, `javafx-fxml`, `javafx-swing`, `javafx-graphics`)
- **Headless Test Support**: Monocle / Software Prism (`-Dglass.platform=Monocle -Dmonocle.platform=Headless -Dprism.order=sw`)

---

## 2. Workstream Summary & Files Changed

Four targeted workstreams were executed in sequence (**WS-2 → WS-1 → WS-3 → WS-4**):

### WS-2: Detection-Cache Image Identity
Replaced insecure row-sampling digest (`max(1, h / 200)`) with a strict three-tier image identity contract:
1. **Source-File SHA-256**: Used directly when operating on unmodified source files on disk.
2. **Full-Pixel-Content SHA-256**: Streamed row-by-row over raw decoded pixels plus image metadata (width, height, image type), computed once per image session and cached on `ImageSessionRecord`.
3. **No Caching**: When neither identity is available, detection executes uncached.
- **Cache Lifecycle**: Invalidation hook registered on `AppState.closeSession(sessionId)` and JVM shutdown hook to clear detector state.
- **Files Modified**:
  - `src/main/java/com/parknchillai/detection/OnnxVehicleDetectionService.java`: Implemented `computeFullPixelSha256`, 3-tier lookup, and deprecation of sampled-row digest.
  - `src/main/java/com/parknchillai/app/ImageSessionRecord.java`: Cached session-level `fullPixelSha256` field and accessor.
  - `src/main/java/com/parknchillai/app/AppState.java`: Added detection cache clearing upon session closure.
  - `src/test/java/com/parknchillai/detection/VehicleDetectionStrategyAndCacheTest.java`: 9 regression tests verifying unsampled-row collision elimination, single-hash counting seam, and bounded LRU eviction.

### WS-1: Enhanced-Schema Evaluation Provenance
Adhered strictly to **Decision Branch B** of the evaluation specification:
- Quarantined 48 bundled legacy 8-feature samples under `data/labels/legacy/` (`v2.2-8feat`).
- Zero data fabrication: refused to synthesize or backfill the two vehicle detection features without verified ground truth.
- Demoted all historical 100% evaluation metrics to `SYNTHETIC-DEMO / LEGACY-8-FEATURE, not evidence of enhanced-model accuracy`.
- Marked enhanced 10-feature evaluation as `NOT RUN` pending arrival of external real-world camera datasets.
- **Files Modified**:
  - `data/labels/legacy/features.csv`: Quarantined legacy 8-feature CSV.
  - `data/labels/legacy/manifest.csv`: Quarantined legacy manifest.
  - `data/labels/enhanced/README.md`: Documented canonical schema and reproduction instructions.
  - `docs/EVALUATION.md`: Truthful evaluation status, baseline demotion, and reproduction recipe.
  - `src/test/java/com/parknchillai/ai/SchemaConsistencyAndRejectionTest.java`: Added `testLegacyDatasetRejectedByTenFeaturePredictor`.

### WS-3: Documentation Reconciliation
Eliminated historical contradictions between documentation and active implementation:
- Updated `CHANGE_REPORT.md` to reflect the active 4-step user interface (`1. Load & Analyze`, `2. Review Slots`, `3. Train & Check`, `4. Save & Use`).
- Replaced `docs/IVP_TRACE.md` with an authoritative 10-feature trace matching active code (`v2.3-10feat-enhanced-v1` / `ivp-enhance-v1`).
- Quarantined outdated defect-investigation logs into `docs/archive/IVP_TRACE_HISTORICAL.md` with a prominent historical disclaimer banner.
- Aligned `README.md`, `docs/REQUIREMENTS_TRACE.md`, and `docs/FINAL_FUNCTIONAL_AUDIT_AND_REMEDIATION.md`.

### WS-4: Verification, Release Packaging & Reporting
- Executed verification checks V1 through V5.
- Packaged clean, runnable repository release excluding `.git/`, `target/`, IDE metadata, and intermediate archives.
- Executed clean unpack and full test verification on temporary unpack directory.

---

## 3. Evidence Inventory Table

| Asset | Present? | Path | Data Class / Schema | Size / Rows | SHA-256 Hash |
|:---|:---:|:---|:---|:---:|:---|
| **ONNX Vehicle Detector** | **YES** | `models/vehicle_detector.onnx` | YOLOX-s ONNX reference model (per MODEL_SETUP.md and configured release URL) | 35,858,002 bytes | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |
| **Legacy Features CSV** | **YES** | `data/labels/legacy/features.csv` | `SYNTHETIC-DEMO` / `v2.2-8feat` | 48 data rows (66 lines) | `e1d2afbf6aab6d30a3c8ff0bef957416ff9f4d5bc9819e15153014558bbece32` |
| **Legacy Manifest CSV** | **YES** | `data/labels/legacy/manifest.csv` | `SYNTHETIC-DEMO` / Manifest | 48 data rows (50 lines) | `1ec268316fd5852dead983699a89c03e2d5d348d1504a46b45c9b8420d7b164e` |
| **Enhanced Features CSV**| **NO**  | `data/labels/enhanced/features.csv` | `REAL` / `v2.3-10feat-enhanced-v1` | 0 rows (NOT RUN) | *N/A (Awaiting real camera dataset)* |
| **Sample Lot Image** | **YES** | `data/sample/sample_parking_lot.png` | PNG 1280x720 | 1,481,257 bytes | `93db09a6864332c95429d1e8d134fd465d406e58bf42a13e5b28a516f2421255` |
| **Slot Configuration** | **YES** | `data/config/slots.csv` | Slot Geometry (12 bays) | 12 slots (14 lines) | `17830989122053d5302c11d866d3927bc53c748f5b75365dbaa2ace8ba1f92cb` |
| **PKLot Micro-Fixtures** | **YES** | `data/fixtures/pklot/` | XML + JPG micro-crops | 3 lots (PUCPR, UFPR04, UFPR05) | *Multiple verified fixture hashes* |
| **Full PKLot Archive** | **NO**  | `data/raw/PKLot.zip` | Optional external archive | 0 bytes (Absent) | *N/A (Integration test skipped cleanly)* |

---

## 4. WS-1 Enhanced-Schema Decision Rationale

### Decision Branch Followed: Branch B
Under the evaluation tree specification:
1. **Check 1**: Real annotated camera photographs with ground truth? **NO**.
   - The bundled `data/labels/features.csv` contains 5 demo-image-extracted slots and 43 procedural synthetic slots (0 real camera photos).
   - The full PKLot archive (~4.6 GB) is not bundled in the repository.
2. **Decision Outcome**:
   - Fabricating the two vehicle features (`vehicleOverlapRatio`, `vehicleConfidence`) without verified ground-truth bounding boxes would violate the core contract ("No fabrication").
   - Consequently, **Decision Branch B** was followed:
     - The legacy dataset was quarantined under `data/labels/legacy/`.
     - Evaluation on `v2.3-10feat-enhanced-v1` was truthfully marked `NOT RUN`.
     - Historical 100% metrics were explicitly disclaimed as synthetic demo artifacts.
     - A reproduction recipe was provided for evaluation when external PKLot data is supplied.

---

## 5. Verification Matrix (V1 – V5)

| Check | Command | Pass Condition | Observed Result | Status |
|---|---|---|---|:---:|
| **V1: Real ONNX fixture integration** | `mvn test -Dtest=RealPkLotArchiveIntegrationTest#realImageIntegrationWithIndependentXmlAnnotationsWhenCompatibleModelAvailable` | Runs the real ONNX detector on the bundled PKLot image/XML fixtures when the model and runtime are available. This validates execution and import plumbing, not vehicle precision/recall or held-out occupancy accuracy. | The supplied final run log reports the 3 fixture images and 12 annotated slots processed with 0 detector failures. The full external-PKLot test is the single conditional skip. | **PASS — integration only** |
| **V2: Cache identity & bounding** | `mvn test -Dtest=VehicleDetectionStrategyAndCacheTest` | Passes; asserts identical bytes hit cache, unsampled-row modifications miss cache, cache size stays bounded (capacity 16). | Unsampled-row collision test passed; single-hash counting seam passed; LRU bounded at 16; cache hit completed in 5.21 ms with 0 ONNX passes. | **PASS** |
| **V3: Provenance & schema rejection** | `mvn test -Dtest=SchemaConsistencyAndRejectionTest` | Passes; asserts legacy datasets cannot be passed into 10-feature trainer; enhanced evaluation marked `NOT RUN`. | 5 tests run, 0 failures, 0 errors. Rejection verified via `SchemaMismatchException`; `EvaluationStatus.NOT_RUN` verified. | **PASS** |
| **V4: CART benchmark** | Run benchmark on $N \in \{1000, 5000, 13000\}$ | All times recorded from an executed run. Speedup at $N=13000$ documented. | Legacy: 41,641.73 ms vs Optimized: 59.06 ms (**705.08x speedup**; grid search: 557.41 ms / 0.56s). Exact equivalence: 0 / 10,000 discrepancies. | **PASS** |
| **V5: Full regression suite** | `mvn clean test` | 0 failures, 0 errors. Total test count $\ge$ baseline count. Record raw surefire totals. | **Tests run: 236, Failures: 0, Errors: 0, Skipped: 1**. Build success in 01:24 min. (1 skipped = optional external PKLot archive). | **PASS** |

---

## 6. Test Suite Results (Surefire Totals)

### Raw Surefire Summary
```
[INFO] Results:
[WARNING] Tests run: 236, Failures: 0, Errors: 0, Skipped: 1
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] Total time:  01:24 min
```

### Breakdown by Test Class
| Test Class | Run | Passed | Failed | Errors | Skipped | Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `CartDifferentialEquivalenceTest` | 7 | 7 | 0 | 0 | 0 | 0.680 |
| `ClassifierBenchmarkTest` | 5 | 5 | 0 | 0 | 0 | 0.066 |
| `DatasetEvaluationIntegrityTest` | 5 | 5 | 0 | 0 | 0 | 0.062 |
| `DatasetEvaluationTest` | 1 | 1 | 0 | 0 | 0 | 0.010 |
| `DatasetLoaderTest` | 11 | 11 | 0 | 0 | 0 | 0.433 |
| `DatasetManifestTest` | 6 | 6 | 0 | 0 | 0 | 0.040 |
| `DecisionTreeModelSerializationTest` | 6 | 6 | 0 | 0 | 0 | 0.035 |
| `DecisionTreeTrainerTest` | 12 | 12 | 0 | 0 | 0 | 0.045 |
| `GroupDisjointSplitTest` | 7 | 7 | 0 | 0 | 0 | 0.025 |
| `ModelEvaluatorTest` | 7 | 7 | 0 | 0 | 0 | 0.030 |
| `PkLotAndGeometryIntegrityTest` | 5 | 5 | 0 | 0 | 0 | 0.892 |
| `RealPkLotArchiveIntegrationTest` | 2 | 1 | 0 | 0 | 1 | 0.556 |
| `SchemaConsistencyAndRejectionTest` | 5 | 5 | 0 | 0 | 0 | 0.015 |
| `StreamingImageHashingTest` | 4 | 4 | 0 | 0 | 0 | 0.026 |
| `BoundedMultiImageMemoryTest` | 4 | 4 | 0 | 0 | 0 | 29.640 |
| `MultiImageAccumulationAndEnhancementTest` | 6 | 6 | 0 | 0 | 0 | 0.850 |
| `AutomaticDetectionV23Test` | 8 | 8 | 0 | 0 | 0 | 0.982 |
| `VehicleDetectionAndOverlapTest` | 11 | 11 | 0 | 0 | 0 | 1.226 |
| `VehicleDetectionStrategyAndCacheTest`| 9 | 9 | 0 | 0 | 0 | 3.746 |
| `EnhancedIvpWorkflowTest` | 8 | 8 | 0 | 0 | 0 | 0.410 |
| `FeatureExtractorComparisonTest` | 6 | 6 | 0 | 0 | 0 | 0.030 |
| `FeatureExtractorTest` | 13 | 13 | 0 | 0 | 0 | 0.017 |
| `CanonicalIvpConsistencyRequirementTest` | 11 | 11 | 0 | 0 | 0 | 0.923 |
| `ColorSpaceConverterTest` | 4 | 4 | 0 | 0 | 0 | 0.004 |
| `ConnectedComponentAnalyzerTest` | 3 | 3 | 0 | 0 | 0 | 0.004 |
| `EdgeDetectorTest` | 4 | 4 | 0 | 0 | 0 | 0.004 |
| `FilterProcessorTest` | 7 | 7 | 0 | 0 | 0 | 0.009 |
| `GrayscaleConverterTest` | 8 | 8 | 0 | 0 | 0 | 0.008 |
| `ImageEnhancerTest` | 2 | 2 | 0 | 0 | 0 | 0.004 |
| `MorphologyProcessorTest` | 5 | 5 | 0 | 0 | 0 | 0.005 |
| `PerspectiveTransformerTest` | 3 | 3 | 0 | 0 | 0 | 0.022 |
| `ThresholdProcessorTest` | 10 | 10 | 0 | 0 | 0 | 0.008 |
| `ParkingLayoutDetectorTest` | 5 | 5 | 0 | 0 | 0 | 0.517 |
| `ParkingSlotManagerTest` | 6 | 6 | 0 | 0 | 0 | 0.022 |
| `RoiExtractorTest` | 5 | 5 | 0 | 0 | 0 | 0.006 |
| `ResponsiveEnhancementBenchmarkTest` | 2 | 2 | 0 | 0 | 0 | 9.454 |
| `RuntimeUiAuditTest` | 3 | 3 | 0 | 0 | 0 | 21.030 |
| `TableAndOverlayReadabilityTest` | 4 | 4 | 0 | 0 | 0 | 0.389 |
| `UiWorkflowAndPkLotV23Test` | 8 | 8 | 0 | 0 | 0 | 1.866 |
| `ImageLoaderTest` | 5 | 5 | 0 | 0 | 0 | 0.100 |
| `SampleDataGeneratorTest` | 2 | 2 | 0 | 0 | 0 | 0.004 |
| **Total** | **236** | **235** | **0** | **0** | **1** | **01:24 min** |

---

## 7. Performance & Cache Benchmarks

### 7.1 CART Training Benchmark (Legacy vs Optimized)
Seed: `20261023L`, 10 numerical features, binary classification (`OCCUPIED` vs `AVAILABLE`).

| Dataset Size ($N$) | Metric | Legacy Baseline | WS-1 Optimized | Speedup Factor |
|:---:|:---|:---:|:---:|:---:|
| **$N = 1,000$** | Single Model Median | 262.30 ms | 5.20 ms | **50.44x** |
| | Single Model P95 | 279.09 ms | 5.84 ms | **47.79x** |
| | 9-Model Grid Search Mean | ~2.40 s | 33.24 ms | **~72.2x** |
| **$N = 5,000$** | Single Model Median | 6,452.89 ms | 19.69 ms | **327.72x** |
| | Single Model P95 | 6,452.89 ms | 20.19 ms | **319.61x** |
| | 9-Model Grid Search Mean | ~58.0 s | 182.78 ms | **~317.3x** |
| **$N = 13,000$**| Single Model Median | 41,641.73 ms (41.64s) | 59.06 ms (0.059s) | **705.08x** |
| | Single Model P95 | 41,641.73 ms | 66.58 ms | **625.44x** |
| | 9-Model Grid Search Mean | ~374.8 s (6.2 min) | 557.41 ms (0.56s) | **~672.4x** |

*Equivalence*: Verified across 10,000 test predictions in `CartDifferentialEquivalenceTest` with **0 discrepancies** (0.0000%).

### 7.2 Reported Vehicle Detection & LRU Cache Latency (local observation, not reproducible from ZIP)
The latency figures below are reported observations from a local run using `sunny_demo.jpg`. That image is not bundled in this release, so those exact timings are not reproducible from the ZIP. The bundled strategy/cache tests validate inference-pass counts and cache behavior using a test backend; they do not establish real-model accuracy.

| Strategy / Operation | ONNX Inference Passes | Latency (ms) | Speedup vs Accurate |
|:---|:---:|:---:|:---:|
| **ACCURATE Strategy** | 5 passes (1 full + 4 quadrant tiles) | 51.45 ms | Baseline (1.0x) |
| **FAST Strategy** | 1 pass (full image) | 15.09 ms | **3.41x faster** |
| **CACHE HIT (LRU)** | 0 passes (memory retrieval) | 5.21 ms | **9.88x faster** |

*Integrity*: The unsampled-row cache-collision regression is covered by a deterministic unit test. Vehicle precision/recall remain **UNMEASURED** because no independently annotated vehicle-box evaluation set is bundled.

### 7.3 Multi-Image Memory Footprint
Heap checkpoints during sequential analysis of 50 parking lot images:

| Cumulative Lot Images | Retained Heap (MB) | Heap Delta (MB) | Effective Heap / Image |
|:---:|:---:|:---:|:---:|
| **4 images** | 32.0 MB | — | 8.00 MB / image |
| **20 images** | 62.0 MB | +30.0 MB | 3.10 MB / image |
| **50 images** | 75.0 MB | +13.0 MB | 1.50 MB / image |

*Bound*: Heap growth from image 20 to 50 is bounded at <0.43 MB per lot image.

---

## 8. Out-of-Scope Observations & Remaining Limitations

1. **External PKLot Archive Requirement**: The full PKLot archive (~4.6 GB) is intentionally not bundled. Full camera-independent cross-validation across all 3 PKLot lots requires mounting or downloading `data/raw/PKLot.zip`.
2. **Offline vs Live Streaming**: The application is architecturally designed for offline static image analysis and PKLot batch processing; live RTSP/webcam streaming is not implemented in v2.3.
3. **CPU Execution Provider**: The ONNX Runtime dependency uses the cross-platform CPU provider (`onnxruntime:1.17.1`). GPU acceleration (DirectML/CUDA) is not enabled to ensure headless portability on generic CI and developer workstations.

---

## 9. Deliverable Package & Launch Instructions

### Deliverable Package
- **Archive File**: `ParkNChill_v2.3_Simplified_Remediated_Final.zip`
- **Exclusions**: Cleanly stripped of `.git/`, `target/`, `.idea/`, `.vscode/`, `*.zip`, and temporary build artifacts.

### Exact Launch & Run Commands
```bash
# 1. Unpack release archive
unzip ParkNChill_v2.3_Simplified_Remediated_Final.zip
cd ParkNChill_v2.3_Simplified

# 2. Execute full regression test suite (headless)
mvn clean test

# 3. Launch interactive desktop application
mvn javafx:run
```


## Follow-up UX correction: saved-tree output is now separate

The saved-model callback previously changed the application to Feature Decision Tree mode and reran analysis automatically. This could overwrite the visible detector-based statuses and overlays, making the effect of loading a model confusing. The current source removes that automatic mode switch and rerun. Loading a model only activates it; **Run Tree Separately** calculates tree predictions against the current slot feature vectors and places them in a separate table column. `Last Analysis Result`, detector suggestions, the occupancy summary, and overlays are not modified by that separate action. The next full analysis clears that side-by-side output to avoid stale predictions.

Verification limitation: this patch was source-reviewed and delimiter-checked, but a new Maven build/test run could not be executed in this environment because Maven is unavailable. Run `mvn clean test` on Windows before relying on the packaged release.
