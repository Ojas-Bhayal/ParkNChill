# ParkNChill v2.3 — Empirical Evaluation & Provenance Audit Report

## 1. Executive Summary & Evaluation Status

| Evaluation Scope | Schema ID | Enhancement Policy | Status | Result / Notes |
|:---|:---|:---|:---:|:---|
| **Enhanced 10-Feature Model** | `v2.3-10feat-enhanced-v1` | `ivp-enhance-v1` | **NOT RUN** | **Requires external annotated PKLot archive.** Cannot be fabricated from legacy 8-feature data because the 2 vehicle-detection features (`vehicleOverlapScore`, `vehicleConfidenceScore`) require live ONNX detector passes over real lot imagery with ground truth. |
| **Legacy 8-Feature Baseline** | `v2.2-8feat` | *None (Raw)* | **DEMO ONLY** | **SYNTHETIC-DEMO / LEGACY-8-FEATURE, not evidence of enhanced-model accuracy.** Describes 48 legacy 8-feature rows: 5 extracted from `sample_parking_lot.png` + 43 procedural Java 2D calibration renders. |
| **Vehicle Detection Precision/Recall** | *YOLOX-s ONNX* | `YOLOX_BGR_255` | **UNMEASURED** | **No independently annotated vehicle bounding-box test set was bundled.** Real ONNX latency and pass counts are verified, but mAP/precision/recall are unmeasured. |

> [!IMPORTANT]
> **Provenance & Integrity Standard**: Previous reports that quoted "100% accuracy, precision, recall, and F1-score on enhanced 10-feature models" were evaluating an 8-feature decision tree on a 48-sample synthetic/procedural dataset and mislabeling the output. In strict adherence to the project contract, those figures are downgraded to **`NOT RUN`** for the enhanced 10-feature schema, and **`SYNTHETIC-DEMO / LEGACY-8-FEATURE`** for the 48-sample baseline corpus.

---

## 2. Bundled Corpus Evidence Inventory

The files currently residing in `data/labels/` (and archived under `data/labels/legacy/`) contain:
- **Total Samples**: 48 rows.
- **Feature Dimension**: 8 numeric features + 1 class label (`v2.2-8feat`).
- **Data Class Breakdown**:
  - `LOT_A_DEMO`: 4 samples extracted from `data/sample/sample_parking_lot.png` (synthetic demo render).
  - `LOT_TRAIN_SYNTH_1..4`: 28 procedural synthetic Java 2D calibration renders.
  - `LOT_VAL_OCC_CALIB / LOT_VAL_AVAIL_CALIB`: 8 procedural synthetic Java 2D calibration renders.
  - `LOT_TEST_OCC_HOLD / LOT_TEST_AVAIL_HOLD`: 8 procedural synthetic Java 2D calibration renders.
- **Real Camera Photographs**: **0 verified real-camera photographs** in the bundled 48-sample set.
- **Vehicle Features**: Neither `vehicleOverlapScore` nor `vehicleConfidenceScore` exists in this legacy CSV.

---

## 3. Legacy 8-Feature Baseline Performance (`SYNTHETIC-DEMO`)

*Model: CART Decision Tree (`maxDepth = 4`, `minSamplesPerLeaf = 2`), Data Class: `SYNTHETIC-DEMO / LEGACY-8-FEATURE`*

### Evaluation Results (Resubstitution & Synthetic Holdout)
- **Train Partition ($N=32$)**: Accuracy = 1.0000, Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (Resubstitution — Not Independent)
- **Validation Partition ($N=8$)**: Accuracy = 1.0000, Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (Synthetic Holdout)
- **Held-Out Test Partition ($N=8$)**: Accuracy = 1.0000, Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (Synthetic Holdout)

```
Confusion Matrix (rows=Actual, cols=Predicted):
                 Pred:OCC  Pred:AVAIL
  Actual:OCC          4         0
  Actual:AVAIL        0         4
```

### Learned Tree Structure
```
if largestBlobRatio <= 0.0735:
  if saturationRatio <= 0.3211:
    -> AVAILABLE (100% purity, occ=0, avail=16, n=16)
  else:
    -> OCCUPIED (100% purity, occ=2, avail=0, n=2)
else:
  -> OCCUPIED (100% purity, occ=14, avail=0, n=14)
```

> [!WARNING]
> This 100% figure represents an over-simplified synthetic calibration toy problem with clean separable boundaries. It is **not evidence of real-world accuracy or enhanced-model performance**.

---

## 4. Enhanced 10-Feature Reproduction Recipe

To evaluate the Canonical Enhanced model (`v2.3-10feat-enhanced-v1`) on real-world parking lots, follow this reproduction recipe using the external **PKLot** dataset:

### Step 1: Download PKLot Dataset
Download the official PKLot dataset archive (`PKLot.tar.gz` or `PKLot.zip`, ~4.6 GB containing 12,417 images across PUCPR, UFPR04, and UFPR05 lots) from:
`https://web.inf.ufpr.br/vri/databases/parking-lot-database/`

Extract or place the archive at:
`data/raw/PKLot.zip` or `data/raw/PKLot/`

### Step 2: Run Production PKLot Feature Extraction
Execute the PKLot batch importer to run the production feature extraction pipeline (combining 8 IVP features computed from contrast/gamma-enhanced crops with 2 vehicle features computed from ONNX detection on the original photograph):
```bash
# Set Java 17 environment
set JAVA_HOME=E:\JAVA17
set PATH=E:\JAVA17\bin;%PATH%

# Run full PKLot integration test / importer
mvn test -Dtest=RealPkLotArchiveIntegrationTest
```

### Step 3: Train and Evaluate with Group-Disjoint Integrity
```bash
# Train enhanced model on PKLot TRAIN partition only and evaluate on camera-independent TEST partition
mvn test -Dtest=DatasetEvaluationIntegrityTest
```

### Expected Output & Schemas
- **Feature Schema**: `v2.3-10feat-enhanced-v1`
- **Enhancement Policy**: `ivp-enhance-v1`
- **Capture-Group Partitions**: Preserved from PKLot XML metadata without re-splitting. If fewer than 3 independent camera lots are loaded, the evaluation will strictly tag:
  `Scope: INSUFFICIENT_GROUPS_NON_INDEPENDENT` (groupDisjoint = false).
