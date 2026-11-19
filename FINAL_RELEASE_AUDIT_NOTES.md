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

# Final Release Audit Notes — ParkNChill v2.3

## What was checked
- Inspected the uploaded source ZIP and the clean release ZIP embedded inside it.
- Verified the clean release ZIP opens without corruption and contains no `.git/` or `target/` trees.
- Checked key source paths for the optimized CART trainer, full-pixel image hashing and bounded detection cache, detection strategy controls, debounced image-enhancement previews, multi-image memory safeguards, and schema/group-evaluation checks.
- Reconciled several stale documentation claims with the code and actual included data.

## Performance results reported by the supplied run log
These are reported measurements, not independently rerun during this audit:
- CART at 13,000 records: legacy 41.64 s versus optimized 59.06 ms; reported speedup 705.08x. The 9-model grid search is reported at about 0.56 s.
- Enhancement preview at 4000x3000: 36.89 ms; full-resolution Apply: 1434.46 ms.
- Multi-image retained heap: 32 MB at 4 images, 62 MB at 20, and 75 MB at 50.
- ONNX detection: FAST 1 pass, ACCURATE 5 passes, bounded LRU cache used for repeat requests.

## Test status
The supplied final execution log reports `mvn clean test`: 236 tests run, 235 passed, zero failures, zero errors, one skipped optional full-PKLot integration test. The outer submitted ZIP also contained Surefire XML reports that summed to those totals. Maven was not available in this review environment, so I did not independently rerun the suite.

## Remaining limitations
1. The full PKLot archive is external and is not included.
2. `data/labels/legacy/` contains 48 legacy 8-feature demo/synthetic records. The enhanced 10-feature dataset is explicitly not generated/bundled; enhanced-model accuracy is therefore NOT RUN.
3. Vehicle detection precision/recall is UNMEASURED without an independently annotated bounding-box dataset.
4. A reported local latency image `sunny_demo.jpg` is not included, so those exact timing figures cannot be reproduced from the clean ZIP.
5. Run `mvn javafx:run` on the target Windows machine to confirm the interactive UI and local ONNX runtime behavior.

## Important documentation corrections made in this release
- Identified the configured reference model as YOLOX-s and corrected the default input profile to `YOLOX_BGR_255` (BGR, float32 0–255, top-left letterbox).
- Corrected the legacy corpus composition to 5 demo-image-extracted rows and 43 procedural synthetic rows.
- Reconciled CART and detector latency values to the final audit run, while clarifying which tests use mock backends.
- Corrected the description of the real ONNX fixture test: it verifies live inference/import plumbing, not vehicle accuracy.
- Marked the earlier 200-test report as a historical snapshot.

## Recommendation
This is a substantially improved engineering prototype suitable for continued project work. Do not claim real-world enhanced Decision Tree accuracy until PKLot is imported, a valid independent group partition is created, and the enhanced 10-feature model is evaluated on its held-out test data.
