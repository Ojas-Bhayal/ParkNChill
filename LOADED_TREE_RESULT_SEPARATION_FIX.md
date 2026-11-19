# Saved Decision Tree Result Separation Fix

## Why the previous behavior was confusing

The saved-model callback switched the primary analysis mode to Feature Decision Tree and automatically reran the full image-analysis pipeline. This replaced the visible result produced by the previous mode, so the user could not see the detector result and the newly loaded tree result side by side.

## Updated behavior

1. **Load Saved Decision Tree** activates a compatible model only. It does not change the selected analysis mode or rerun detection.
2. **Last Analysis Result** remains the status from the most recent full Detect and Analyse run.
3. **Detector Suggestion** remains the evidence from ONNX detections and slot overlap, without stale disagreement tags from an earlier tree.
4. **Run Tree Separately** applies the active compatible ten-feature model to the current slot feature vectors, and writes predictions into the separate **Loaded Tree Prediction** column.
5. Selecting a slot shows the separate tree's result, leaf purity, and rule path in the detail panel.
6. The separate action does not alter the current summary, existing slot statuses, detector suggestions, or image overlays.
7. A new full analysis clears the separate predictions because the input feature vectors may have changed. The tree can be applied again.
8. The **Classifier Mode** selector remains available for users who intentionally want Feature Decision Tree or Comparison mode to produce the primary analysis result. Changing that mode and running analysis is separate from loading a model.

## Verification status

- Source-level checks: action wiring, callback flow, separate prediction map, and table/detail display were inspected.
- Regression coverage added to `UiWorkflowAndPkLotV23Test`: loading a saved tree should not change the primary mode or replace the current summary; running the tree separately should populate one row per analysed slot while keeping existing statuses unchanged.
- `mvn clean test`: **not run** because Maven is not installed in this environment (`mvn: command not found`). Run this command on the target Windows PC before relying on this build.
