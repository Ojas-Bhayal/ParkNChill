# Loaded Decision Tree: Separate Prediction View

Loading a saved Decision Tree should not overwrite the detector output already visible in the app. The current behavior is deliberately explicit:

1. **Load Saved Decision Tree** activates a schema-compatible model and updates the model status. It does not change the analysis-mode selector, rerun detection, rewrite the status values, or redraw the image overlays.
2. Return to the image/slot review workspace. The **Run Tree Separately** button is enabled when a compatible ten-feature model and current analysis results are available.
3. Click **Run Tree Separately**. ParkNChill applies the active tree to the already extracted ten-feature vectors and writes the results into a separate **Loaded Tree Prediction** column.
4. Select a slot to inspect the loaded tree's result, leaf purity, and rule path.

The table's **Last Analysis Result** and **Detector Suggestion** remain unchanged. Occupancy totals and overlays also remain unchanged. This makes the tree output comparable with the detector result instead of silently replacing it.

If you deliberately want the Tree to become the primary `Last Analysis Result`, change the **Classifier Mode** selector to **Feature-Based Decision Tree** and click **Detect and Analyse**. That action intentionally runs the entire analysis in tree mode and may produce different primary statuses.

A new analysis or session change clears the separate tree-prediction column because the feature vectors may have changed. Click **Run Tree Separately** again to evaluate the active tree against the new result. Legacy or incompatible eight-feature models cannot be applied to enhanced ten-feature records.
