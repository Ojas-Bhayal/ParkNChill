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
