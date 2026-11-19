package com.parknchillai.detection;

import com.parknchillai.ai.DecisionTreeModel;
import com.parknchillai.parking.SlotStatus;

import java.util.Locale;

/**
 * Responsibility 5 of the Vehicle Detection Pipeline:
 * Final occupancy determination across the three explicitly selectable modes
 * (ParkNChill v2.3 Requirements 1, 4 &amp; 5):
 * <ol>
 *   <li>{@link AnalysisMode#VEHICLE_DETECTION} (Default) — 1. Automatic Vehicle Detection
 *       (raw-image detector + spatial slot overlap is the primary source of occupancy status;
 *       the Decision Tree provides a secondary prediction for comparison without overwriting
 *       the detector's evidence, and any disagreement is flagged for operator visibility)</li>
 *   <li>{@link AnalysisMode#FEATURE_DECISION_TREE} — 2. Feature-Based Decision Tree</li>
 *   <li>{@link AnalysisMode#COMPARISON} — 3. Comparison Mode (routes any disagreement to
 *       {@link SlotStatus#NEEDS_REVIEW} without blending scores)</li>
 * </ol>
 */
public final class OccupancyDecisionEngine {

    private OccupancyDecisionEngine() { /* utility class */ }

    public enum AnalysisMode {
        VEHICLE_DETECTION("1. Automatic Vehicle Detection"),
        FEATURE_DECISION_TREE("2. Feature-Based Decision Tree"),
        COMPARISON("3. Comparison Mode");

        private final String displayName;
        AnalysisMode(String displayName) { this.displayName = displayName; }
        public String getDisplayName() { return displayName; }
        @Override public String toString() { return displayName; }
    }

    public record CombinedSlotDecision(
            AnalysisMode mode,
            SlotStatus finalStatus,
            SlotStatus treeStatus,
            SlotStatus vehicleDetectorStatus,
            boolean methodsDisagree,
            String predictionSource,
            String reviewReason,
            String comparisonSummary
    ) {}

    /**
     * Resolves the final occupancy status for a single slot under the selected {@link AnalysisMode}
     * (defaulting to {@link AnalysisMode#VEHICLE_DETECTION}) without ever blending scores from
     * the Decision Tree and Vehicle Detector.
     */
    public static CombinedSlotDecision decide(
            AnalysisMode mode,
            DecisionTreeModel.DecisionTrace treeTrace,
            VehicleDetectionService.DetectionBatchResult detectorBatch,
            SlotOverlapAnalyzer.SlotOverlapEvidence overlapEvidence) {
        AnalysisMode activeMode = mode != null ? mode : AnalysisMode.VEHICLE_DETECTION;
        SlotStatus treeStatus = treeTrace != null ? treeTrace.finalStatus() : SlotStatus.UNCLASSIFIED;
        double leafPurityPct = treeTrace != null ? treeTrace.leaf().leafPurity() * 100.0 : 0.0;
        String treeReviewReason = (treeTrace != null && treeTrace.reviewRequired())
            ? treeTrace.uncertaintyReason()
            : (treeTrace == null ? "Decision Tree model unavailable or not trained." : "");

        boolean detectorAvailable = detectorBatch != null && detectorBatch.isAvailable();
        boolean inferenceFailed = detectorBatch != null
            && detectorBatch.status() == VehicleDetectionService.DetectorStatus.INFERENCE_FAILED;
        SlotStatus vehStatus = (detectorAvailable && overlapEvidence != null)
            ? overlapEvidence.status()
            : (inferenceFailed ? SlotStatus.UNCLASSIFIED : SlotStatus.NEEDS_REVIEW);

        switch (activeMode) {
            case FEATURE_DECISION_TREE -> {
                String source = "Feature-Based CART Decision Tree (v2.3)";
                String review = (treeStatus == SlotStatus.NEEDS_REVIEW || treeStatus == SlotStatus.UNCLASSIFIED)
                    ? treeReviewReason
                    : "";
                String summary = String.format(Locale.ROOT,
                    "Decision Tree: %s (Leaf Purity %.0f%%, occ=%d, avail=%d, n=%d)%s",
                    treeStatus, leafPurityPct,
                    treeTrace != null ? treeTrace.leaf().occupiedCount() : 0,
                    treeTrace != null ? treeTrace.leaf().availableCount() : 0,
                    treeTrace != null ? treeTrace.leaf().totalSamples() : 0,
                    detectorAvailable && overlapEvidence != null
                        ? String.format(Locale.ROOT, " | Vehicle Detector (info): %s (bestIoS=%.2f)",
                            vehStatus, overlapEvidence.bestIoS())
                        : " | Vehicle Detector: UNAVAILABLE (" + (detectorBatch != null ? detectorBatch.status() : "NONE") + ")");
                return new CombinedSlotDecision(
                    activeMode, treeStatus, treeStatus, vehStatus, false, source, review, summary
                );
            }

            case VEHICLE_DETECTION -> {
                if (!detectorAvailable || overlapEvidence == null) {
                    String statusLabel = detectorBatch != null ? detectorBatch.status().name() : "UNCONFIGURED";
                    String msg = detectorBatch != null
                        ? detectorBatch.statusMessage()
                        : "Vehicle detector is not configured.";
                    SlotStatus failState = inferenceFailed ? SlotStatus.UNCLASSIFIED : SlotStatus.NEEDS_REVIEW;
                    return new CombinedSlotDecision(
                        activeMode,
                        failState,
                        treeStatus,
                        failState,
                        false,
                        "Vehicle Detection [UNAVAILABLE: " + statusLabel + "]",
                        msg,
                        "Vehicle Detection unavailable (" + statusLabel + "): " + msg
                    );
                }
                boolean disagree = treeTrace != null && vehStatus != treeStatus;
                String source = "Vehicle Detection (" + detectorBatch.modelName() + ") + Slot Overlap";
                String review = (vehStatus == SlotStatus.NEEDS_REVIEW || vehStatus == SlotStatus.UNCLASSIFIED)
                    ? overlapEvidence.reviewReason()
                    : (disagree
                        ? String.format(Locale.ROOT,
                            "Detector/Tree Disagreement (Detector=%s [primary] vs Secondary CART Tree=%s — review recommended)",
                            vehStatus, treeStatus)
                        : "");
                String secondaryTreeNote = treeTrace != null
                    ? String.format(Locale.ROOT, " | Secondary CART Tree: %s (Leaf Purity %.0f%%)%s",
                        treeStatus, leafPurityPct,
                        disagree ? " [DISAGREEMENT — Review Recommended]" : "")
                    : "";
                return new CombinedSlotDecision(
                    activeMode,
                    vehStatus,
                    treeStatus,
                    vehStatus,
                    disagree,
                    source,
                    review,
                    overlapEvidence.explanation() + secondaryTreeNote
                );
            }

            case COMPARISON -> {
                if (!detectorAvailable || overlapEvidence == null) {
                    String statusLabel = detectorBatch != null ? detectorBatch.status().name() : "UNCONFIGURED";
                    String msg = "Comparison incomplete: Vehicle Detector is " + statusLabel
                        + " (" + (detectorBatch != null ? detectorBatch.statusMessage() : "unconfigured")
                        + "). Decision Tree predicted " + treeStatus + ".";
                    return new CombinedSlotDecision(
                        activeMode,
                        SlotStatus.NEEDS_REVIEW,
                        treeStatus,
                        vehStatus,
                        true,
                        "Comparison (Vehicle Detector Unavailable -> Needs Review)",
                        msg,
                        msg
                    );
                }

                if (treeStatus == SlotStatus.NEEDS_REVIEW || treeStatus == SlotStatus.UNCLASSIFIED
                        || vehStatus == SlotStatus.NEEDS_REVIEW || vehStatus == SlotStatus.UNCLASSIFIED) {
                    String reason = (treeStatus == SlotStatus.NEEDS_REVIEW || treeStatus == SlotStatus.UNCLASSIFIED)
                            && (vehStatus == SlotStatus.NEEDS_REVIEW || vehStatus == SlotStatus.UNCLASSIFIED)
                        ? ("Both models flagged review: Tree=[" + treeReviewReason + "]; Detector=[" + overlapEvidence.reviewReason() + "]")
                        : ((treeStatus == SlotStatus.NEEDS_REVIEW || treeStatus == SlotStatus.UNCLASSIFIED)
                            ? ("Decision Tree flagged review (" + treeReviewReason + "); Vehicle Detector predicted " + vehStatus)
                            : ("Vehicle Detector flagged review (" + overlapEvidence.reviewReason() + "); Decision Tree predicted " + treeStatus));
                    return new CombinedSlotDecision(
                        activeMode,
                        SlotStatus.NEEDS_REVIEW,
                        treeStatus,
                        vehStatus,
                        treeStatus != vehStatus,
                        "Comparison (Ambiguous Model Output -> Needs Review)",
                        reason,
                        reason
                    );
                }

                if (treeStatus != vehStatus) {
                    String conflictReason = String.format(Locale.ROOT,
                        "Method Disagreement: Decision Tree predicted %s (Leaf Purity %.0f%%) while Vehicle Detector predicted %s (bestIoS=%.2f, maxScore=%.2f, dets=%d). Flagged for human review.",
                        treeStatus, leafPurityPct, vehStatus,
                        overlapEvidence.bestIoS(), overlapEvidence.maxDetectionScore(),
                        overlapEvidence.contributingDetections().size());
                    return new CombinedSlotDecision(
                        activeMode,
                        SlotStatus.NEEDS_REVIEW,
                        treeStatus,
                        vehStatus,
                        true,
                        "Comparison (Disagreement -> Needs Review)",
                        conflictReason,
                        conflictReason
                    );
                }

                String agreedSummary = String.format(Locale.ROOT,
                    "Both methods AGREE on %s: Decision Tree (Leaf Purity %.0f%%, n=%d) & Vehicle Detector (bestIoS=%.2f, dets=%d).",
                    treeStatus, leafPurityPct,
                    treeTrace != null ? treeTrace.leaf().totalSamples() : 0,
                    overlapEvidence.bestIoS(), overlapEvidence.contributingDetections().size());
                return new CombinedSlotDecision(
                    activeMode,
                    treeStatus,
                    treeStatus,
                    vehStatus,
                    false,
                    "Comparison (Both Agreed: Tree + Vehicle Detector)",
                    "",
                    agreedSummary
                );
            }
        }
        throw new IllegalStateException("Unexpected mode: " + activeMode);
    }
}
