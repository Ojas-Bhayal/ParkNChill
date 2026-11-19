package com.parknchillai.ai;

import com.parknchillai.parking.SlotStatus;

import java.util.Locale;

/**
 * Evaluates a trained classifier on a labelled partition and reports classification
 * metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, False-Available Rate,
 * and False-Occupied Rate) alongside the explicit evaluation partition scope, single-class
 * availability guards, and provenance disclaimers (ParkNChill v2.3 Requirement 4).
 */
public class ModelEvaluator {

    public enum EvaluationScope {
        TRAINING_SET("Measured on Training Data (Resubstitution — Not Independent)"),
        VALIDATION_PARTITION("Measured on Validation Partition (Group-Disjoint)"),
        RANDOM_STRATIFIED_VALIDATION("Measured on Validation Split (Non-Group-Disjoint Fallback)"),
        HELD_OUT_TEST_PARTITION("Measured on Held-Out Test Partition (Group-Disjoint)"),
        RANDOM_STRATIFIED_HOLDOUT("Measured on Random Stratified Holdout (Non-Group-Disjoint Fallback)"),
        INSUFFICIENT_GROUPS_NON_INDEPENDENT("INSUFFICIENT_GROUPS / NON_INDEPENDENT"),
        NOT_ESTABLISHED("Not Established (Insufficient or Single-Class Partition)");

        private final String description;
        EvaluationScope(String description) { this.description = description; }
        public String getDescription()      { return description; }
        @Override public String toString()  { return description; }
    }

    private ModelEvaluator() { /* utility class */ }

    public static EvaluationReport evaluate(DecisionTreePredictor predictor,
                                            double[][] features,
                                            SlotStatus[] labels) {
        return evaluate(predictor, features, labels, EvaluationScope.HELD_OUT_TEST_PARTITION, "");
    }

    public static EvaluationReport evaluate(DecisionTreePredictor predictor,
                                            double[][] features,
                                            SlotStatus[] labels,
                                            EvaluationScope scope,
                                            String provenanceNote) {
        if (features == null || features.length == 0) {
            throw new IllegalArgumentException("Evaluation set must not be empty.");
        }
        if (labels == null || labels.length != features.length) {
            throw new IllegalArgumentException("labels length must match features rows.");
        }

        int tp = 0, fn = 0, fp = 0, tn = 0;

        for (int i = 0; i < features.length; i++) {
            SlotStatus predicted = predictor.getModel().predict(features[i]);
            SlotStatus actual    = labels[i];
            if (actual == SlotStatus.OCCUPIED  && predicted == SlotStatus.OCCUPIED)  tp++;
            else if (actual == SlotStatus.OCCUPIED  && predicted == SlotStatus.AVAILABLE) fn++;
            else if (actual == SlotStatus.AVAILABLE && predicted == SlotStatus.OCCUPIED)  fp++;
            else tn++;
        }

        return buildReportFromCounts(tp, fn, fp, tn, scope, provenanceNote);
    }

    /**
     * Builds an explicitly typed result when independent held-out evaluation cannot proceed
     * because fewer than 3 independent groups exist (WS-7 requirement).
     */
    public static EvaluationReport insufficientGroupsResult(String reason) {
        String msg = (reason != null && !reason.isBlank())
            ? reason
            : "Evaluation refused: fewer than 3 independent groups exist (INSUFFICIENT_GROUPS / NON_INDEPENDENT).";
        return new EvaluationReport(
            Double.NaN, Double.NaN, Double.NaN, Double.NaN,
            0, 0, 0, 0,
            EvaluationScope.INSUFFICIENT_GROUPS_NON_INDEPENDENT,
            "INSUFFICIENT_GROUPS / NON_INDEPENDENT",
            false,
            msg
        );
    }

    public static EvaluationReport buildReportFromCounts(int tp, int fn, int fp, int tn,
                                                         EvaluationScope scope,
                                                         String provenanceNote) {
        int total = tp + fn + fp + tn;
        if (total <= 0) {
            throw new IllegalArgumentException("Total evaluated count must be > 0.");
        }
        int actualOcc   = tp + fn;
        int actualAvail = fp + tn;
        double accuracy = (double) (tp + tn) / total;

        if (total < 4) {
            String reason = "Insufficient test data: N=" + total
                + " is too small to compute reliable metrics (requires N >= 4 with both classes).";
            return new EvaluationReport(
                Double.NaN, Double.NaN, Double.NaN, Double.NaN,
                tp, fn, fp, tn,
                scope != null ? scope : EvaluationScope.NOT_ESTABLISHED,
                provenanceNote != null ? provenanceNote : "",
                false,
                reason
            );
        }

        if (actualOcc == 0 || actualAvail == 0) {
            String missingClass = (actualOcc == 0) ? "OCCUPIED" : "AVAILABLE";
            String reason = "Single-class evaluation partition: 0 actual " + missingClass
                + " samples present; precision/recall/F1 are undefined and marked UNAVAILABLE.";
            return new EvaluationReport(
                accuracy, Double.NaN, Double.NaN, Double.NaN,
                tp, fn, fp, tn,
                scope != null ? scope : EvaluationScope.NOT_ESTABLISHED,
                provenanceNote != null ? provenanceNote : "",
                false,
                reason
            );
        }

        double precision = (tp + fp) == 0 ? 0.0 : (double) tp / (tp + fp);
        double recall    = (tp + fn) == 0 ? 0.0 : (double) tp / (tp + fn);
        double f1        = (precision + recall) == 0 ? 0.0
                           : 2 * precision * recall / (precision + recall);

        return new EvaluationReport(
            accuracy, precision, recall, f1, tp, fn, fp, tn,
            scope != null ? scope : EvaluationScope.HELD_OUT_TEST_PARTITION,
            provenanceNote != null ? provenanceNote : "",
            true,
            ""
        );
    }

    /** Immutable report of classification evaluation metrics, availability status, and partition scope. */
    public static final class EvaluationReport {
        public final double accuracy;
        public final double precision;
        public final double recall;
        public final double f1;
        public final int tp, fn, fp, tn;
        public final EvaluationScope scope;
        public final String provenanceNote;
        public final boolean metricsAvailable;
        public final String unavailableReason;

        public EvaluationReport(double accuracy, double precision, double recall, double f1,
                                int tp, int fn, int fp, int tn) {
            this(accuracy, precision, recall, f1, tp, fn, fp, tn,
                EvaluationScope.HELD_OUT_TEST_PARTITION, "", true, "");
        }

        public EvaluationReport(double accuracy, double precision, double recall, double f1,
                                int tp, int fn, int fp, int tn,
                                EvaluationScope scope, String provenanceNote) {
            this(accuracy, precision, recall, f1, tp, fn, fp, tn, scope, provenanceNote,
                (tp + fn) > 0 && (fp + tn) > 0 && Double.isFinite(precision) && Double.isFinite(recall) && Double.isFinite(f1),
                ((tp + fn) == 0 || (fp + tn) == 0) ? "Single-class partition" : "");
        }

        public EvaluationReport(double accuracy, double precision, double recall, double f1,
                                int tp, int fn, int fp, int tn,
                                EvaluationScope scope, String provenanceNote,
                                boolean metricsAvailable, String unavailableReason) {
            this.accuracy          = accuracy;
            this.precision         = precision;
            this.recall            = recall;
            this.f1                = f1;
            this.tp = tp; this.fn = fn;
            this.fp = fp; this.tn = tn;
            this.scope             = scope != null ? scope : EvaluationScope.HELD_OUT_TEST_PARTITION;
            this.provenanceNote    = provenanceNote != null ? provenanceNote : "";
            this.metricsAvailable  = metricsAvailable;
            this.unavailableReason = unavailableReason != null ? unavailableReason : "";
        }

        public int totalEvaluated()      { return tp + fn + fp + tn; }
        public int falseAvailableCount() { return fn; }
        public int falseOccupiedCount()  { return fp; }

        /**
         * Returns the False-Available Rate ({@code FN / (TP + FN)}), i.e. the fraction of truly
         * occupied spaces mistakenly predicted as Available.
         */
        public double falseAvailableRate() {
            return (tp + fn) == 0 ? Double.NaN : (double) fn / (tp + fn);
        }

        /**
         * Returns the False-Occupied Rate ({@code FP / (TN + FP)}), i.e. the fraction of truly
         * empty spaces mistakenly predicted as Occupied.
         */
        public double falseOccupiedRate() {
            return (tn + fp) == 0 ? Double.NaN : (double) fp / (tn + fp);
        }

        public String formatMetric(double value) {
            if (!metricsAvailable || !Double.isFinite(value)) {
                return "UNAVAILABLE";
            }
            return String.format(Locale.ROOT, "%.4f", value);
        }

        @Override
        public String toString() {
            String metricNotice = metricsAvailable ? ""
                : ("Metric Status   : UNAVAILABLE (" + unavailableReason + ")\n");
            return String.format(Locale.ROOT,
                "Scope           : %s%n" +
                "%s" +
                "Accuracy        : %.4f (%d / %d)%n" +
                "Precision (OCC) : %s%n" +
                "Recall    (OCC) : %s%n" +
                "F1-score  (OCC) : %s%n" +
                "False-Available : %d (FN, Rate=%s)%n" +
                "Confusion Matrix (rows=Actual, cols=Predicted):%n" +
                "                 Pred:OCC  Pred:AVAIL%n" +
                "  Actual:OCC     %6d    %6d%n" +
                "  Actual:AVAIL   %6d    %6d%n" +
                "%s",
                scope.getDescription(),
                metricNotice,
                accuracy, tp + tn, totalEvaluated(),
                formatMetric(precision), formatMetric(recall), formatMetric(f1),
                fn, formatMetric(falseAvailableRate()),
                tp, fn, fp, tn,
                provenanceNote.isBlank() ? "" : ("Provenance Note : " + provenanceNote + "\n"));
        }
    }
}
