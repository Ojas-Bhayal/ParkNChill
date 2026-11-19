package com.parknchillai.ai;

import com.parknchillai.parking.SlotStatus;

import java.util.Arrays;
import java.util.Locale;

/**
 * Executes the validation-driven classifier comparison experiment (PNC-FIX-05, PNC-REM-05) between:
 * <ol>
 *   <li>Baseline single <strong>CART Decision Tree</strong> ({@link DecisionTreeTrainer})</li>
 *   <li><strong>Random Forest</strong> ({@link RandomForestClassifier}, 15 bootstrap trees with random feature subspaces)</li>
 * </ol>
 *
 * <p>Pre-declared model selection rules (applied strictly to the <strong>Validation</strong>
 * partition — held-out Test labels/metrics are never consulted during model selection):
 * <ol>
 *   <li><strong>Validation Adequacy Gate</strong>: Requires a group-disjoint split
 *       ({@code partitions.groupDisjoint() == true}), at least {@link SelectionPolicy#minValidationSamples()}
 *       ({@code 6}) validation samples, and both classes present in the validation partition
 *       ({@code metricsAvailable == true}). Otherwise returns {@link SelectedModel#NO_RELIABLE_WINNER}.</li>
 *   <li><strong>Mandatory Occupied-Recall Safety Floor ({@code PNC-REM-05})</strong>:
 *       Falsely reporting an occupied bay as available ({@code FN}) is operationally hazardous.
 *       If <em>neither</em> candidate meets {@link SelectionPolicy#minOccupiedRecall()} ({@code 0.80})
 *       on the Validation partition, returns {@link SelectedModel#NO_RELIABLE_WINNER} and does
 *       not evaluate or promote either model on the Held-Out Test set. If only one candidate
 *       meets the recall floor, that candidate wins.</li>
 *   <li><strong>Validation False-Available ({@code FN}), F1 & Accuracy Criteria</strong>:
 *       When both candidates meet the recall floor, prefers fewer validation {@code FN} errors,
 *       then higher validation {@code F1} (tolerance {@code 1e-4}), then higher validation {@code Accuracy}.</li>
 *   <li><strong>Documented Tie-Breaker (Measured Median Latency & Single-Tree Explainability)</strong>:
 *       Only when validation {@code FN}, {@code F1}, and {@code Accuracy} are tied after passing
 *       the recall safety gate does the selector break the tie using measured post-warmup median
 *       latency and deterministic single-tree path interpretability.</li>
 * </ol>
 */
public final class ClassifierBenchmark {

    public static final int    MIN_VALIDATION_SAMPLES = 6;
    public static final double MIN_OCCUPIED_RECALL    = 0.80;
    public static final int    WARMUP_ITERATIONS      = 25;
    public static final int    MEASURED_TRIALS        = 31;

    private ClassifierBenchmark() { /* utility class */ }

    /**
     * Explicit, configurable validation-selection policy (PNC-REM-05).
     */
    public record SelectionPolicy(
            int minValidationSamples,
            double minOccupiedRecall
    ) {
        public SelectionPolicy {
            if (minValidationSamples < 2) {
                throw new IllegalArgumentException("minValidationSamples must be >= 2.");
            }
            if (!Double.isFinite(minOccupiedRecall) || minOccupiedRecall <= 0.0 || minOccupiedRecall > 1.0) {
                throw new IllegalArgumentException("minOccupiedRecall must be in (0.0, 1.0].");
            }
        }

        public static SelectionPolicy defaultPolicy() {
            return new SelectionPolicy(MIN_VALIDATION_SAMPLES, MIN_OCCUPIED_RECALL);
        }
    }

    public enum SelectedModel {
        CART_DECISION_TREE("CART Decision Tree"),
        RANDOM_FOREST("Random Forest (15 Trees)"),
        NO_RELIABLE_WINNER("NO RELIABLE WINNER");

        private final String displayName;
        SelectedModel(String displayName) { this.displayName = displayName; }
        public String getDisplayName()    { return displayName; }
    }

    public record BenchmarkResult(
            SelectedModel selectedModel,
            ModelEvaluator.EvaluationReport cartValidationReport,
            ModelEvaluator.EvaluationReport rfValidationReport,
            ModelEvaluator.EvaluationReport selectedModelTestReport,
            ModelEvaluator.EvaluationReport cartTestReport,
            ModelEvaluator.EvaluationReport rfTestReport,
            double cartAvgInferenceMicros,
            double rfAvgInferenceMicros,
            String selectionRationale
    ) {
        public String formatSummaryTable() {
            String testLine;
            if (selectedModelTestReport != null) {
                testLine = String.format(Locale.ROOT,
                    "Selected Model Held-Out Test (%s): Acc=%.4f (%d/%d), Precision=%s, Recall=%s, F1=%s, False-Available(FN)=%d",
                    selectedModel.getDisplayName(),
                    selectedModelTestReport.accuracy,
                    selectedModelTestReport.tp + selectedModelTestReport.tn,
                    selectedModelTestReport.totalEvaluated(),
                    selectedModelTestReport.formatMetric(selectedModelTestReport.precision),
                    selectedModelTestReport.formatMetric(selectedModelTestReport.recall),
                    selectedModelTestReport.formatMetric(selectedModelTestReport.f1),
                    selectedModelTestReport.fn);
            } else {
                testLine = "Selected Model Held-Out Test: Not Evaluated (No Candidate Passed Validation Gates)";
            }

            return String.format(Locale.ROOT,
                "Classifier Comparison (Selection Driven Strictly by Validation Partition; JVM Warmup=%d, Trials=%d Median):%n" +
                "  Model                    Val Acc   Val F1        Val Recall    Val FN   Median Latency   Interpretability%n" +
                "  -----------------------------------------------------------------------------------------------------------------%n" +
                "  CART Decision Tree       %.4f    %-12s  %-12s  %2d       %6.2f µs        Exact Single-Tree Path%n" +
                "  Random Forest (15 Trees) %.4f    %-12s  %-12s  %2d       %6.2f µs        Ensemble Vote Only%n" +
                "Validation Winner  : %s%n" +
                "Selection Rationale: %s%n" +
                "%s",
                WARMUP_ITERATIONS, MEASURED_TRIALS,
                cartValidationReport.accuracy,
                cartValidationReport.formatMetric(cartValidationReport.f1),
                cartValidationReport.formatMetric(cartValidationReport.recall),
                cartValidationReport.fn,
                cartAvgInferenceMicros,
                rfValidationReport.accuracy,
                rfValidationReport.formatMetric(rfValidationReport.f1),
                rfValidationReport.formatMetric(rfValidationReport.recall),
                rfValidationReport.fn,
                rfAvgInferenceMicros,
                selectedModel.getDisplayName(),
                selectionRationale,
                testLine);
        }
    }

    public static BenchmarkResult compare(DatasetLoader.GroupedPartitions partitions,
                                          int maxDepth,
                                          int minSamplesPerLeaf,
                                          long seed) {
        return compare(partitions, maxDepth, minSamplesPerLeaf, seed, SelectionPolicy.defaultPolicy());
    }

    public static BenchmarkResult compare(DatasetLoader.GroupedPartitions partitions,
                                          int maxDepth,
                                          int minSamplesPerLeaf,
                                          long seed,
                                          SelectionPolicy policy) {
        if (partitions == null) {
            throw new IllegalArgumentException("Partitions must not be null.");
        }
        SelectionPolicy activePolicy = (policy != null) ? policy : SelectionPolicy.defaultPolicy();

        ModelEvaluator.EvaluationScope valScope = partitions.groupDisjoint()
            ? ModelEvaluator.EvaluationScope.VALIDATION_PARTITION
            : ModelEvaluator.EvaluationScope.RANDOM_STRATIFIED_VALIDATION;
        ModelEvaluator.EvaluationScope testScope = partitions.groupDisjoint()
            ? ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION
            : ModelEvaluator.EvaluationScope.RANDOM_STRATIFIED_HOLDOUT;

        // 1. Fit both candidates strictly on partitions.train()
        DecisionTreeTrainer cartTrainer = new DecisionTreeTrainer(maxDepth, minSamplesPerLeaf);
        DecisionTreeModel cartModel = cartTrainer.train(partitions.train().features, partitions.train().labels);
        DecisionTreePredictor cartPredictor = new DecisionTreePredictor(cartModel);

        RandomForestClassifier rf = new RandomForestClassifier(15, maxDepth, minSamplesPerLeaf, seed);
        rf.fit(partitions.train().features, partitions.train().labels);

        // 2. Evaluate both candidates on the exact same validation rows
        ModelEvaluator.EvaluationReport cartVal = ModelEvaluator.evaluate(
            cartPredictor, partitions.validation().features, partitions.validation().labels,
            valScope, "CART Validation");
        ModelEvaluator.EvaluationReport rfVal = evaluateRf(
            rf, partitions.validation().features, partitions.validation().labels,
            valScope);

        // 3. Measure post-warmup median per-sample inference latency on validation/train features
        double[][] timingSamples = partitions.validation().size() > 0
            ? partitions.validation().features
            : partitions.train().features;
        double cartMedianMicros = measureMedianLatencyMicros(timingSamples, cartModel::predict);
        double rfMedianMicros   = measureMedianLatencyMicros(timingSamples, rf::predict);

        // 4. Apply pre-declared validation-only selection rules (NEVER inspect partitions.test() here)
        SelectedModel winner;
        String rationale;

        int nVal = partitions.validation().size();
        if (!partitions.groupDisjoint() || nVal < activePolicy.minValidationSamples()
                || !cartVal.metricsAvailable || !rfVal.metricsAvailable) {
            winner = SelectedModel.NO_RELIABLE_WINNER;
            rationale = String.format(Locale.ROOT,
                "NO RELIABLE WINNER: Validation evidence failed the Validation Adequacy Gate " +
                "(N_val=%d [min=%d], groupDisjoint=%s, cartMetricsAvailable=%s, rfMetricsAvailable=%s). " +
                "Observed Validation Acc: CART=%.4f (FN=%d), RF=%.4f (FN=%d); Median Latency: CART=%.2f µs, RF=%.2f µs.",
                nVal, activePolicy.minValidationSamples(), partitions.groupDisjoint(),
                cartVal.metricsAvailable, rfVal.metricsAvailable,
                cartVal.accuracy, cartVal.fn, rfVal.accuracy, rfVal.fn,
                cartMedianMicros, rfMedianMicros);
        } else {
            boolean cartMeetsRecall = cartVal.recall >= activePolicy.minOccupiedRecall();
            boolean rfMeetsRecall   = rfVal.recall >= activePolicy.minOccupiedRecall();

            if (!cartMeetsRecall && !rfMeetsRecall) {
                // PNC-REM-05: Both candidates failed the required occupied-recall safety floor -> NO_RELIABLE_WINNER!
                winner = SelectedModel.NO_RELIABLE_WINNER;
                rationale = String.format(Locale.ROOT,
                    "NO RELIABLE WINNER: Both candidates failed the mandatory Validation Occupied-Recall safety gate (>= %.2f): " +
                    "CART Val Recall=%.4f (Val FN=%d, Val F1=%.4f, Val Acc=%.4f, %.2f µs) and " +
                    "Random Forest Val Recall=%.4f (Val FN=%d, Val F1=%.4f, Val Acc=%.4f, %.2f µs). " +
                    "Neither candidate is promoted or evaluated on the Held-Out Test partition.",
                    activePolicy.minOccupiedRecall(),
                    cartVal.recall, cartVal.fn, cartVal.f1, cartVal.accuracy, cartMedianMicros,
                    rfVal.recall, rfVal.fn, rfVal.f1, rfVal.accuracy, rfMedianMicros);
            } else if (cartMeetsRecall && !rfMeetsRecall) {
                winner = SelectedModel.CART_DECISION_TREE;
                rationale = String.format(Locale.ROOT,
                    "Selected CART Decision Tree on Validation Safety Gate (Occupied Recall >= %.2f): " +
                    "CART passed with Val Recall=%.4f (Val FN=%d, Val F1=%.4f, Val Acc=%.4f, %.2f µs) " +
                    "while Random Forest failed the recall floor with Val Recall=%.4f (Val FN=%d, Val F1=%.4f, %.2f µs).",
                    activePolicy.minOccupiedRecall(),
                    cartVal.recall, cartVal.fn, cartVal.f1, cartVal.accuracy, cartMedianMicros,
                    rfVal.recall, rfVal.fn, rfVal.f1, rfMedianMicros);
            } else if (rfMeetsRecall && !cartMeetsRecall) {
                winner = SelectedModel.RANDOM_FOREST;
                rationale = String.format(Locale.ROOT,
                    "Selected Random Forest (15 Trees) on Validation Safety Gate (Occupied Recall >= %.2f): " +
                    "Random Forest passed with Val Recall=%.4f (Val FN=%d, Val F1=%.4f, Val Acc=%.4f, %.2f µs) " +
                    "while CART failed the recall floor with Val Recall=%.4f (Val FN=%d, Val F1=%.4f, %.2f µs).",
                    activePolicy.minOccupiedRecall(),
                    rfVal.recall, rfVal.fn, rfVal.f1, rfVal.accuracy, rfMedianMicros,
                    cartVal.recall, cartVal.fn, cartVal.f1, cartMedianMicros);
            } else if (cartVal.fn != rfVal.fn) {
                if (cartVal.fn < rfVal.fn) {
                    winner = SelectedModel.CART_DECISION_TREE;
                    rationale = String.format(Locale.ROOT,
                        "Selected CART Decision Tree on Validation Rule 1 (fewer False-Available FN errors after passing Recall >= %.2f): " +
                        "CART had Val FN=%d (Val Recall=%.4f, Val F1=%.4f, Val Acc=%.4f, %.2f µs) vs Random Forest Val FN=%d (Val Recall=%.4f, Val F1=%.4f, Val Acc=%.4f, %.2f µs).",
                        activePolicy.minOccupiedRecall(),
                        cartVal.fn, cartVal.recall, cartVal.f1, cartVal.accuracy, cartMedianMicros,
                        rfVal.fn, rfVal.recall, rfVal.f1, rfVal.accuracy, rfMedianMicros);
                } else {
                    winner = SelectedModel.RANDOM_FOREST;
                    rationale = String.format(Locale.ROOT,
                        "Selected Random Forest (15 Trees) on Validation Rule 1 (fewer False-Available FN errors after passing Recall >= %.2f): " +
                        "Random Forest had Val FN=%d (Val Recall=%.4f, Val F1=%.4f, Val Acc=%.4f, %.2f µs) vs CART Val FN=%d (Val Recall=%.4f, Val F1=%.4f, Val Acc=%.4f, %.2f µs).",
                        activePolicy.minOccupiedRecall(),
                        rfVal.fn, rfVal.recall, rfVal.f1, rfVal.accuracy, rfMedianMicros,
                        cartVal.fn, cartVal.recall, cartVal.f1, cartVal.accuracy, cartMedianMicros);
                }
            } else if (Math.abs(cartVal.f1 - rfVal.f1) > 1e-4) {
                if (cartVal.f1 > rfVal.f1) {
                    winner = SelectedModel.CART_DECISION_TREE;
                    rationale = String.format(Locale.ROOT,
                        "Selected CART Decision Tree on Validation Rule 2 (higher Validation F1): " +
                        "CART achieved Val F1=%.4f (Val Acc=%.4f, Val Recall=%.4f, Val FN=%d, %.2f µs) vs Random Forest Val F1=%.4f (Val Acc=%.4f, Val Recall=%.4f, Val FN=%d, %.2f µs).",
                        cartVal.f1, cartVal.accuracy, cartVal.recall, cartVal.fn, cartMedianMicros,
                        rfVal.f1, rfVal.accuracy, rfVal.recall, rfVal.fn, rfMedianMicros);
                } else {
                    winner = SelectedModel.RANDOM_FOREST;
                    rationale = String.format(Locale.ROOT,
                        "Selected Random Forest (15 Trees) on Validation Rule 2 (higher Validation F1): " +
                        "Random Forest achieved Val F1=%.4f (Val Acc=%.4f, Val Recall=%.4f, Val FN=%d, %.2f µs) vs CART Val F1=%.4f (Val Acc=%.4f, Val Recall=%.4f, Val FN=%d, %.2f µs).",
                        rfVal.f1, rfVal.accuracy, rfVal.recall, rfVal.fn, rfMedianMicros,
                        cartVal.f1, cartVal.accuracy, cartVal.recall, cartVal.fn, cartMedianMicros);
                }
            } else if (Math.abs(cartVal.accuracy - rfVal.accuracy) > 1e-4) {
                if (cartVal.accuracy > rfVal.accuracy) {
                    winner = SelectedModel.CART_DECISION_TREE;
                    rationale = String.format(Locale.ROOT,
                        "Selected CART Decision Tree on Validation Rule 2 (higher Validation Accuracy): " +
                        "CART achieved Val Acc=%.4f (Val F1=%.4f, Val FN=%d, %.2f µs) vs Random Forest Val Acc=%.4f (Val F1=%.4f, Val FN=%d, %.2f µs).",
                        cartVal.accuracy, cartVal.f1, cartVal.fn, cartMedianMicros,
                        rfVal.accuracy, rfVal.f1, rfVal.fn, rfMedianMicros);
                } else {
                    winner = SelectedModel.RANDOM_FOREST;
                    rationale = String.format(Locale.ROOT,
                        "Selected Random Forest (15 Trees) on Validation Rule 2 (higher Validation Accuracy): " +
                        "Random Forest achieved Val Acc=%.4f (Val F1=%.4f, Val FN=%d, %.2f µs) vs CART Val Acc=%.4f (Val F1=%.4f, Val FN=%d, %.2f µs).",
                        rfVal.accuracy, rfVal.f1, rfVal.fn, rfMedianMicros,
                        cartVal.accuracy, cartVal.f1, cartVal.fn, cartMedianMicros);
                }
            } else {
                winner = SelectedModel.CART_DECISION_TREE;
                rationale = String.format(Locale.ROOT,
                    "Validation tie resolved to CART Decision Tree via Rule 3 tie-breaker (lower latency & exact single-tree explainability): " +
                    "both models passed Recall >= %.2f and tied on Validation (N_val=%d) with Val Acc=%.4f, Val F1=%.4f, Val Recall=%.4f, and Val FN=%d; " +
                    "CART measured median latency was %.2f µs vs Random Forest %.2f µs. " +
                    "(Note: Validation selection does not by itself establish real-world camera generalization.)",
                    activePolicy.minOccupiedRecall(),
                    nVal, cartVal.accuracy, cartVal.f1, cartVal.recall, cartVal.fn,
                    cartMedianMicros, rfMedianMicros);
            }
        }

        // 5. Evaluate the protected Held-Out Test partition ONLY when a valid winner passed all Validation gates
        ModelEvaluator.EvaluationReport selectedTest = null;
        ModelEvaluator.EvaluationReport cartTest = null;
        ModelEvaluator.EvaluationReport rfTest = null;

        if (winner == SelectedModel.CART_DECISION_TREE) {
            cartTest = ModelEvaluator.evaluate(
                cartPredictor, partitions.test().features, partitions.test().labels,
                testScope, "CART Held-Out Test (Selected Winner)");
            selectedTest = cartTest;
        } else if (winner == SelectedModel.RANDOM_FOREST) {
            rfTest = evaluateRf(
                rf, partitions.test().features, partitions.test().labels,
                testScope);
            selectedTest = rfTest;
        }

        return new BenchmarkResult(
            winner, cartVal, rfVal, selectedTest, cartTest, rfTest,
            cartMedianMicros, rfMedianMicros, rationale
        );
    }

    @FunctionalInterface
    private interface VectorPredictor {
        SlotStatus predict(double[] fv);
    }

    private static double measureMedianLatencyMicros(double[][] samples, VectorPredictor predictor) {
        if (samples == null || samples.length == 0) return 0.0;
        for (int w = 0; w < WARMUP_ITERATIONS; w++) {
            for (double[] fv : samples) {
                predictor.predict(fv);
            }
        }
        double[] trialMicros = new double[MEASURED_TRIALS];
        for (int t = 0; t < MEASURED_TRIALS; t++) {
            long start = System.nanoTime();
            for (double[] fv : samples) {
                predictor.predict(fv);
            }
            long elapsed = Math.max(1L, System.nanoTime() - start);
            trialMicros[t] = (elapsed / 1000.0) / samples.length;
        }
        Arrays.sort(trialMicros);
        return trialMicros[MEASURED_TRIALS / 2];
    }

    private static ModelEvaluator.EvaluationReport evaluateRf(RandomForestClassifier rf,
                                                              double[][] features,
                                                              SlotStatus[] labels,
                                                              ModelEvaluator.EvaluationScope scope) {
        int tp = 0, fn = 0, fp = 0, tn = 0;
        for (int i = 0; i < features.length; i++) {
            SlotStatus pred = rf.predict(features[i]);
            SlotStatus actual = labels[i];
            if (actual == SlotStatus.OCCUPIED  && pred == SlotStatus.OCCUPIED)  tp++;
            else if (actual == SlotStatus.OCCUPIED  && pred == SlotStatus.AVAILABLE) fn++;
            else if (actual == SlotStatus.AVAILABLE && pred == SlotStatus.OCCUPIED)  fp++;
            else tn++;
        }
        return ModelEvaluator.buildReportFromCounts(tp, fn, fp, tn, scope, "Random Forest (15 Trees)");
    }
}
