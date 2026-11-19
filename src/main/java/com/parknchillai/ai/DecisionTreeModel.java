package com.parknchillai.ai;

import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.SlotStatus;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * A hand-built, interpretable Decision Tree classifier trained with the
 * CART (Classification and Regression Tree) algorithm using the Gini
 * impurity splitting criterion.
 *
 * <p>Supports 5-feature ({@code v2.0-5feat}), 8-feature ({@code v2.2-8feat}), and
 * 10-feature detection-augmented ({@code v2.3-10feat-enhanced-v1}) schemas, empirical leaf class-purity,
 * observed training-domain bounds, and structured decision traces.
 */
public class DecisionTreeModel {

    public record DecisionStep(
            int featureIndex,
            String featureName,
            double observedValue,
            double threshold,
            String comparisonOperator,
            boolean wentLeft,
            String branchFollowed
    ) {
        public String formatStep() {
            return String.format(Locale.ROOT, "%s = %.4f %s %.4f (%s)",
                featureName, observedValue, comparisonOperator, threshold, branchFollowed);
        }
    }

    public record LeafSummary(
            SlotStatus predictedClass,
            int occupiedCount,
            int availableCount,
            int totalSamples,
            double leafPurity
    ) {}

    public record DecisionTrace(
            List<DecisionStep> steps,
            LeafSummary leaf,
            List<String> outOfDomainFeatures,
            SlotStatus finalStatus,
            boolean reviewRequired,
            String uncertaintyReason,
            String compactPath,
            String humanReadableExplanation
    ) {}

    abstract static class Node {
        abstract SlotStatus predict(double[] features);
        abstract double predictConfidence(double[] features);
        abstract LeafNode findLeaf(double[] features, List<DecisionStep> steps);
    }

    static final class InternalNode extends Node {
        final int featureIndex;
        final double threshold;
        final Node left;
        final Node right;

        InternalNode(int featureIndex, double threshold, Node left, Node right) {
            this.featureIndex = featureIndex;
            this.threshold    = threshold;
            this.left         = left;
            this.right        = right;
        }

        @Override
        SlotStatus predict(double[] features) {
            return features[featureIndex] <= threshold
                ? left.predict(features)
                : right.predict(features);
        }

        @Override
        double predictConfidence(double[] features) {
            return features[featureIndex] <= threshold
                ? left.predictConfidence(features)
                : right.predictConfidence(features);
        }

        @Override
        LeafNode findLeaf(double[] features, List<DecisionStep> steps) {
            String fname = featureIndex < FEATURE_NAMES.length
                ? FEATURE_NAMES[featureIndex]
                : "feature[" + featureIndex + "]";
            double val = features[featureIndex];
            boolean goLeft = val <= threshold;
            if (steps != null) {
                steps.add(new DecisionStep(
                    featureIndex,
                    fname,
                    val,
                    threshold,
                    goLeft ? "<=" : ">",
                    goLeft,
                    goLeft ? "Left branch (<= threshold)" : "Right branch (> threshold)"
                ));
            }
            return goLeft ? left.findLeaf(features, steps) : right.findLeaf(features, steps);
        }
    }

    static final class LeafNode extends Node {
        final SlotStatus label;
        final int occupiedCount;
        final int availableCount;

        LeafNode(SlotStatus label) {
            this(label, label == SlotStatus.OCCUPIED ? 1 : 0, label == SlotStatus.AVAILABLE ? 1 : 0);
        }

        LeafNode(SlotStatus label, int occupiedCount, int availableCount) {
            this.label          = label;
            this.occupiedCount  = Math.max(0, occupiedCount);
            this.availableCount = Math.max(0, availableCount);
        }

        @Override
        SlotStatus predict(double[] features) { return label; }

        @Override
        double predictConfidence(double[] features) {
            int total = occupiedCount + availableCount;
            if (total <= 0) return 1.0;
            int winning = (label == SlotStatus.OCCUPIED) ? occupiedCount : availableCount;
            return (double) winning / total;
        }

        @Override
        LeafNode findLeaf(double[] features, List<DecisionStep> steps) {
            return this;
        }
    }

    private Node root;
    private int featureCount;
    private int maxDepth;
    private double[] trainMin;
    private double[] trainMax;

    DecisionTreeModel() { /* package-private; created by DecisionTreeTrainer */ }

    public SlotStatus predict(double[] features) {
        validateFeatures(features);
        return root.predict(features);
    }

    public double predictConfidence(double[] features) {
        validateFeatures(features);
        return root.predictConfidence(features);
    }

    public DecisionTrace traceDecision(double[] features, UncertaintyPolicy policy) {
        UncertaintyPolicy activePolicy = policy != null ? policy : UncertaintyPolicy.defaultProvisionalPolicy();

        if (root == null) {
            LeafSummary emptyLeaf = new LeafSummary(SlotStatus.NEEDS_REVIEW, 0, 0, 0, 0.0);
            String msg = "Prediction requires review because the Decision Tree model is untrained.";
            return new DecisionTrace(List.of(), emptyLeaf, List.of(), SlotStatus.NEEDS_REVIEW, true,
                "Model untrained", "(untrained)", msg);
        }
        if (features == null || features.length != featureCount) {
            LeafSummary emptyLeaf = new LeafSummary(SlotStatus.NEEDS_REVIEW, 0, 0, 0, 0.0);
            String reason = "Incompatible feature vector length (expected " + featureCount + ")";
            return new DecisionTrace(List.of(), emptyLeaf, List.of(), SlotStatus.NEEDS_REVIEW, true,
                reason, reason, "Prediction requires review: " + reason + ".");
        }
        for (int i = 0; i < features.length; i++) {
            if (!Double.isFinite(features[i])) {
                LeafSummary emptyLeaf = new LeafSummary(SlotStatus.NEEDS_REVIEW, 0, 0, 0, 0.0);
                String reason = "Non-finite value at feature index " + i;
                return new DecisionTrace(List.of(), emptyLeaf, List.of(), SlotStatus.NEEDS_REVIEW, true,
                    reason, reason, "Prediction requires review: " + reason + ".");
            }
            if (i < FeatureVector.EXTENDED_SPECS.size()
                    && !FeatureVector.EXTENDED_SPECS.get(i).isWithinValidBounds(features[i])) {
                LeafSummary emptyLeaf = new LeafSummary(SlotStatus.NEEDS_REVIEW, 0, 0, 0, 0.0);
                String featName = (i < FEATURE_NAMES.length) ? FEATURE_NAMES[i] : ("Feature#" + i);
                String reason = "Feature " + featName + " (" + features[i] + ") is outside valid mathematical bounds";
                return new DecisionTrace(List.of(), emptyLeaf, List.of(featName), SlotStatus.NEEDS_REVIEW, true,
                    reason, reason, "Prediction requires review: " + reason + ".");
            }
        }

        List<DecisionStep> steps = new ArrayList<>();
        LeafNode leafNode = root.findLeaf(features, steps);
        int total = leafNode.occupiedCount + leafNode.availableCount;
        double purity = leafNode.predictConfidence(features);
        LeafSummary leafSummary = new LeafSummary(
            leafNode.label, leafNode.occupiedCount, leafNode.availableCount, total, purity);

        List<String> outOfDomainStrictly = getOutOfDomainFeatures(features, 0.0);
        List<String> outOfDomainWithMargin = getOutOfDomainFeatures(features, activePolicy.getDomainMarginFraction());

        boolean reviewRequired = false;
        List<String> reasons = new ArrayList<>();

        if (purity < activePolicy.getMinLeafPurity()) {
            reviewRequired = true;
            reasons.add(String.format(Locale.ROOT, "Leaf class purity (%.1f%%) is below policy threshold (%.1f%%)",
                purity * 100.0, activePolicy.getMinLeafPurity() * 100.0));
        }
        if (total > 0 && total < activePolicy.getMinLeafSupport()) {
            reviewRequired = true;
            reasons.add(String.format(Locale.ROOT, "Leaf support (n=%d) is below minimum support (%d)",
                total, activePolicy.getMinLeafSupport()));
        }
        if (outOfDomainWithMargin.size() >= activePolicy.getMaxOutOfDomainFeatures()) {
            reviewRequired = true;
            reasons.add(String.format(Locale.ROOT, "%d features lie outside training domain (%s)",
                outOfDomainWithMargin.size(), String.join(", ", outOfDomainWithMargin)));
        }

        SlotStatus finalStatus = reviewRequired ? SlotStatus.NEEDS_REVIEW : leafNode.label;
        String uncertaintyReason = reasons.isEmpty() ? "Within supported domain and leaf purity criteria" : String.join("; ", reasons);
        String compactPath = explain(features);
        String narrative = buildHumanReadableExplanation(steps, leafSummary, outOfDomainStrictly, finalStatus, reviewRequired, uncertaintyReason);

        return new DecisionTrace(
            Collections.unmodifiableList(steps),
            leafSummary,
            Collections.unmodifiableList(outOfDomainStrictly),
            finalStatus,
            reviewRequired,
            uncertaintyReason,
            compactPath,
            narrative
        );
    }

    private static String buildHumanReadableExplanation(List<DecisionStep> steps,
                                                        LeafSummary leaf,
                                                        List<String> outOfDomain,
                                                        SlotStatus finalStatus,
                                                        boolean reviewRequired,
                                                        String uncertaintyReason) {
        StringBuilder sb = new StringBuilder();
        if (reviewRequired) {
            sb.append(String.format(Locale.ROOT,
                "This space was marked as NEEDS_REVIEW (tentative tree branch: %s) because %s. ",
                leaf.predictedClass(), uncertaintyReason));
        } else {
            sb.append(String.format(Locale.ROOT,
                "The selected space was classified as %s because its extracted measurements followed the %s branch of the trained CART tree: ",
                finalStatus, leaf.predictedClass()));
        }

        for (int i = 0; i < steps.size(); i++) {
            DecisionStep s = steps.get(i);
            if (i > 0) sb.append("; then ");
            sb.append(String.format(Locale.ROOT, "%s (%.4f) was %s threshold %.4f (%s)",
                s.featureName(), s.observedValue(), s.comparisonOperator(), s.threshold(), s.branchFollowed()));
        }
        sb.append(". ");

        sb.append(String.format(Locale.ROOT,
            "The terminal leaf is supported by %d training sample(s) (%d Occupied, %d Available; %.1f%% leaf class purity — empirical training-set ratio, not a calibrated probability).",
            leaf.totalSamples(), leaf.occupiedCount(), leaf.availableCount(), leaf.leafPurity() * 100.0));

        if (!outOfDomain.isEmpty()) {
            sb.append(String.format(Locale.ROOT,
                " Note: %d feature(s) (%s) lie outside the observed training-set range.",
                outOfDomain.size(), String.join(", ", outOfDomain)));
        }
        return sb.toString();
    }

    public List<String> getOutOfDomainFeatures(double[] features, double marginFraction) {
        if (trainMin == null || trainMax == null || features == null) {
            return List.of();
        }
        List<String> out = new ArrayList<>();
        int limit = Math.min(features.length, Math.min(trainMin.length, trainMax.length));
        for (int f = 0; f < limit; f++) {
            if (isFeatureOutOfDomain(f, features[f], marginFraction)) {
                out.add(f < FEATURE_NAMES.length ? FEATURE_NAMES[f] : "feature[" + f + "]");
            }
        }
        return out;
    }

    public boolean isFeatureOutOfDomain(int featureIndex, double value, double marginFraction) {
        if (trainMin == null || trainMax == null
                || featureIndex < 0 || featureIndex >= trainMin.length) {
            return false;
        }
        double min = trainMin[featureIndex];
        double max = trainMax[featureIndex];
        double span = Math.max(1e-6, max - min);
        double tol = span * Math.max(0.0, marginFraction);
        return value < (min - tol - 1e-6) || value > (max + tol + 1e-6);
    }

    public double getTrainMin(int featureIndex) {
        if (trainMin == null || featureIndex < 0 || featureIndex >= trainMin.length) return Double.NaN;
        return trainMin[featureIndex];
    }

    public double getTrainMax(int featureIndex) {
        if (trainMax == null || featureIndex < 0 || featureIndex >= trainMax.length) return Double.NaN;
        return trainMax[featureIndex];
    }

    void setTrainDomainBounds(double[] minBounds, double[] maxBounds) {
        this.trainMin = minBounds != null ? minBounds.clone() : null;
        this.trainMax = maxBounds != null ? maxBounds.clone() : null;
    }

    private void validateFeatures(double[] features) {
        if (root == null) throw new IllegalStateException("Model has not been trained.");
        if (features == null || features.length != featureCount) {
            throw new IllegalArgumentException(
                "Expected " + featureCount + " features; got "
                    + (features == null ? "null" : features.length));
        }
        for (int i = 0; i < features.length; i++) {
            if (!Double.isFinite(features[i])) {
                throw new IllegalArgumentException("Feature at index " + i + " is not finite: " + features[i]);
            }
        }
    }

    void setRoot(Node root)             { this.root         = root; }
    void setFeatureCount(int count)     { this.featureCount = count; }
    void setMaxDepth(int depth)         { this.maxDepth     = depth; }
    public int getFeatureCount()        { return featureCount; }
    public int getMaxDepth()            { return maxDepth; }
    Node getRoot()                      { return root; }

    public String toTreeString() {
        if (root == null) return "(untrained)";
        StringBuilder sb = new StringBuilder();
        appendNode(sb, root, 0);
        return sb.toString();
    }

    public static final String[] FEATURE_NAMES = {
        "meanIntensity",
        "foregroundRatio",
        "darkPixelRatio",
        "edgeDensity",
        "objectArea",
        "meanSaturation",
        "saturationRatio",
        "largestBlobRatio",
        "vehicleOverlapRatio",
        "vehicleMatchedFlag"
    };

    public String explain(double[] features) {
        if (root == null) return "(untrained)";
        if (features == null || features.length != featureCount) {
            throw new IllegalArgumentException(
                "Expected " + featureCount + " features; got "
                    + (features == null ? "null" : features.length));
        }
        StringBuilder sb = new StringBuilder();
        explainNode(sb, root, features);
        return sb.toString();
    }

    private static void explainNode(StringBuilder sb, Node node, double[] features) {
        if (node instanceof LeafNode leaf) {
            double purityPct = leaf.predictConfidence(features) * 100.0;
            int total = leaf.occupiedCount + leaf.availableCount;
            sb.append(String.format(Locale.ROOT, "Leaf [%s, %.0f%% leaf purity, occ=%d, avail=%d, n=%d]",
                leaf.label, purityPct, leaf.occupiedCount, leaf.availableCount, total));
        } else if (node instanceof InternalNode internal) {
            String fname = internal.featureIndex < FEATURE_NAMES.length
                ? FEATURE_NAMES[internal.featureIndex]
                : "feature[" + internal.featureIndex + "]";
            double val = features[internal.featureIndex];
            if (val <= internal.threshold) {
                sb.append(String.format(Locale.ROOT, "%s (%.4f) <= %.4f -> ", fname, val, internal.threshold));
                explainNode(sb, internal.left, features);
            } else {
                sb.append(String.format(Locale.ROOT, "%s (%.4f) > %.4f -> ", fname, val, internal.threshold));
                explainNode(sb, internal.right, features);
            }
        }
    }

    private static void appendNode(StringBuilder sb, Node node, int depth) {
        String indent = "  ".repeat(depth);
        if (node instanceof LeafNode leaf) {
            int total = leaf.occupiedCount + leaf.availableCount;
            double confPct = leaf.predictConfidence(null) * 100.0;
            if (total > 1) {
                sb.append(String.format(Locale.ROOT, "%s→ %s (%.0f%% purity, occ=%d, avail=%d, n=%d)%n",
                    indent, leaf.label, confPct, leaf.occupiedCount, leaf.availableCount, total));
            } else {
                sb.append(indent).append("→ ").append(leaf.label).append('\n');
            }
        } else if (node instanceof InternalNode internal) {
            String fname = internal.featureIndex < FEATURE_NAMES.length
                ? FEATURE_NAMES[internal.featureIndex]
                : "feature[" + internal.featureIndex + "]";
            sb.append(indent)
              .append("if ").append(fname)
              .append(" <= ").append(String.format(Locale.ROOT, "%.4f", internal.threshold)).append(":\n");
            appendNode(sb, internal.left,  depth + 1);
            sb.append(indent).append("else:\n");
            appendNode(sb, internal.right, depth + 1);
        }
    }
}
