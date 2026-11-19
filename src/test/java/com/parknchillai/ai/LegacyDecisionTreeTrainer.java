package com.parknchillai.ai;

import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.SlotStatus;

import java.util.*;

/**
 * Baseline CART trainer preserved in test scope as the differential-testing oracle
 * and 'before' benchmark baseline (WS-1 requirement).
 */
public class LegacyDecisionTreeTrainer {

    private final int maxDepth;
    private final int minSamplesPerLeaf;

    public LegacyDecisionTreeTrainer(int maxDepth, int minSamplesPerLeaf) {
        if (maxDepth < 1) throw new IllegalArgumentException("maxDepth must be >= 1.");
        if (minSamplesPerLeaf < 1)
            throw new IllegalArgumentException("minSamplesPerLeaf must be >= 1.");
        this.maxDepth          = maxDepth;
        this.minSamplesPerLeaf = minSamplesPerLeaf;
    }

    public DecisionTreeModel trainFromReviewedSlotRecords(List<SlotTrainingRecord> reviewedRecords) {
        return trainFromReviewedSlotRecords(reviewedRecords, FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT);
    }

    public DecisionTreeModel trainFromReviewedSlotRecords(List<SlotTrainingRecord> reviewedRecords,
                                                          int targetFeatureCount) {
        if (reviewedRecords == null || reviewedRecords.isEmpty()) {
            throw new IllegalArgumentException("Reviewed slot training records must not be null or empty.");
        }
        double[][] features = new double[reviewedRecords.size()][];
        SlotStatus[] labels = new SlotStatus[reviewedRecords.size()];
        for (int i = 0; i < reviewedRecords.size(); i++) {
            SlotTrainingRecord rec = reviewedRecords.get(i);
            if (rec == null) {
                throw new IllegalArgumentException("SlotTrainingRecord at index " + i + " must not be null.");
            }
            if (rec.provisional() || !rec.isHumanVerified()) {
                throw new IllegalArgumentException(
                    "Cannot train Decision Tree on unreviewed provisional slot record '" + rec.sampleId()
                    + "' (reviewStatus=" + rec.reviewStatus() + "). Call promoteToHumanVerified() first.");
            }
            features[i] = rec.features().toArray(targetFeatureCount);
            labels[i]   = rec.occupancyLabel();
        }
        return train(features, labels);
    }

    public DecisionTreeModel train(double[][] features, SlotStatus[] labels) {
        validateDataset(features, labels);
        int featureCount = features[0].length;
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < features.length; i++) indices.add(i);

        double[] minBounds = new double[featureCount];
        double[] maxBounds = new double[featureCount];
        Arrays.fill(minBounds, Double.POSITIVE_INFINITY);
        Arrays.fill(maxBounds, Double.NEGATIVE_INFINITY);
        for (double[] row : features) {
            for (int f = 0; f < featureCount; f++) {
                if (row[f] < minBounds[f]) minBounds[f] = row[f];
                if (row[f] > maxBounds[f]) maxBounds[f] = row[f];
            }
        }

        DecisionTreeModel model = new DecisionTreeModel();
        model.setFeatureCount(featureCount);
        model.setMaxDepth(maxDepth);
        model.setTrainDomainBounds(minBounds, maxBounds);
        model.setRoot(buildNode(features, labels, indices, 0, featureCount));
        return model;
    }

    private DecisionTreeModel.Node buildNode(double[][] features, SlotStatus[] labels,
                                              List<Integer> indices, int depth, int featureCount) {
        int occCount = countClass(labels, indices, SlotStatus.OCCUPIED);
        int availCount = indices.size() - occCount;
        SlotStatus majority = (occCount >= availCount) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;

        if (isPure(labels, indices) || depth >= maxDepth || indices.size() < 2 * minSamplesPerLeaf) {
            return new DecisionTreeModel.LeafNode(majority, occCount, availCount);
        }

        int   bestFeature    = -1;
        double bestThreshold = 0.0;
        double bestGini      = Double.MAX_VALUE;
        List<Integer> bestLeft = null, bestRight = null;

        int[] featureOrder = buildFeatureSearchOrder(featureCount);

        for (int f : featureOrder) {
            final int featureIdx = f;
            double[] vals = indices.stream()
                .mapToDouble(i -> features[i][featureIdx])
                .sorted()
                .distinct()
                .toArray();

            for (int k = 0; k < vals.length - 1; k++) {
                double threshold = (vals[k] + vals[k + 1]) / 2.0;
                List<Integer> left  = new ArrayList<>();
                List<Integer> right = new ArrayList<>();
                for (int i : indices) {
                    (features[i][featureIdx] <= threshold ? left : right).add(i);
                }
                if (left.size() < minSamplesPerLeaf || right.size() < minSamplesPerLeaf) continue;

                double gini = weightedGini(labels, left, right, indices.size());
                if (gini < bestGini - 1e-12) {
                    bestGini      = gini;
                    bestFeature   = f;
                    bestThreshold = threshold;
                    bestLeft      = left;
                    bestRight     = right;
                }
            }
        }

        if (bestFeature == -1) {
            return new DecisionTreeModel.LeafNode(majority, occCount, availCount);
        }

        DecisionTreeModel.Node leftChild  = buildNode(features, labels, bestLeft,  depth + 1, featureCount);
        DecisionTreeModel.Node rightChild = buildNode(features, labels, bestRight, depth + 1, featureCount);
        return new DecisionTreeModel.InternalNode(bestFeature, bestThreshold, leftChild, rightChild);
    }

    private static int[] buildFeatureSearchOrder(int featureCount) {
        if (featureCount == 10) {
            return new int[]{8, 9, 3, 7, 4, 6, 1, 5, 2, 0};
        }
        if (featureCount == 8) {
            return new int[]{3, 7, 4, 6, 1, 5, 2, 0};
        }
        if (featureCount == 5) {
            return new int[]{3, 4, 1, 2, 0};
        }
        int[] order = new int[featureCount];
        for (int i = 0; i < featureCount; i++) order[i] = i;
        return order;
    }

    private static double weightedGini(SlotStatus[] labels,
                                       List<Integer> left, List<Integer> right,
                                       int total) {
        return ((double) left.size()  / total) * gini(labels, left)
             + ((double) right.size() / total) * gini(labels, right);
    }

    private static double gini(SlotStatus[] labels, List<Integer> indices) {
        if (indices.isEmpty()) return 0.0;
        int occupied = countClass(labels, indices, SlotStatus.OCCUPIED);
        double p = (double) occupied / indices.size();
        return 1.0 - (p * p + (1 - p) * (1 - p));
    }

    private static boolean isPure(SlotStatus[] labels, List<Integer> indices) {
        SlotStatus first = labels[indices.get(0)];
        for (int i : indices) if (labels[i] != first) return false;
        return true;
    }

    private static int countClass(SlotStatus[] labels, List<Integer> indices, SlotStatus target) {
        int count = 0;
        for (int i : indices) {
            if (labels[i] == target) count++;
        }
        return count;
    }

    private static void validateDataset(double[][] features, SlotStatus[] labels) {
        if (features == null || features.length == 0) {
            throw new IllegalArgumentException("Training dataset must not be empty.");
        }
        if (labels == null || labels.length != features.length) {
            throw new IllegalArgumentException(
                "Labels array length must match features row count.");
        }
        int cols = features[0] == null ? 0 : features[0].length;
        if (cols == 0) {
            throw new IllegalArgumentException("Feature rows must contain at least one feature.");
        }
        for (int r = 0; r < features.length; r++) {
            double[] row = features[r];
            if (row == null || row.length != cols) {
                throw new IllegalArgumentException(
                    "All feature rows must have the same length (" + cols + ").");
            }
            for (int c = 0; c < cols; c++) {
                if (!Double.isFinite(row[c])) {
                    throw new IllegalArgumentException(
                        "Training feature at row " + r + ", col " + c + " is not finite: " + row[c]);
                }
            }
            if (labels[r] == null || labels[r] == SlotStatus.NEEDS_REVIEW || labels[r] == SlotStatus.UNCLASSIFIED) {
                throw new IllegalArgumentException(
                    "Training label at row " + r + " must be OCCUPIED or AVAILABLE; got " + labels[r]);
            }
        }
    }
}
