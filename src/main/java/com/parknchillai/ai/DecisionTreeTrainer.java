package com.parknchillai.ai;

import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.SlotStatus;

import java.util.*;

/**
 * Trains a {@link DecisionTreeModel} using the CART algorithm with the
 * Gini impurity criterion, records per-leaf class sample counts, and
 * captures the observed training-set feature domain bounds {@code [trainMin, trainMax]}.
 *
 * <p>Key optimizations (WS-1 requirement):
 * <ol>
 *   <li>Evaluates candidate splits in O(F · n log n) per node by sorting primitive indices
 *       once per feature using 3-way in-place quicksort and sweeping with running class counters in O(1).</li>
 *   <li>Reuses primitive arrays and eliminates per-candidate Object/List allocations.</li>
 *   <li>Partitions rows once, strictly after the best split candidate is identified.</li>
 *   <li>Preserves exact deterministic tie-breaking (earlier feature in search order, lower threshold,
 *       strictly {@code gini < bestGini - 1e-12}) and minSamplesPerLeaf constraints.</li>
 * </ol>
 */
public class DecisionTreeTrainer {

    private final int maxDepth;
    private final int minSamplesPerLeaf;

    public DecisionTreeTrainer(int maxDepth, int minSamplesPerLeaf) {
        if (maxDepth < 1) throw new IllegalArgumentException("maxDepth must be >= 1.");
        if (minSamplesPerLeaf < 1)
            throw new IllegalArgumentException("minSamplesPerLeaf must be >= 1.");
        this.maxDepth          = maxDepth;
        this.minSamplesPerLeaf = minSamplesPerLeaf;
    }

    /**
     * Trains a 10-feature (or 8-feature) CART {@link DecisionTreeModel} from human-reviewed
     * {@link SlotTrainingRecord} examples.
     *
     * <p>Strictly rejects any unreviewed provisional record ({@link SlotTrainingRecord#provisional()})
     * so unverified detector predictions are never treated as ground-truth training labels.
     */
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
        int n = features.length;

        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

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

        // Preallocate reusable sort scratch buffer of length n
        int[] sortBuffer = new int[n];
        boolean[] isOccupied = new boolean[n];
        for (int i = 0; i < n; i++) {
            isOccupied[i] = (labels[i] == SlotStatus.OCCUPIED);
        }

        model.setRoot(buildNode(features, isOccupied, indices, 0, featureCount, sortBuffer));
        return model;
    }

    private DecisionTreeModel.Node buildNode(double[][] features, boolean[] isOccupied,
                                             int[] indices, int depth, int featureCount,
                                             int[] sortBuffer) {
        int n = indices.length;
        int occCount = 0;
        for (int idx : indices) {
            if (isOccupied[idx]) occCount++;
        }
        int availCount = n - occCount;
        SlotStatus majority = (occCount >= availCount) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;

        // Base cases: pure node, reached max depth, or fewer samples than 2 * minSamplesPerLeaf
        if (occCount == 0 || availCount == 0 || depth >= maxDepth || n < 2 * minSamplesPerLeaf) {
            return new DecisionTreeModel.LeafNode(majority, occCount, availCount);
        }

        int    bestFeature    = -1;
        double bestThreshold  = 0.0;
        double bestGini       = Double.MAX_VALUE;

        int[] featureOrder = buildFeatureSearchOrder(featureCount);

        for (int f : featureOrder) {
            // Copy node indices into sortBuffer for sorting
            System.arraycopy(indices, 0, sortBuffer, 0, n);
            sortIndicesByFeature(sortBuffer, features, f, 0, n - 1);

            int runningOcc = 0;
            for (int p = 0; p < n - 1; p++) {
                int sampleIdx = sortBuffer[p];
                if (isOccupied[sampleIdx]) {
                    runningOcc++;
                }

                int nextSampleIdx = sortBuffer[p + 1];
                double valCurrent = features[sampleIdx][f];
                double valNext    = features[nextSampleIdx][f];

                // Only evaluate candidate split between strictly distinct adjacent values
                if (valCurrent < valNext) {
                    int leftCount = p + 1;
                    int rightCount = n - leftCount;

                    if (leftCount < minSamplesPerLeaf || rightCount < minSamplesPerLeaf) {
                        continue;
                    }

                    int rightOcc = occCount - runningOcc;
                    double gini = weightedGini(leftCount, runningOcc, rightCount, rightOcc, n);

                    // Deterministic tie-breaking: strictly improves by at least 1e-12
                    // Preserves earlier feature in featureOrder and lower threshold
                    if (gini < bestGini - 1e-12) {
                        bestGini      = gini;
                        bestFeature   = f;
                        bestThreshold = (valCurrent + valNext) / 2.0;
                    }
                }
            }
        }

        if (bestFeature == -1) {
            return new DecisionTreeModel.LeafNode(majority, occCount, availCount);
        }

        // Partition rows ONCE after best split is chosen
        int leftCount = 0;
        for (int idx : indices) {
            if (features[idx][bestFeature] <= bestThreshold) {
                leftCount++;
            }
        }
        int rightCount = n - leftCount;
        int[] leftIndices = new int[leftCount];
        int[] rightIndices = new int[rightCount];
        int lPtr = 0, rPtr = 0;
        for (int idx : indices) {
            if (features[idx][bestFeature] <= bestThreshold) {
                leftIndices[lPtr++] = idx;
            } else {
                rightIndices[rPtr++] = idx;
            }
        }

        DecisionTreeModel.Node leftChild  = buildNode(features, isOccupied, leftIndices, depth + 1, featureCount, sortBuffer);
        DecisionTreeModel.Node rightChild = buildNode(features, isOccupied, rightIndices, depth + 1, featureCount, sortBuffer);
        return new DecisionTreeModel.InternalNode(bestFeature, bestThreshold, leftChild, rightChild);
    }

    /**
     * In-place 3-way quicksort sorting indices by {@code features[idx[k]][f]}.
     * Efficiently handles duplicate values in O(N).
     */
    private static void sortIndicesByFeature(int[] idx, double[][] features, int f, int low, int high) {
        if (high <= low) return;
        if (high - low < 16) {
            for (int i = low + 1; i <= high; i++) {
                int key = idx[i];
                double keyVal = features[key][f];
                int j = i - 1;
                while (j >= low && features[idx[j]][f] > keyVal) {
                    idx[j + 1] = idx[j];
                    j--;
                }
                idx[j + 1] = key;
            }
            return;
        }

        // 3-way partitioning
        int mid = low + (high - low) / 2;
        swap(idx, low, mid);
        double pivot = features[idx[low]][f];

        int lt = low;      // idx[low..lt-1] < pivot
        int gt = high;     // idx[gt+1..high] > pivot
        int i = low + 1;   // idx[lt..i-1] == pivot

        while (i <= gt) {
            double v = features[idx[i]][f];
            if (v < pivot) {
                swap(idx, lt++, i++);
            } else if (v > pivot) {
                swap(idx, i, gt--);
            } else {
                i++;
            }
        }

        sortIndicesByFeature(idx, features, f, low, lt - 1);
        sortIndicesByFeature(idx, features, f, gt + 1, high);
    }

    private static void swap(int[] arr, int i, int j) {
        int t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
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

    private static double weightedGini(int leftSize, int leftOcc, int rightSize, int rightOcc, int total) {
        double pL = (double) leftOcc / leftSize;
        double giniL = 1.0 - (pL * pL + (1.0 - pL) * (1.0 - pL));
        double pR = (double) rightOcc / rightSize;
        double giniR = 1.0 - (pR * pR + (1.0 - pR) * (1.0 - pR));
        return ((double) leftSize / total) * giniL + ((double) rightSize / total) * giniR;
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
