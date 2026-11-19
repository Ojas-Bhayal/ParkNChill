package com.parknchillai.ai;

import com.parknchillai.parking.SlotStatus;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Reproducible traditional Random Forest classifier (bootstrap-aggregated ensemble
 * of CART decision trees with random feature subspaces {@code m = ceil(sqrt(F))} at each split)
 * used for the Phase 7 comparative baseline experiment against the single CART Decision Tree.
 */
public final class RandomForestClassifier {

    private final int numTrees;
    private final int maxDepth;
    private final int minSamplesPerLeaf;
    private final long seed;
    private final List<DecisionTreeModel> trees = new ArrayList<>();
    private int featureCount;

    public RandomForestClassifier(int numTrees, int maxDepth, int minSamplesPerLeaf, long seed) {
        if (numTrees < 1) throw new IllegalArgumentException("numTrees must be >= 1.");
        if (maxDepth < 1) throw new IllegalArgumentException("maxDepth must be >= 1.");
        if (minSamplesPerLeaf < 1) throw new IllegalArgumentException("minSamplesPerLeaf must be >= 1.");
        this.numTrees          = numTrees;
        this.maxDepth          = maxDepth;
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.seed              = seed;
    }

    public void fit(double[][] features, SlotStatus[] labels) {
        if (features == null || features.length == 0 || labels == null || labels.length != features.length) {
            throw new IllegalArgumentException("Invalid training dataset for RandomForestClassifier.");
        }
        this.featureCount = features[0].length;
        this.trees.clear();

        int n = features.length;
        int mTry = Math.max(1, (int) Math.ceil(Math.sqrt(featureCount)));
        Random masterRng = new Random(seed);

        for (int t = 0; t < numTrees; t++) {
            long treeSeed = masterRng.nextLong();
            Random treeRng = new Random(treeSeed);

            // Bootstrap sample with replacement
            double[][] bootF = new double[n][featureCount];
            SlotStatus[] bootL = new SlotStatus[n];
            for (int i = 0; i < n; i++) {
                int pick = treeRng.nextInt(n);
                bootF[i] = features[pick];
                bootL[i] = labels[pick];
            }

            DecisionTreeModel tree = buildSubspaceTree(bootF, bootL, mTry, treeRng);
            trees.add(tree);
        }
    }

    public SlotStatus predict(double[] features) {
        if (trees.isEmpty()) throw new IllegalStateException("RandomForestClassifier has not been fitted.");
        int occVotes = 0;
        for (DecisionTreeModel tree : trees) {
            if (tree.predict(features) == SlotStatus.OCCUPIED) {
                occVotes++;
            }
        }
        int availVotes = trees.size() - occVotes;
        return occVotes >= availVotes ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
    }

    public double predictVoteFraction(double[] features) {
        if (trees.isEmpty()) throw new IllegalStateException("RandomForestClassifier has not been fitted.");
        int occVotes = 0;
        for (DecisionTreeModel tree : trees) {
            if (tree.predict(features) == SlotStatus.OCCUPIED) {
                occVotes++;
            }
        }
        int winning = Math.max(occVotes, trees.size() - occVotes);
        return (double) winning / trees.size();
    }

    public int getNumTrees() { return numTrees; }

    private DecisionTreeModel buildSubspaceTree(double[][] features,
                                                SlotStatus[] labels,
                                                int mTry,
                                                Random rng) {
        List<Integer> indices = new ArrayList<>(features.length);
        for (int i = 0; i < features.length; i++) indices.add(i);

        DecisionTreeModel model = new DecisionTreeModel();
        model.setFeatureCount(featureCount);
        model.setMaxDepth(maxDepth);
        model.setRoot(buildSubspaceNode(features, labels, indices, 0, mTry, rng));
        return model;
    }

    private DecisionTreeModel.Node buildSubspaceNode(double[][] features,
                                                     SlotStatus[] labels,
                                                     List<Integer> indices,
                                                     int depth,
                                                     int mTry,
                                                     Random rng) {
        int occCount = 0;
        for (int idx : indices) if (labels[idx] == SlotStatus.OCCUPIED) occCount++;
        int availCount = indices.size() - occCount;
        SlotStatus majority = (occCount >= availCount) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;

        if (occCount == 0 || availCount == 0 || depth >= maxDepth || indices.size() < 2 * minSamplesPerLeaf) {
            return new DecisionTreeModel.LeafNode(majority, occCount, availCount);
        }

        List<Integer> allFeatures = new ArrayList<>(featureCount);
        for (int f = 0; f < featureCount; f++) allFeatures.add(f);
        Collections.shuffle(allFeatures, rng);
        List<Integer> candidateFeatures = allFeatures.subList(0, Math.min(mTry, featureCount));

        int bestFeature = -1;
        double bestThreshold = 0.0;
        double bestGini = Double.MAX_VALUE;
        List<Integer> bestLeft = null, bestRight = null;

        for (int f : candidateFeatures) {
            final int fIdx = f;
            double[] vals = indices.stream().mapToDouble(i -> features[i][fIdx]).sorted().distinct().toArray();
            for (int k = 0; k < vals.length - 1; k++) {
                double threshold = (vals[k] + vals[k + 1]) / 2.0;
                List<Integer> left = new ArrayList<>();
                List<Integer> right = new ArrayList<>();
                for (int i : indices) {
                    (features[i][fIdx] <= threshold ? left : right).add(i);
                }
                if (left.size() < minSamplesPerLeaf || right.size() < minSamplesPerLeaf) continue;

                double g = weightedGini(labels, left, right, indices.size());
                if (g < bestGini - 1e-12) {
                    bestGini = g;
                    bestFeature = f;
                    bestThreshold = threshold;
                    bestLeft = left;
                    bestRight = right;
                }
            }
        }

        if (bestFeature == -1) {
            return new DecisionTreeModel.LeafNode(majority, occCount, availCount);
        }

        return new DecisionTreeModel.InternalNode(
            bestFeature,
            bestThreshold,
            buildSubspaceNode(features, labels, bestLeft, depth + 1, mTry, rng),
            buildSubspaceNode(features, labels, bestRight, depth + 1, mTry, rng)
        );
    }

    private static double weightedGini(SlotStatus[] labels, List<Integer> left, List<Integer> right, int total) {
        return ((double) left.size() / total) * gini(labels, left)
             + ((double) right.size() / total) * gini(labels, right);
    }

    private static double gini(SlotStatus[] labels, List<Integer> indices) {
        if (indices.isEmpty()) return 0.0;
        int occ = 0;
        for (int i : indices) if (labels[i] == SlotStatus.OCCUPIED) occ++;
        double p = (double) occ / indices.size();
        return 1.0 - (p * p + (1.0 - p) * (1.0 - p));
    }
}
