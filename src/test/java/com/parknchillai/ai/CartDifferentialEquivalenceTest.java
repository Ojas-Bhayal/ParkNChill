package com.parknchillai.ai;

import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.SlotStatus;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("WS-1: CART Differential Equivalence Tests (Optimized vs Legacy Oracle)")
class CartDifferentialEquivalenceTest {

    private static final double EPSILON = 1e-9;

    record SyntheticData(double[][] features, SlotStatus[] labels) {}

    @ParameterizedTest(name = "seed={0}, N={1}, maxDepth={2}, minSamplesSplit={3}")
    @CsvSource({
        "42, 50, 3, 2",
        "42, 200, 5, 2",
        "101, 500, 5, 5",
        "2024, 1000, 8, 10",
        "777, 250, 4, 3"
    })
    void testExactEquivalenceOnSyntheticData(long seed, int sampleCount, int maxDepth, int minSamplesSplit) {
        Random rng = new Random(seed);
        SyntheticData data = generateSyntheticSamples(rng, sampleCount, 10);

        DecisionTreeTrainer optimizedTrainer = new DecisionTreeTrainer(maxDepth, minSamplesSplit);
        LegacyDecisionTreeTrainer legacyTrainer = new LegacyDecisionTreeTrainer(maxDepth, minSamplesSplit);

        DecisionTreeModel optModel = optimizedTrainer.train(data.features, data.labels);
        DecisionTreeModel legModel = legacyTrainer.train(data.features, data.labels);

        assertNotNull(optModel);
        assertNotNull(legModel);

        // 1. Recursive tree equivalence
        assertTreeNodesEquivalent(optModel.getRoot(), legModel.getRoot());

        // 2. Tree string representation equivalence
        assertEquals(legModel.toTreeString(), optModel.toTreeString());

        // 3. Domain bounds equivalence
        for (int i = 0; i < 10; i++) {
            assertEquals(legModel.getTrainMin(i), optModel.getTrainMin(i), EPSILON, "trainMin[" + i + "]");
            assertEquals(legModel.getTrainMax(i), optModel.getTrainMax(i), EPSILON, "trainMax[" + i + "]");
        }

        // 4. Prediction and confidence equivalence on test vectors
        for (int i = 0; i < 200; i++) {
            double[] testFeat = new double[10];
            for (int f = 0; f < 10; f++) {
                testFeat[f] = rng.nextDouble();
            }
            SlotStatus legPred = legModel.predict(testFeat);
            SlotStatus optPred = optModel.predict(testFeat);
            assertEquals(legPred, optPred, "Prediction mismatch at test vector " + i);

            double legConf = legModel.predictConfidence(testFeat);
            double optConf = optModel.predictConfidence(testFeat);
            assertEquals(legConf, optConf, EPSILON, "Confidence mismatch at test vector " + i);

            assertEquals(legModel.explain(testFeat), optModel.explain(testFeat));
        }
    }

    @Test
    void testExactEquivalenceWithDuplicateValuesAndTies() {
        Random rng = new Random(999);
        int sampleCount = 300;
        double[][] features = new double[sampleCount][10];
        SlotStatus[] labels = new SlotStatus[sampleCount];

        for (int i = 0; i < sampleCount; i++) {
            // Feature 0 has only 3 discrete values
            features[i][0] = (i % 3) * 0.5;
            // Feature 1 is completely constant
            features[i][1] = 0.42;
            // Feature 2 is identical for half the rows
            features[i][2] = (i < 150) ? 0.1 : 0.9;
            // Other features random quantized
            for (int f = 3; f < 10; f++) {
                features[i][f] = Math.round(rng.nextDouble() * 4.0) / 4.0;
            }
            labels[i] = (features[i][0] + features[i][2] > 0.8) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
        }

        DecisionTreeTrainer optimizedTrainer = new DecisionTreeTrainer(6, 4);
        LegacyDecisionTreeTrainer legacyTrainer = new LegacyDecisionTreeTrainer(6, 4);

        DecisionTreeModel optModel = optimizedTrainer.train(features, labels);
        DecisionTreeModel legModel = legacyTrainer.train(features, labels);

        assertTreeNodesEquivalent(optModel.getRoot(), legModel.getRoot());
        assertEquals(legModel.toTreeString(), optModel.toTreeString());
    }

    @Test
    void testIdenticalRowsAndPureSubsets() {
        int sampleCount = 40;
        double[][] features = new double[sampleCount][10];
        SlotStatus[] labels = new SlotStatus[sampleCount];

        // All identical available samples
        for (int i = 0; i < 20; i++) {
            features[i] = new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
            labels[i] = SlotStatus.AVAILABLE;
        }
        // All identical occupied samples
        for (int i = 20; i < 40; i++) {
            features[i] = new double[]{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0};
            labels[i] = SlotStatus.OCCUPIED;
        }

        DecisionTreeTrainer optimizedTrainer = new DecisionTreeTrainer(5, 2);
        LegacyDecisionTreeTrainer legacyTrainer = new LegacyDecisionTreeTrainer(5, 2);

        DecisionTreeModel optModel = optimizedTrainer.train(features, labels);
        DecisionTreeModel legModel = legacyTrainer.train(features, labels);

        assertTreeNodesEquivalent(optModel.getRoot(), legModel.getRoot());
        assertEquals(legModel.toTreeString(), optModel.toTreeString());
    }

    private static SyntheticData generateSyntheticSamples(Random rng, int count, int featureCount) {
        double[][] features = new double[count][featureCount];
        SlotStatus[] labels = new SlotStatus[count];
        for (int i = 0; i < count; i++) {
            for (int f = 0; f < featureCount; f++) {
                features[i][f] = rng.nextDouble();
            }
            boolean occupied = (features[i][8] * 0.4 + features[i][3] * 0.3 + features[i][1] * 0.3) > 0.5;
            labels[i] = occupied ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
        }
        return new SyntheticData(features, labels);
    }

    private static void assertTreeNodesEquivalent(DecisionTreeModel.Node optNode, DecisionTreeModel.Node legNode) {
        if (optNode == null && legNode == null) return;
        assertNotNull(optNode, "Optimized node is null but legacy node is not");
        assertNotNull(legNode, "Legacy node is null but optimized node is not");

        if (legNode instanceof DecisionTreeModel.LeafNode legLeaf) {
            assertTrue(optNode instanceof DecisionTreeModel.LeafNode, "Expected leaf node but got internal node");
            DecisionTreeModel.LeafNode optLeaf = (DecisionTreeModel.LeafNode) optNode;
            assertEquals(legLeaf.label, optLeaf.label, "Leaf label mismatch");
            assertEquals(legLeaf.occupiedCount, optLeaf.occupiedCount, "Leaf occupiedCount mismatch");
            assertEquals(legLeaf.availableCount, optLeaf.availableCount, "Leaf availableCount mismatch");
        } else if (legNode instanceof DecisionTreeModel.InternalNode legInternal) {
            assertTrue(optNode instanceof DecisionTreeModel.InternalNode, "Expected internal node but got leaf node");
            DecisionTreeModel.InternalNode optInternal = (DecisionTreeModel.InternalNode) optNode;
            assertEquals(legInternal.featureIndex, optInternal.featureIndex, "Internal node featureIndex mismatch");
            assertEquals(legInternal.threshold, optInternal.threshold, EPSILON, "Internal node threshold mismatch");

            assertTreeNodesEquivalent(optInternal.left, legInternal.left);
            assertTreeNodesEquivalent(optInternal.right, legInternal.right);
        }
    }
}
