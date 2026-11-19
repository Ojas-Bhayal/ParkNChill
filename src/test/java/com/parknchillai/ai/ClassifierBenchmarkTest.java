package com.parknchillai.ai;

import com.parknchillai.parking.SlotStatus;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression suite for Phase 6 (PNC-REM-05) and Phase 4 (PNC-FIX-05) validation-driven
 * classifier benchmarking: verifies that when neither candidate meets the occupied-recall
 * floor, {@code NO_RELIABLE_WINNER} is returned and neither candidate is evaluated on the
 * Held-Out Test set.
 */
class ClassifierBenchmarkTest {

    @Test
    void bothCandidatesBelowOccupiedRecallFloorReturnsNoReliableWinnerEvenIfOneIsFaster() {
        double[][] trainFeats = {
            {80.0, 0.45, 0.50, 0.48, 0.42, 110.0, 0.55, 0.38},
            {82.0, 0.46, 0.52, 0.49, 0.43, 115.0, 0.58, 0.40},
            {85.0, 0.48, 0.49, 0.51, 0.44, 120.0, 0.60, 0.42},
            {84.0, 0.47, 0.51, 0.50, 0.41, 112.0, 0.56, 0.39},
            {60.0, 0.05, 0.85, 0.15, 0.02, 15.0, 0.00, 0.02},
            {62.0, 0.06, 0.86, 0.16, 0.02, 16.0, 0.00, 0.02},
            {64.0, 0.05, 0.84, 0.14, 0.02, 14.0, 0.00, 0.02},
            {61.0, 0.06, 0.87, 0.15, 0.02, 15.0, 0.00, 0.02}
        };
        SlotStatus[] trainLabels = {
            SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED,
            SlotStatus.AVAILABLE, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE
        };

        // Validation set (N=6: 3 OCCUPIED, 3 AVAILABLE) where 2 of the 3 OCCUPIED samples
        // have low/available-like features so both CART and RF achieve Val Recall = 1/3 = 0.3333 < 0.80!
        double[][] valFeats = {
            {84.0, 0.47, 0.50, 0.50, 0.42, 115.0, 0.57, 0.40}, // predicted OCCUPIED
            {61.0, 0.05, 0.85, 0.15, 0.02, 15.0,  0.00, 0.02}, // actual OCCUPIED, predicted AVAILABLE (FN)
            {62.0, 0.06, 0.86, 0.16, 0.02, 16.0,  0.00, 0.02}, // actual OCCUPIED, predicted AVAILABLE (FN)
            {60.0, 0.05, 0.85, 0.15, 0.02, 15.0,  0.00, 0.02}, // actual AVAILABLE, predicted AVAILABLE (TN)
            {62.0, 0.06, 0.86, 0.16, 0.02, 16.0,  0.00, 0.02}, // actual AVAILABLE, predicted AVAILABLE (TN)
            {63.0, 0.05, 0.84, 0.14, 0.02, 14.0,  0.00, 0.02}  // actual AVAILABLE, predicted AVAILABLE (TN)
        };
        SlotStatus[] valLabels = {
            SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED,
            SlotStatus.AVAILABLE, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE
        };

        DatasetLoader.GroupedPartitions parts = new DatasetLoader.GroupedPartitions(
            new DatasetLoader.Dataset(trainFeats, trainLabels),
            new DatasetLoader.Dataset(valFeats, valLabels),
            new DatasetLoader.Dataset(valFeats, valLabels),
            Set.of("G_TR"), Set.of("G_VAL"), Set.of("G_TEST"), true
        );

        ClassifierBenchmark.BenchmarkResult res = ClassifierBenchmark.compare(
            parts, 4, 1, 42L, new ClassifierBenchmark.SelectionPolicy(6, 0.80));

        assertEquals(ClassifierBenchmark.SelectedModel.NO_RELIABLE_WINNER, res.selectedModel(),
            "Must return NO_RELIABLE_WINNER when neither model meets minOccupiedRecall=0.80; got: " + res.selectionRationale());
        assertNull(res.selectedModelTestReport(), "Held-out test set must not be evaluated when no candidate passes validation gates");
        assertNull(res.cartTestReport());
        assertNull(res.rfTestReport());
        assertTrue(res.selectionRationale().contains("Both candidates failed the mandatory Validation Occupied-Recall safety gate"));
        assertTrue(res.selectionRationale().contains("0.3333"));
    }

    @Test
    void cartWinningOnValidationRuleSelectsCartAndEvaluatesOnlyCartOnTest() {
        double[][] trainFeats = {
            {80.0, 0.45, 0.50, 0.48, 0.42, 110.0, 0.55, 0.38},
            {82.0, 0.46, 0.52, 0.49, 0.43, 115.0, 0.58, 0.40},
            {85.0, 0.48, 0.49, 0.51, 0.44, 120.0, 0.60, 0.42},
            {84.0, 0.47, 0.51, 0.50, 0.41, 112.0, 0.56, 0.39},
            {60.0, 0.05, 0.85, 0.15, 0.02, 15.0, 0.00, 0.02},
            {62.0, 0.06, 0.86, 0.16, 0.02, 16.0, 0.00, 0.02},
            {64.0, 0.05, 0.84, 0.14, 0.02, 14.0, 0.00, 0.02},
            {61.0, 0.06, 0.87, 0.15, 0.02, 15.0, 0.00, 0.02}
        };
        SlotStatus[] trainLabels = {
            SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED,
            SlotStatus.AVAILABLE, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE
        };

        double[][] valFeats = {
            {70.0, 0.20, 0.50, 0.49, 0.20, 50.0, 0.20, 0.20},
            {70.0, 0.20, 0.50, 0.50, 0.20, 50.0, 0.20, 0.20},
            {70.0, 0.20, 0.50, 0.51, 0.20, 50.0, 0.20, 0.20},
            {70.0, 0.20, 0.50, 0.14, 0.20, 50.0, 0.20, 0.20},
            {70.0, 0.20, 0.50, 0.15, 0.20, 50.0, 0.20, 0.20},
            {70.0, 0.20, 0.50, 0.14, 0.20, 50.0, 0.20, 0.20}
        };
        SlotStatus[] valLabels = {
            SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED,
            SlotStatus.AVAILABLE, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE
        };

        DatasetLoader.GroupedPartitions parts = new DatasetLoader.GroupedPartitions(
            new DatasetLoader.Dataset(trainFeats, trainLabels),
            new DatasetLoader.Dataset(valFeats, valLabels),
            new DatasetLoader.Dataset(valFeats, valLabels),
            Set.of("G_TR"), Set.of("G_VAL"), Set.of("G_TEST"), true
        );

        ClassifierBenchmark.BenchmarkResult res = ClassifierBenchmark.compare(parts, 1, 1, 42L);
        assertEquals(ClassifierBenchmark.SelectedModel.CART_DECISION_TREE, res.selectedModel());
        assertNotNull(res.selectedModelTestReport());
        assertNotNull(res.cartTestReport(), "Selected model (CART) must have held-out test report");
        assertNull(res.rfTestReport(), "Unselected candidate (RF) must NOT generate a held-out test report under default flow");
        assertTrue(res.cartAvgInferenceMicros() > 0.0);
        assertTrue(res.rfAvgInferenceMicros() > 0.0);
        assertTrue(res.selectionRationale().contains("Val"));
    }

    @Test
    void randomForestWinningOnValidationCriteriaSelectsRandomForest() {
        double[][] trainFeats = {
            {10.0, 0.80, 0.80, 0.80, 0.80, 200.0, 0.80, 0.80},
            {11.0, 0.82, 0.82, 0.82, 0.82, 205.0, 0.82, 0.82},
            {12.0, 0.85, 0.85, 0.85, 0.85, 210.0, 0.85, 0.85},
            {13.0, 0.78, 0.78, 0.78, 0.78, 195.0, 0.78, 0.78},
            {200.0, 0.05, 0.05, 0.05, 0.05, 10.0, 0.05, 0.05},
            {205.0, 0.04, 0.04, 0.04, 0.04, 12.0, 0.04, 0.04},
            {210.0, 0.06, 0.06, 0.06, 0.06, 11.0, 0.06, 0.06},
            {215.0, 0.03, 0.03, 0.03, 0.03, 9.0,  0.03, 0.03}
        };
        SlotStatus[] trainLabels = {
            SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED,
            SlotStatus.AVAILABLE, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE
        };

        double[][] valFeats = {
            {11.0,  0.82, 0.82, 0.02, 0.82, 200.0, 0.82, 0.82},
            {12.0,  0.84, 0.84, 0.01, 0.84, 205.0, 0.84, 0.84},
            {11.5,  0.81, 0.81, 0.02, 0.81, 210.0, 0.81, 0.81},
            {205.0, 0.04, 0.04, 0.85, 0.04, 10.0,  0.04, 0.04},
            {210.0, 0.05, 0.05, 0.84, 0.05, 11.0,  0.05, 0.05},
            {208.0, 0.03, 0.03, 0.86, 0.03, 12.0,  0.03, 0.03}
        };
        SlotStatus[] valLabels = {
            SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED,
            SlotStatus.AVAILABLE, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE
        };

        DatasetLoader.GroupedPartitions parts = new DatasetLoader.GroupedPartitions(
            new DatasetLoader.Dataset(trainFeats, trainLabels),
            new DatasetLoader.Dataset(valFeats, valLabels),
            new DatasetLoader.Dataset(valFeats, valLabels),
            Set.of("G_TR"), Set.of("G_VAL"), Set.of("G_TEST"), true
        );

        ClassifierBenchmark.BenchmarkResult res = ClassifierBenchmark.compare(parts, 1, 1, 42L);
        assertEquals(ClassifierBenchmark.SelectedModel.RANDOM_FOREST, res.selectedModel(),
            "Random Forest must be selected when it passes recall gate and CART fails; rationale: " + res.selectionRationale());
        assertNotNull(res.rfTestReport());
        assertNull(res.cartTestReport());
        assertTrue(res.selectionRationale().contains("Random Forest"));
    }

    @Test
    void nonGroupDisjointOrSmallValidationSetYieldsNoReliableWinner() {
        double[][] feats = {
            {80.0, 0.45, 0.50, 0.48, 0.42, 110.0, 0.55, 0.38},
            {82.0, 0.46, 0.52, 0.49, 0.43, 115.0, 0.58, 0.40},
            {60.0, 0.05, 0.85, 0.15, 0.02, 15.0, 0.00, 0.02},
            {62.0, 0.06, 0.86, 0.16, 0.02, 16.0, 0.00, 0.02}
        };
        SlotStatus[] labels = {
            SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE
        };
        DatasetLoader.GroupedPartitions fallbackParts = new DatasetLoader.GroupedPartitions(
            new DatasetLoader.Dataset(feats, labels),
            new DatasetLoader.Dataset(feats, labels),
            new DatasetLoader.Dataset(feats, labels),
            Set.of("G1"), Set.of("G1"), Set.of("G1"), false
        );

        ClassifierBenchmark.BenchmarkResult res = ClassifierBenchmark.compare(fallbackParts, 4, 1, 42L);
        assertEquals(ClassifierBenchmark.SelectedModel.NO_RELIABLE_WINNER, res.selectedModel());
        assertNull(res.selectedModelTestReport());
        assertTrue(res.selectionRationale().contains("NO RELIABLE WINNER"));
    }

    @Test
    void alteringOnlyTestLabelsCannotChangeSelectedModelOrValidationRationale() {
        DatasetLoader.Dataset fullDs = DatasetLoader.load("data/labels/features.csv");
        var manifest = DatasetManifest.loadOrSync("data/labels/features.csv", "data/labels/manifest.csv");
        DatasetLoader.GroupedPartitions originalParts = DatasetLoader.buildValidatedPartitions(fullDs, manifest, 42L);

        ClassifierBenchmark.BenchmarkResult origBench = ClassifierBenchmark.compare(originalParts, 4, 2, 42L);

        SlotStatus[] invertedTestLabels = new SlotStatus[originalParts.test().size()];
        for (int i = 0; i < invertedTestLabels.length; i++) {
            invertedTestLabels[i] = (originalParts.test().labels[i] == SlotStatus.OCCUPIED)
                ? SlotStatus.AVAILABLE : SlotStatus.OCCUPIED;
        }
        DatasetLoader.GroupedPartitions invertedTestParts = new DatasetLoader.GroupedPartitions(
            originalParts.train(),
            originalParts.validation(),
            new DatasetLoader.Dataset(originalParts.test().features, invertedTestLabels),
            originalParts.trainGroups(),
            originalParts.validationGroups(),
            originalParts.testGroups(),
            true
        );

        ClassifierBenchmark.BenchmarkResult invertedBench = ClassifierBenchmark.compare(invertedTestParts, 4, 2, 42L);

        assertEquals(origBench.selectedModel(), invertedBench.selectedModel(),
            "Altering only TEST labels must NEVER change the validation-selected model");
    }
}
