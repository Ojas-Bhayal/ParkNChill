package com.parknchillai.ai;

import com.parknchillai.parking.SlotStatus;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DecisionTreeTrainerTest {

    private static final double[][] FEATURES = {
        {100.0, 0.70, 0.10, 0.40, 0.65}, // OCCUPIED
        {110.0, 0.75, 0.08, 0.38, 0.70}, // OCCUPIED
        {105.0, 0.68, 0.12, 0.42, 0.60}, // OCCUPIED
        {200.0, 0.05, 0.02, 0.03, 0.04}, // AVAILABLE
        {210.0, 0.08, 0.01, 0.04, 0.06}, // AVAILABLE
        {195.0, 0.06, 0.03, 0.02, 0.05}, // AVAILABLE
    };

    private static final SlotStatus[] LABELS = {
        SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED,
        SlotStatus.AVAILABLE, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE
    };

    @Test
    void trainedModelClassifiesAllTrainingSamplesCorrectlyAndIsDeterministic() {
        DecisionTreeTrainer trainer = new DecisionTreeTrainer(5, 1);
        DecisionTreeModel model1 = trainer.train(FEATURES, LABELS);
        DecisionTreeModel model2 = trainer.train(FEATURES, LABELS);

        assertEquals(model1.toTreeString(), model2.toTreeString(), "CART training must be 100% deterministic");

        for (int i = 0; i < FEATURES.length; i++) {
            SlotStatus predicted = model1.predict(FEATURES[i]);
            assertEquals(LABELS[i], predicted,
                "Sample " + i + " should be " + LABELS[i] + " but got " + predicted);
        }
    }

    @Test
    void traceDecisionReturnsStructuredStepsLeafPurityAndDomainBounds() {
        DecisionTreeTrainer trainer = new DecisionTreeTrainer(4, 1);
        DecisionTreeModel model = trainer.train(FEATURES, LABELS);

        DecisionTreeModel.DecisionTrace trace = model.traceDecision(
            new double[]{105.0, 0.72, 0.10, 0.40, 0.65},
            UncertaintyPolicy.defaultProvisionalPolicy()
        );
        assertEquals(SlotStatus.OCCUPIED, trace.finalStatus());
        assertFalse(trace.reviewRequired());
        assertFalse(trace.steps().isEmpty());
        assertEquals(3, trace.leaf().occupiedCount());
        assertEquals(0, trace.leaf().availableCount());
        assertEquals(3, trace.leaf().totalSamples());
        assertEquals(1.0, trace.leaf().leafPurity(), 1e-6);
        assertTrue(trace.humanReadableExplanation().contains("leaf class purity"));
        assertTrue(trace.outOfDomainFeatures().isEmpty());

        // Far out-of-domain sample triggers NEEDS_REVIEW under strictDomainPolicy
        DecisionTreeModel.DecisionTrace oodTrace = model.traceDecision(
            new double[]{250.0, 0.99, 0.99, 0.99, 0.99},
            UncertaintyPolicy.strictDomainPolicy()
        );
        assertTrue(oodTrace.reviewRequired());
        assertEquals(SlotStatus.NEEDS_REVIEW, oodTrace.finalStatus());
        assertFalse(oodTrace.outOfDomainFeatures().isEmpty());
    }

    @Test
    void randomForestClassifierFitsAndPredictsDeterministically() {
        RandomForestClassifier rf = new RandomForestClassifier(11, 4, 1, 42L);
        rf.fit(FEATURES, LABELS);
        assertEquals(11, rf.getNumTrees());
        for (int i = 0; i < FEATURES.length; i++) {
            assertEquals(LABELS[i], rf.predict(FEATURES[i]));
            assertTrue(rf.predictVoteFraction(FEATURES[i]) >= 0.80);
        }
    }

    @Test
    void minSamplesPerLeafPreventsSingleSampleChildLeaves() {
        double[][] feats = {
            {100.0, 0.95, 0.10, 0.50, 0.50},
            {200.0, 0.01, 0.01, 0.01, 0.01},
            {200.0, 0.01, 0.01, 0.01, 0.01},
            {200.0, 0.01, 0.01, 0.01, 0.01},
            {200.0, 0.01, 0.01, 0.01, 0.01},
            {200.0, 0.01, 0.01, 0.01, 0.01}
        };
        SlotStatus[] lbls = {
            SlotStatus.OCCUPIED,
            SlotStatus.AVAILABLE, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE,
            SlotStatus.AVAILABLE, SlotStatus.AVAILABLE
        };
        DecisionTreeTrainer trainer = new DecisionTreeTrainer(4, 2);
        DecisionTreeModel model = trainer.train(feats, lbls);
        assertEquals(SlotStatus.AVAILABLE, model.predict(feats[0]),
            "minSamplesPerLeaf=2 should prevent splitting off a 1-sample leaf");

        // Because the 6-sample root leaf has 5 AVAILABLE and 1 OCCUPIED (purity = 5/6 = 83.3%),
        // a policy with minLeafPurity = 0.90 routes it to NEEDS_REVIEW!
        DecisionTreeModel.DecisionTrace trace = model.traceDecision(
            feats[0], new UncertaintyPolicy(0.90, 2, 5, 0.25, true));
        assertEquals(SlotStatus.NEEDS_REVIEW, trace.finalStatus());
        assertEquals(5.0 / 6.0, trace.leaf().leafPurity(), 1e-6);
    }

    @Test
    void toTreeStringIsNonEmpty() {
        DecisionTreeTrainer trainer = new DecisionTreeTrainer(5, 1);
        DecisionTreeModel   model   = trainer.train(FEATURES, LABELS);
        String tree = model.toTreeString();
        assertFalse(tree.isBlank(), "Tree string should not be blank after training.");
    }

    @Test
    void emptyDatasetThrowsException() {
        DecisionTreeTrainer trainer = new DecisionTreeTrainer(5, 1);
        assertThrows(IllegalArgumentException.class,
            () -> trainer.train(new double[0][], new SlotStatus[0]));
    }

    @Test
    void labelLengthMismatchThrowsException() {
        DecisionTreeTrainer trainer = new DecisionTreeTrainer(5, 1);
        SlotStatus[] wrongLabels = {SlotStatus.OCCUPIED};
        assertThrows(IllegalArgumentException.class,
            () -> trainer.train(FEATURES, wrongLabels));
    }

    @Test
    void untrainedModelThrowsWhenPredicting() {
        DecisionTreeModel model = new DecisionTreeModel();
        assertThrows(IllegalStateException.class,
            () -> model.predict(new double[]{1, 2, 3, 4, 5}));
    }

    @Test
    void wrongFeatureCountThrowsOnPredict() {
        DecisionTreeTrainer trainer = new DecisionTreeTrainer(5, 1);
        DecisionTreeModel   model   = trainer.train(FEATURES, LABELS);
        assertThrows(IllegalArgumentException.class,
            () -> model.predict(new double[]{1.0, 2.0}));
    }

    @Test
    void evaluatorReportsHighAccuracyOnSeparableSet() {
        DecisionTreeTrainer trainer   = new DecisionTreeTrainer(5, 1);
        DecisionTreeModel   model     = trainer.train(FEATURES, LABELS);
        DecisionTreePredictor predictor = new DecisionTreePredictor(model);
        ModelEvaluator.EvaluationReport report =
            ModelEvaluator.evaluate(predictor, FEATURES, LABELS);
        assertEquals(1.0, report.accuracy, 1e-6);
    }

    @Test
    void testFinding09OutOfDomainFeaturesSafeIndexing() {
        DecisionTreeTrainer trainer = new DecisionTreeTrainer(5, 1);
        DecisionTreeModel model = trainer.train(FEATURES, LABELS);

        // Feature array with non-finite value
        double[] invalid = new double[]{100.0, Double.NaN, 0.10, 0.40, 0.65};
        DecisionTreeModel.DecisionTrace trace = model.traceDecision(invalid, UncertaintyPolicy.defaultProvisionalPolicy());
        assertTrue(trace.reviewRequired());
        assertEquals(SlotStatus.NEEDS_REVIEW, trace.finalStatus());

        // Value exceeding valid range for feature index 1 (> 1.0)
        double[] outOfBounds = new double[]{100.0, 999.0, 0.10, 0.40, 0.65};
        DecisionTreeModel.DecisionTrace traceOob = model.traceDecision(outOfBounds, UncertaintyPolicy.defaultProvisionalPolicy());
        assertTrue(traceOob.reviewRequired());
        assertEquals(SlotStatus.NEEDS_REVIEW, traceOob.finalStatus());
    }
}
