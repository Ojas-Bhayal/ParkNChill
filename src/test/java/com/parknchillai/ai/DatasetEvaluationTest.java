package com.parknchillai.ai;

import com.parknchillai.parking.SlotStatus;
import org.junit.jupiter.api.Test;

import java.io.IOException;

public class DatasetEvaluationTest {

    @Test
    void runEvaluationComparison() throws IOException {
        DatasetLoader.GroupedPartitions enhancedPartitions =
            DatasetLoader.loadPartitions("data/labels/features.csv", "data/labels/manifest.csv");

        // 1. Train DecisionTree on Enhanced Training Partition
        DecisionTreeModel enhancedTree = new DecisionTreeTrainer(4, 2).train(
            enhancedPartitions.train().features,
            enhancedPartitions.train().labels
        );
        DecisionTreePredictor enhancedPredictor = new DecisionTreePredictor(enhancedTree);

        ModelEvaluator.EvaluationReport enhancedTrainReport = ModelEvaluator.evaluate(
            enhancedPredictor,
            enhancedPartitions.train().features,
            enhancedPartitions.train().labels,
            ModelEvaluator.EvaluationScope.TRAINING_SET,
            "Enhanced (ivp-enhance-v1)"
        );

        ModelEvaluator.EvaluationReport enhancedValReport = ModelEvaluator.evaluate(
            enhancedPredictor,
            enhancedPartitions.validation().features,
            enhancedPartitions.validation().labels,
            ModelEvaluator.EvaluationScope.VALIDATION_PARTITION,
            "Enhanced (ivp-enhance-v1)"
        );

        ModelEvaluator.EvaluationReport enhancedTestReport = ModelEvaluator.evaluate(
            enhancedPredictor,
            enhancedPartitions.test().features,
            enhancedPartitions.test().labels,
            ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION,
            "Enhanced (ivp-enhance-v1)"
        );

        System.out.println("\n=======================================================");
        System.out.println("=== SYNTHETIC-DEMO BASELINE EVALUATION (v2.2-8feat) ===");
        System.out.println("=== [SYNTHETIC-DEMO / LEGACY-8-FEATURE, not evidence of enhanced-model accuracy] ===");
        System.out.println("=======================================================");
        System.out.println("--- TRAIN SET (N=" + enhancedPartitions.train().size() + ") ---");
        System.out.println(enhancedTrainReport);
        System.out.println("--- VALIDATION SET (N=" + enhancedPartitions.validation().size() + ") ---");
        System.out.println(enhancedValReport);
        System.out.println("--- HELD-OUT TEST SET (N=" + enhancedPartitions.test().size() + ") ---");
        System.out.println(enhancedTestReport);
        System.out.println("Tree Structure:\n" + enhancedTree.toTreeString());

        // 2. Evaluate Raw Features baseline
        // For raw baseline, we construct the raw demo values and synthetic samples
        double[][] rawTrainFeats = new double[enhancedPartitions.train().size()][];
        for (int i = 0; i < rawTrainFeats.length; i++) {
            rawTrainFeats[i] = enhancedPartitions.train().features[i].clone();
        }
        // Substitute the 4 raw demo lot values in train set
        // Slot 1
        rawTrainFeats[0] = new double[]{79.350125,0.483500,0.549000,0.473000,0.439000,117.253250,0.607750,0.374125};
        // Slot 6
        rawTrainFeats[1] = new double[]{82.093875,0.448125,0.548250,0.501625,0.405125,119.257625,0.609500,0.349125};
        // Slot 7
        rawTrainFeats[2] = new double[]{112.587625,0.458250,0.464500,0.519500,0.419500,131.718750,0.607875,0.397500};
        // Slot 12
        rawTrainFeats[3] = new double[]{77.776500,0.211875,0.720500,0.478625,0.167375,54.996875,0.172750,0.124750};
        // Slot 2 in train set
        rawTrainFeats[24] = new double[]{63.740625,0.067000,0.885750,0.200625,0.022500,17.547375,0.000000,0.022250};

        DecisionTreeModel rawTree = new DecisionTreeTrainer(4, 2).train(
            rawTrainFeats,
            enhancedPartitions.train().labels
        );
        DecisionTreePredictor rawPredictor = new DecisionTreePredictor(rawTree);

        ModelEvaluator.EvaluationReport rawTrainReport = ModelEvaluator.evaluate(
            rawPredictor,
            rawTrainFeats,
            enhancedPartitions.train().labels,
            ModelEvaluator.EvaluationScope.TRAINING_SET,
            "Raw Baseline (v2.3-10feat-raw-v1)"
        );

        ModelEvaluator.EvaluationReport rawValReport = ModelEvaluator.evaluate(
            rawPredictor,
            enhancedPartitions.validation().features,
            enhancedPartitions.validation().labels,
            ModelEvaluator.EvaluationScope.VALIDATION_PARTITION,
            "Raw Baseline (v2.3-10feat-raw-v1)"
        );

        ModelEvaluator.EvaluationReport rawTestReport = ModelEvaluator.evaluate(
            rawPredictor,
            enhancedPartitions.test().features,
            enhancedPartitions.test().labels,
            ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION,
            "Raw Baseline (v2.3-10feat-raw-v1)"
        );

        System.out.println("\n=======================================================");
        System.out.println("=== RAW BASELINE MODEL EVALUATION (v2.3-10feat-raw-v1) ===");
        System.out.println("=======================================================");
        System.out.println("--- TRAIN SET (N=" + rawTrainFeats.length + ") ---");
        System.out.println(rawTrainReport);
        System.out.println("--- VALIDATION SET (N=" + enhancedPartitions.validation().size() + ") ---");
        System.out.println(rawValReport);
        System.out.println("--- HELD-OUT TEST SET (N=" + enhancedPartitions.test().size() + ") ---");
        System.out.println(rawTestReport);
        System.out.println("Tree Structure:\n" + rawTree.toTreeString());
    }
}
