package com.parknchillai.ai;

import com.parknchillai.parking.SlotStatus;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public class CartBenchmarkRunner {

    private static final int WARMUP_RUNS = 5;
    private static final int MEASURED_RUNS = 10;
    private static final int[] DATASET_SIZES = {1000, 5000, 13000};
    private static final int[] MAX_DEPTHS = {3, 5, 8};
    private static final int[] MIN_SAMPLES = {2, 5, 10};

    record BenchmarkResult(
            String trainerName,
            int datasetSize,
            double minMs,
            double medianMs,
            double p95Ms,
            double maxMs,
            double meanGridSearchMs
    ) {}

    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println("PARKNCHILL v2.3 CART BENCHMARK: Optimized vs Legacy Trainer");
        System.out.println("Warmup: " + WARMUP_RUNS + " runs | Measured: " + MEASURED_RUNS + " runs");
        System.out.println("Grid Search: 9 candidate hyperparameter pairs (depth: 3,5,8 x minSplit: 2,5,10)");
        System.out.println("================================================================================");

        List<BenchmarkResult> results = new ArrayList<>();

        for (int size : DATASET_SIZES) {
            System.gc();
            Random rng = new Random(42);
            double[][] features = new double[size][10];
            SlotStatus[] labels = new SlotStatus[size];
            for (int i = 0; i < size; i++) {
                for (int f = 0; f < 10; f++) {
                    features[i][f] = rng.nextDouble();
                }
                boolean occ = (features[i][8] * 0.4 + features[i][3] * 0.3 + features[i][1] * 0.3) > 0.5;
                labels[i] = occ ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
            }

            // Benchmark Legacy
            BenchmarkResult legacyRes = runBenchmarkForTrainer("Legacy (Baseline)", features, labels, true);
            results.add(legacyRes);

            // Benchmark Optimized
            BenchmarkResult optRes = runBenchmarkForTrainer("Optimized (WS-1)", features, labels, false);
            results.add(optRes);
        }

        // Output markdown table
        System.out.println("\n### CART Training Benchmark Results\n");
        System.out.println("| Dataset Size (N) | Trainer | Min (ms) | Median (ms) | P95 (ms) | Max (ms) | 9-Model Grid Search Mean (ms) | Speedup (Median) |");
        System.out.println("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|");

        for (int i = 0; i < results.size(); i += 2) {
            BenchmarkResult leg = results.get(i);
            BenchmarkResult opt = results.get(i + 1);

            double speedupMedian = leg.medianMs / Math.max(0.001, opt.medianMs);
            double speedupGrid = leg.meanGridSearchMs / Math.max(0.001, opt.meanGridSearchMs);

            System.out.printf(Locale.ROOT, "| %d | %s | %.2f | %.2f | %.2f | %.2f | %.2f | - |\n",
                    leg.datasetSize, leg.trainerName, leg.minMs, leg.medianMs, leg.p95Ms, leg.maxMs, leg.meanGridSearchMs);
            System.out.printf(Locale.ROOT, "| %d | %s | %.2f | %.2f | %.2f | %.2f | %.2f | **%.2fx** (Grid: **%.2fx**) |\n",
                    opt.datasetSize, opt.trainerName, opt.minMs, opt.medianMs, opt.p95Ms, opt.maxMs, opt.meanGridSearchMs, speedupMedian, speedupGrid);
        }
        System.out.println();
    }

    private static BenchmarkResult runBenchmarkForTrainer(String name, double[][] features, SlotStatus[] labels, boolean isLegacy) {
        System.out.printf(Locale.ROOT, "Benchmarking %s on N=%d...\n", name, features.length);

        // Warmup single run (depth=5, minSplit=5)
        for (int w = 0; w < WARMUP_RUNS; w++) {
            if (isLegacy) {
                new LegacyDecisionTreeTrainer(5, 5).train(features, labels);
            } else {
                new DecisionTreeTrainer(5, 5).train(features, labels);
            }
        }

        // Measure single model (depth=5, minSplit=5)
        double[] singleTimes = new double[MEASURED_RUNS];
        for (int m = 0; m < MEASURED_RUNS; m++) {
            long start = System.nanoTime();
            if (isLegacy) {
                new LegacyDecisionTreeTrainer(5, 5).train(features, labels);
            } else {
                new DecisionTreeTrainer(5, 5).train(features, labels);
            }
            long end = System.nanoTime();
            singleTimes[m] = (end - start) / 1_000_000.0;
        }

        Arrays.sort(singleTimes);
        double min = singleTimes[0];
        double median = singleTimes[MEASURED_RUNS / 2];
        double p95 = singleTimes[(int) Math.ceil(0.95 * MEASURED_RUNS) - 1];
        double max = singleTimes[MEASURED_RUNS - 1];

        // Measure full 9-candidate grid search
        double gridTotal = 0;
        int gridIterations = 3;
        for (int g = 0; g < gridIterations; g++) {
            long gStart = System.nanoTime();
            for (int d : MAX_DEPTHS) {
                for (int s : MIN_SAMPLES) {
                    if (isLegacy) {
                        new LegacyDecisionTreeTrainer(d, s).train(features, labels);
                    } else {
                        new DecisionTreeTrainer(d, s).train(features, labels);
                    }
                }
            }
            long gEnd = System.nanoTime();
            gridTotal += (gEnd - gStart) / 1_000_000.0;
        }
        double meanGrid = gridTotal / gridIterations;

        return new BenchmarkResult(name, features.length, min, median, p95, max, meanGrid);
    }
}
