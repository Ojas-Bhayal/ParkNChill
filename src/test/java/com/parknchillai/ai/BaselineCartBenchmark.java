package com.parknchillai.ai;

import com.parknchillai.parking.SlotStatus;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.util.*;

public class BaselineCartBenchmark {

    public static final long BENCHMARK_SEED = 20261023L;

    public static void main(String[] args) {
        System.out.println("=== BASELINE CART BENCHMARK (LegacyDecisionTreeTrainer) ===");
        System.out.println("JVM: " + System.getProperty("java.version") + " (" + System.getProperty("java.vendor") + ")");
        System.out.println("OS: " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
        System.out.println("Hardware Cores: " + Runtime.getRuntime().availableProcessors());
        System.out.println("Max Heap: " + (Runtime.getRuntime().maxMemory() / (1024 * 1024)) + " MB");
        System.out.println();

        System.out.println("N,Warmup,Measured,MinMs,MedianMs,P95Ms,MaxMs,HeapBeforeMB,HeapAfterMB,HeapDeltaMB");
        runBenchmark(1000, 3, 5);
        runBenchmark(5000, 1, 2);
        runBenchmark(13000, 0, 1);
    }

    public static void runBenchmark(int n, int warmup, int measured) {
        Random rng = new Random(BENCHMARK_SEED);
        double[][] features = generateSyntheticFeatures(rng, n, 10);
        SlotStatus[] labels = generateSyntheticLabels(rng, n);

        LegacyDecisionTreeTrainer trainer = new LegacyDecisionTreeTrainer(5, 5);

        for (int i = 0; i < warmup; i++) {
            trainer.train(features, labels);
        }

        forcedGc();
        MemoryMXBean mem = ManagementFactory.getMemoryMXBean();
        long heapBefore = mem.getHeapMemoryUsage().getUsed();

        double[] times = new double[measured];
        for (int i = 0; i < measured; i++) {
            long t0 = System.nanoTime();
            trainer.train(features, labels);
            long t1 = System.nanoTime();
            times[i] = (t1 - t0) / 1_000_000.0;
        }

        forcedGc();
        long heapAfter = mem.getHeapMemoryUsage().getUsed();

        Arrays.sort(times);
        double min = times[0];
        double max = times[times.length - 1];
        double median = times[times.length / 2];
        int p95Idx = (int) Math.ceil(0.95 * times.length) - 1;
        double p95 = times[Math.min(p95Idx, times.length - 1)];

        double heapBeforeMB = heapBefore / (1024.0 * 1024.0);
        double heapAfterMB = heapAfter / (1024.0 * 1024.0);
        double heapDeltaMB = heapAfterMB - heapBeforeMB;

        System.out.printf(Locale.ROOT, "%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f%n",
                n, warmup, measured, min, median, p95, max, heapBeforeMB, heapAfterMB, heapDeltaMB);
    }

    public static double[][] generateSyntheticFeatures(Random rng, int n, int featureCount) {
        double[][] f = new double[n][featureCount];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < featureCount; j++) {
                f[i][j] = rng.nextDouble() * 100.0;
            }
        }
        return f;
    }

    public static SlotStatus[] generateSyntheticLabels(Random rng, int n) {
        SlotStatus[] l = new SlotStatus[n];
        for (int i = 0; i < n; i++) {
            l[i] = rng.nextBoolean() ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
        }
        return l;
    }

    private static void forcedGc() {
        for (int i = 0; i < 3; i++) {
            System.gc();
            try {
                Thread.sleep(50);
            } catch (InterruptedException ignored) {}
        }
    }
}
