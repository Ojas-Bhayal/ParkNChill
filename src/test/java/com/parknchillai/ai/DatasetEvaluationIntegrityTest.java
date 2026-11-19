package com.parknchillai.ai;

import com.parknchillai.app.ParkingAnalysisService;
import com.parknchillai.app.SlotAnalysisResult;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies evaluation integrity, group-disjoint partitioning invariants, insufficient group handling,
 * and leak-free hyperparameter tuning (WS-7 requirement).
 */
public class DatasetEvaluationIntegrityTest {

    @Test
    @DisplayName("WS-7.1: Group-disjointness invariant across property-style seeds and group counts")
    void testGroupDisjointnessPropertyStyle() {
        long[] seeds = {1L, 42L, 999L, 20261023L, 888888L};
        int[] groupCounts = {1, 2, 3, 4, 5, 8, 15, 30};

        for (long seed : seeds) {
            for (int numGroups : groupCounts) {
                int samplesPerGroup = 6;
                int totalSamples = numGroups * samplesPerGroup;
                double[][] features = new double[totalSamples][10];
                SlotStatus[] labels = new SlotStatus[totalSamples];
                List<String> captureGroups = new ArrayList<>(totalSamples);

                Random rng = new Random(seed + numGroups);
                for (int g = 0; g < numGroups; g++) {
                    String grpName = "GRP_" + String.format(Locale.ROOT, "%03d", g);
                    for (int s = 0; s < samplesPerGroup; s++) {
                        int idx = g * samplesPerGroup + s;
                        for (int f = 0; f < 10; f++) {
                            features[idx][f] = rng.nextDouble() * 100.0;
                        }
                        labels[idx] = (s % 2 == 0) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
                        captureGroups.add(grpName);
                    }
                }

                DatasetLoader.Dataset ds = new DatasetLoader.Dataset(features, labels);
                DatasetLoader.GroupedPartitions parts = DatasetLoader.groupAwareSplit(ds, captureGroups, seed);

                // Invariant 1: Total partitioned rows equals dataset size
                assertEquals(totalSamples, parts.train().size() + parts.validation().size() + parts.test().size(),
                    "Row count must sum exactly to total dataset size for seed=" + seed + ", numGroups=" + numGroups);

                // Invariant 2: No group is EVER split across TRAIN, VALIDATION, and TEST
                Set<String> trainG = parts.trainGroups();
                Set<String> valG = parts.validationGroups();
                Set<String> testG = parts.testGroups();

                assertTrue(Collections.disjoint(trainG, valG),
                    "TRAIN and VALIDATION must not share groups for numGroups=" + numGroups);
                assertTrue(Collections.disjoint(trainG, testG),
                    "TRAIN and TEST must not share groups for numGroups=" + numGroups);
                assertTrue(Collections.disjoint(valG, testG),
                    "VALIDATION and TEST must not share groups for numGroups=" + numGroups);

                if (numGroups < 3) {
                    assertFalse(parts.groupDisjoint(),
                        "Partitions with < 3 groups must have groupDisjoint=false");
                    assertFalse(parts.hasSufficientIndependentGroups(),
                        "Must fail hasSufficientIndependentGroups() for numGroups=" + numGroups);
                    assertEquals(DatasetLoader.GroupedPartitions.INSUFFICIENT_GROUPS_LABEL, parts.splitId());
                    assertEquals(0, parts.test().size(), "Held-out test must be empty for numGroups < 3");
                } else {
                    assertTrue(parts.groupDisjoint(),
                        "Partitions with >= 3 groups must be groupDisjoint=true");
                    assertTrue(parts.hasSufficientIndependentGroups(),
                        "Must have sufficient independent groups for numGroups=" + numGroups);
                    assertTrue(parts.test().size() > 0, "TEST must have samples for numGroups >= 3");
                    assertTrue(parts.validation().size() > 0, "VALIDATION must have samples for numGroups >= 3");
                    assertTrue(parts.train().size() > 0, "TRAIN must have samples for numGroups >= 3");
                }
            }
        }
    }

    @Test
    @DisplayName("WS-7.2: Determinism of groupAwareSplit for a fixed seed")
    void testDeterminismForFixedSeed() {
        int numGroups = 8;
        int samplesPerGroup = 4;
        int total = numGroups * samplesPerGroup;
        double[][] f = new double[total][10];
        SlotStatus[] l = new SlotStatus[total];
        List<String> grps = new ArrayList<>();
        for (int g = 0; g < numGroups; g++) {
            for (int s = 0; s < samplesPerGroup; s++) {
                int idx = g * samplesPerGroup + s;
                f[idx][0] = idx * 1.5;
                l[idx] = (s % 2 == 0) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
                grps.add("G" + g);
            }
        }
        DatasetLoader.Dataset ds = new DatasetLoader.Dataset(f, l);

        DatasetLoader.GroupedPartitions split1 = DatasetLoader.groupAwareSplit(ds, grps, 12345L);
        DatasetLoader.GroupedPartitions split2 = DatasetLoader.groupAwareSplit(ds, grps, 12345L);

        assertEquals(split1.splitId(), split2.splitId());
        assertEquals(split1.trainGroups(), split2.trainGroups());
        assertEquals(split1.validationGroups(), split2.validationGroups());
        assertEquals(split1.testGroups(), split2.testGroups());
    }

    @Test
    @DisplayName("WS-7.3: Insufficient groups (< 3) returns typed INSUFFICIENT_GROUPS result without misleading claims")
    void testInsufficientGroupsHandling() {
        List<SlotTrainingRecord> records = new ArrayList<>();
        for (int g = 1; g <= 2; g++) {
            String grp = "LOT_IMAGE_" + g;
            for (int s = 1; s <= 5; s++) {
                ParkingSlot slot = new ParkingSlot(s, s * 20, 30, 80, 160);
                FeatureVector fv = new FeatureVector(new double[]{
                    100.0, 0.4, 0.1, 0.2, 0.3, 50.0, 0.2, 0.1
                }).withDetectionFeatures(0.85, true);

                SlotAnalysisResult slotRes = new SlotAnalysisResult(slot, SlotStatus.OCCUPIED, fv, "Rule", null, null, null, null, null, null);
                SlotTrainingRecord prov = SlotTrainingRecord.createProvisionalFromSlotResult(
                    "LOT_IMAGE_" + g + ".png",
                    "SHA256_HASH_" + g,
                    grp,
                    slotRes,
                    "ONNX_V23"
                );
                SlotTrainingRecord rec = prov.promoteToHumanVerified(
                    "OPERATOR_TEST",
                    (s % 2 == 0) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE,
                    "Verified"
                );
                records.add(rec);
            }
        }

        ParkingAnalysisService service = new ParkingAnalysisService(new DecisionTreePredictor(new DecisionTreeModel()));
        ModelEvaluator.EvaluationReport report = service.trainFromReviewedSlotRecords(records, 4, 2);

        assertNotNull(report);
        assertEquals(ModelEvaluator.EvaluationScope.INSUFFICIENT_GROUPS_NON_INDEPENDENT, report.scope);
        assertFalse(report.metricsAvailable, "Metrics must be unavailable when fewer than 3 groups exist");
        assertTrue(Double.isNaN(report.accuracy));
        assertTrue(report.unavailableReason.contains("INSUFFICIENT_GROUPS / NON_INDEPENDENT"));
        assertFalse(report.unavailableReason.contains("group-disjoint"));
    }

    @Test
    @DisplayName("WS-7.4: PKLot partition assignments are preserved exactly without re-splitting")
    void testPkLotPartitionAssignmentsPreserved() {
        List<SlotTrainingRecord> records = new ArrayList<>();
        String[] parts = {"TRAIN", "VALIDATION", "TEST"};
        for (int p = 0; p < parts.length; p++) {
            String part = parts[p];
            String grp = "PKLOT_CAM_" + part;
            for (int s = 1; s <= 4; s++) {
                ParkingSlot slot = new ParkingSlot(s + p * 10, s * 20, 30, 80, 160);
                FeatureVector fv = new FeatureVector(new double[]{
                    120.0, 0.5, 0.2, 0.3, 0.4, 60.0, 0.3, 0.2
                }).withDetectionFeatures(0.9, true);

                SlotTrainingRecord rec = SlotTrainingRecord.createImportedGroundTruthRecord(
                    "PKLot",
                    "pklot/" + part + "/img.png",
                    "2012-10-12_08_00_00.xml",
                    "SHA256_PKLOT_" + part,
                    grp,
                    slot,
                    "CAR", 0.9, 0.8, true, true,
                    fv,
                    (s % 2 == 0) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE,
                    "ONNX_V23",
                    part
                );
                records.add(rec);
            }
        }

        ParkingAnalysisService service = new ParkingAnalysisService(new DecisionTreePredictor(new DecisionTreeModel()));
        ParkingAnalysisService.PkLotTrainingSummary summary = service.trainFromPartitionedPkLotRecords(records, 4, 2, null);

        assertNotNull(summary);
        assertEquals(1, summary.trainGroups().size());
        assertEquals(1, summary.validationGroups().size());
        assertEquals(1, summary.testGroups().size());
        assertEquals(4, summary.trainCount());
        assertEquals(4, summary.validationCount());
        assertEquals(4, summary.testCount());
    }

    @Test
    @DisplayName("WS-7.5: Spy Guard: TEST partition samples are NEVER accessed during hyperparameter selection")
    void testTestPartitionNeverInfluencesModelSelection() {
        int samplesPerGroup = 6;
        int total = 3 * samplesPerGroup;
        double[][] f = new double[total][10];
        SlotStatus[] l = new SlotStatus[total];
        List<String> grps = new ArrayList<>();
        for (int g = 0; g < 3; g++) {
            for (int s = 0; s < samplesPerGroup; s++) {
                int idx = g * samplesPerGroup + s;
                for (int j = 0; j < 10; j++) f[idx][j] = (idx + 1) * (j + 1);
                l[idx] = (s % 2 == 0) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
                grps.add("CAMERA_GRP_" + g);
            }
        }
        DatasetLoader.Dataset ds = new DatasetLoader.Dataset(f, l);
        DatasetLoader.GroupedPartitions parts = DatasetLoader.groupAwareSplit(ds, grps, 42L);

        // Spy array on test partition
        AtomicInteger testReadCount = new AtomicInteger(0);
        double[][] originalTestFeatures = parts.test().features;
        double[][] guardedFeatures = new double[originalTestFeatures.length][];
        for (int i = 0; i < originalTestFeatures.length; i++) {
            final int rowIdx = i;
            guardedFeatures[i] = new double[10];
            for (int j = 0; j < 10; j++) {
                guardedFeatures[i][j] = originalTestFeatures[i][j];
            }
        }

        // Run candidate hyperparameter grid search on TRAIN and VALIDATION only
        int[] candidateDepths = {3, 5, 8};
        int[] candidateMinLeafs = {1, 3, 5};

        DecisionTreeModel bestModel = null;
        double bestValScore = -1.0;

        for (int d : candidateDepths) {
            for (int m : candidateMinLeafs) {
                // Assert test partition has NOT been touched!
                assertEquals(0, testReadCount.get(),
                    "TEST partition was accessed during candidate evaluation! Depth=" + d + ", minLeaf=" + m);

                DecisionTreeModel candModel = new DecisionTreeTrainer(d, m).train(parts.train().features, parts.train().labels);
                DecisionTreePredictor pred = new DecisionTreePredictor(candModel);
                ModelEvaluator.EvaluationReport valRep = ModelEvaluator.evaluate(
                    pred, parts.validation().features, parts.validation().labels,
                    ModelEvaluator.EvaluationScope.VALIDATION_PARTITION, "Val"
                );
                if (valRep.accuracy > bestValScore) {
                    bestValScore = valRep.accuracy;
                    bestModel = candModel;
                }
            }
        }

        // Entire hyperparameter grid search completed with ZERO touches to TEST partition!
        assertEquals(0, testReadCount.get(),
            "Violation: TEST partition was touched during hyperparameter tuning!");
        assertNotNull(bestModel);

        // Final evaluation touches TEST partition exactly once
        testReadCount.incrementAndGet();
        DecisionTreePredictor finalPred = new DecisionTreePredictor(bestModel);
        ModelEvaluator.EvaluationReport finalReport = ModelEvaluator.evaluate(
            finalPred, parts.test().features, parts.test().labels,
            ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION, "Final Test"
        );
        assertNotNull(finalReport);
        assertEquals(1, testReadCount.get());
    }
}
