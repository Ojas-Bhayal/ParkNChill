package com.parknchillai.app;

import com.parknchillai.ai.ClassifierBenchmark;
import com.parknchillai.ai.DatasetLoader;
import com.parknchillai.ai.DatasetValidator;
import com.parknchillai.ai.ModelEvaluator;
import com.parknchillai.ai.UncertaintyPolicy;
import com.parknchillai.detection.OccupancyDecisionEngine;
import com.parknchillai.detection.OnnxVehicleDetectionService;
import com.parknchillai.detection.VehicleDetectionService;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.layout.LayoutDetectionResult;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class ParkingAnalysisServiceTest {

    private ParkingAnalysisService service;

    @BeforeEach
    void setUp() {
        service = ParkingAnalysisService.createWithDefaultModel();
        // Verify Requirement 1: Vehicle detection is the default live analysis mode
        assertEquals(OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION, service.getAnalysisMode());
        // Select FEATURE_DECISION_TREE for the CART-specific unit tests in this suite
        service.setAnalysisMode(OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE);
    }

    @Test
    void defaultLiveModeIsVehicleDetectionAndNeverSilentlyFallsBackToDecisionTreeWhenModelMissing() throws Exception {
        ParkingAnalysisService freshSvc = ParkingAnalysisService.createWithDefaultModel();
        Path absentModel = Files.createTempDirectory("parknchill-missing-model").resolve("definitely-missing.onnx");
        freshSvc.setVehicleDetectionService(new OnnxVehicleDetectionService(absentModel));
        assertEquals(OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION, freshSvc.getAnalysisMode(),
            "Default live analysis mode must be VEHICLE_DETECTION");

        BufferedImage img = new BufferedImage(120, 120, BufferedImage.TYPE_INT_RGB);
        ParkingSlot slot = new ParkingSlot(1, 10, 10, 80, 100);
        AnalysisSummary summary = freshSvc.analyze(img, List.of(slot));
        assertEquals(OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION, summary.getAnalysisMode());
        assertEquals(VehicleDetectionService.DetectorStatus.MODEL_MISSING, summary.getDetectionBatchResult().status());
        assertEquals(SlotStatus.NEEDS_REVIEW, summary.getSlotResults().get(0).getStatus(),
            "Missing ONNX model in default VEHICLE_DETECTION mode must route to NEEDS_REVIEW and never fall back to Decision Tree");
    }

    @Test
    void serviceInitializedWithGroupDisjointPartitionsAndHonestProvenance() {
        assertNotNull(service.getPredictor());
        assertNotNull(service.getPredictor().getModel());
        String treeStr = service.getPredictor().getModel().toTreeString();
        assertFalse(treeStr.isEmpty());
        assertFalse(treeStr.contains("50% purity"), "Default trained model must have 100% pure leaves; got:\n" + treeStr);
        assertEquals(FeatureVector.FEATURE_COUNT, service.getPredictor().getModel().getFeatureCount());

        assertTrue(service.getLastTrainSize() > 0, "Train split size must be positive");
        assertTrue(service.getLastValidationSize() > 0, "Validation split size must be positive");
        assertTrue(service.getLastTestSize() > 0, "Test split size must be positive");
        assertTrue(service.getLastGroupedPartitions().groupDisjoint(),
            "Default manifest partitions must be strictly group-disjoint across Train, Validation, and Test");

        DatasetValidator.ValidationSummary vs = service.getLastValidationSummary();
        assertNotNull(vs);
        assertTrue(vs.validForTraining());
        assertFalse(vs.realImageSamplesPresent());
        assertTrue(vs.groupDisjointEvaluationPossible());
        assertFalse(vs.realWorldGeneralizationEstablished(),
            "Bundled dataset must honestly report that real-world camera generalization is NOT ESTABLISHED");

        ModelEvaluator.EvaluationReport testReport = service.evaluateModel(ParkingAnalysisService.DEFAULT_DATASET_PATH);
        assertEquals(ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION, testReport.scope);
        assertEquals(service.getLastTestSize(), testReport.totalEvaluated());
        assertEquals(1.0, testReport.accuracy, 1e-6, "Group-disjoint held-out test accuracy should be 1.0000");
        assertEquals(0, testReport.fn, "False-available (FN) count must be 0");
        assertEquals(0.0, testReport.falseAvailableRate(), 1e-6);

        ModelEvaluator.EvaluationReport valReport = service.evaluateValidationPartition();
        assertEquals(ModelEvaluator.EvaluationScope.VALIDATION_PARTITION, valReport.scope);
        assertEquals(1.0, valReport.accuracy, 1e-6);

        DatasetLoader.Dataset fullDs = DatasetLoader.load(ParkingAnalysisService.DEFAULT_DATASET_PATH);
        ModelEvaluator.EvaluationReport fullReport = ModelEvaluator.evaluate(service.getPredictor(), fullDs.features, fullDs.labels);
        assertEquals(1.0, fullReport.accuracy, 1e-6, "Full dataset accuracy should be 1.0000 across all 48 samples");
    }

    @Test
    void classifierBenchmarkComparesCartAndRandomForestUnderValidationDrivenProtocol() {
        ClassifierBenchmark.BenchmarkResult bench = service.runClassifierBenchmark();
        assertNotNull(bench);
        assertEquals(ClassifierBenchmark.SelectedModel.CART_DECISION_TREE, bench.selectedModel());
        assertEquals(1.0, bench.cartValidationReport().accuracy, 1e-6);
        assertEquals(1.0, bench.rfValidationReport().accuracy, 1e-6);
        assertNotNull(bench.selectedModelTestReport());
        assertEquals(1.0, bench.selectedModelTestReport().accuracy, 1e-6);
        assertNotNull(bench.cartTestReport());
        assertNull(bench.rfTestReport(), "Default flow evaluates held-out test set only for the validation-selected model");
        assertTrue(bench.formatSummaryTable().contains("CART Decision Tree"));
        assertTrue(bench.formatSummaryTable().contains("Random Forest (15 Trees)"));
    }

    @Test
    void analyzeProducesValidSummaryAndPerSlotResultsWithConsistentCounts() {
        BufferedImage img = new BufferedImage(200, 150, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(new Color(50, 50, 50));
        g.fillRect(0, 0, 200, 150);

        g.setColor(new Color(200, 200, 200));
        g.drawRect(10, 10, 80, 100);

        for (int y = 20; y < 100; y += 4) {
            for (int x = 110; x < 180; x += 4) {
                g.setColor((x + y) % 8 == 0 ? Color.WHITE : Color.RED);
                g.fillRect(x, y, 4, 4);
            }
        }
        g.dispose();

        ParkingSlot slot1 = new ParkingSlot(1, 10, 10, 80, 100);
        ParkingSlot slot2 = new ParkingSlot(2, 105, 10, 80, 100);

        AnalysisSummary summary = service.analyze(img, List.of(slot1, slot2));

        assertNotNull(summary);
        assertEquals(2, summary.getTotalSlots());
        assertEquals(2, summary.getSlotResults().size());
        assertEquals(summary.getTotalSlots(),
            summary.getOccupiedSlots() + summary.getAvailableSlots() + summary.getNeedsReviewSlots());
        assertEquals(SlotStatus.AVAILABLE, summary.getSlotResults().get(0).getStatus());
        assertEquals(SlotStatus.OCCUPIED, summary.getSlotResults().get(1).getStatus());
        assertEquals(1.0, summary.getSlotResults().get(0).getLeafPurity(), 1e-6);
        assertNotNull(summary.getSlotResults().get(0).getDecisionTrace());
        assertNotNull(summary.getSlotResults().get(0).getSaturationImage());
        assertNotNull(summary.getSlotResults().get(0).getLargestBlobMask());
    }

    @Test
    void outOfDomainPredictionRoutesToNeedsReviewInsteadOfSilentlyBecomingAvailable() {
        service.getPredictor().setUncertaintyPolicy(UncertaintyPolicy.strictDomainPolicy());
        FeatureVector extremeOod = new FeatureVector(new double[]{
            254.0, 0.01, 0.0, 0.01, 0.01, 250.0, 0.99, 0.01
        });
        assertEquals(SlotStatus.NEEDS_REVIEW, service.getPredictor().predictWithUncertainty(extremeOod),
            "Out-of-domain feature vector must be routed to NEEDS_REVIEW rather than silently becoming AVAILABLE");
    }

    @Test
    void uniformEmptyAsphaltClassifiedAsAvailable() {
        BufferedImage uniformLot = new BufferedImage(100, 120, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = uniformLot.createGraphics();
        g.setColor(new Color(55, 58, 62));
        g.fillRect(0, 0, 100, 120);
        g.dispose();

        ParkingSlot slot = new ParkingSlot(1, 10, 10, 80, 100);
        AnalysisSummary summary = service.analyze(uniformLot, List.of(slot));
        assertEquals(SlotStatus.AVAILABLE, summary.getSlotResults().get(0).getStatus(),
            "Uniform empty dark asphalt must be classified as AVAILABLE");
    }

    @Test
    void darkMatteBlackVehicleClassifiedAsOccupiedInBothOtsuAndAdaptiveModes() {
        BufferedImage lot = new BufferedImage(100, 120, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = lot.createGraphics();
        g.setColor(new Color(60, 62, 66));
        g.fillRect(0, 0, 100, 120);
        g.setColor(new Color(25, 27, 30));
        g.fillRoundRect(22, 22, 56, 76, 12, 12);
        g.setColor(new Color(15, 18, 22));
        g.fillRect(28, 34, 44, 15);
        g.dispose();

        ParkingSlot slotOtsu = new ParkingSlot(1, 10, 10, 80, 100);
        AnalysisSummary otsuSummary = service.analyze(lot, List.of(slotOtsu), false);
        assertEquals(SlotStatus.OCCUPIED, otsuSummary.getSlotResults().get(0).getStatus(),
            "Matte-black vehicle must be classified as OCCUPIED in Otsu mode");

        ParkingSlot slotAdaptive = new ParkingSlot(1, 10, 10, 80, 100);
        AnalysisSummary adaptiveSummary = service.analyze(lot, List.of(slotAdaptive), true);
        assertEquals(SlotStatus.OCCUPIED, adaptiveSummary.getSlotResults().get(0).getStatus(),
            "Matte-black vehicle must be classified as OCCUPIED in two-sided Adaptive mode");
    }

    @Test
    void speckledHighForegroundEmptyPavementClassifiedAsAvailable() {
        FeatureVector speckledEmpty = new FeatureVector(new double[]{
            68.0, 0.2800, 0.8500, 0.1400, 0.2750, 10.0, 0.0000, 0.0200
        });
        assertEquals(SlotStatus.AVAILABLE, service.getPredictor().predict(speckledEmpty),
            "Weathered/speckled empty stall with high foregroundRatio (0.28) but tiny largestBlobRatio (0.02) must be AVAILABLE");
    }

    @Test
    void csvRowExportFormattingProducesTwelveRfc4180Columns() {
        ParkingSlot slot = new ParkingSlot(7, 10, 20, 80, 100);
        FeatureVector fv = new FeatureVector(new double[]{
            79.35, 0.4835, 0.5490, 0.4730, 0.4390, 117.25, 0.6078, 0.3741
        });
        SlotAnalysisResult res = new SlotAnalysisResult(
            slot, SlotStatus.OCCUPIED, 1.0, fv,
            "largestBlobRatio (0.3741) > 0.1436 -> \"OCCUPIED\"\nLeaf",
            null, null, null, null, null, null, null, null
        );
        String csv = res.toCsvRow().trim();
        assertTrue(csv.startsWith("7,OCCUPIED,1.0000,79.3500,0.4835,0.5490,0.4730,0.4390,117.2500,0.6078,0.3741,"));
        assertTrue(csv.endsWith("\"largestBlobRatio (0.3741) > 0.1436 -> \"\"OCCUPIED\"\" Leaf\""),
            "Quotes and newlines in decision rule must be RFC 4180 escaped; got: " + csv);
    }

    @Test
    void inspectDemoLotFeaturesVerifiesAllTwelveSlotsAndZeroConfigAutoDetection() throws Exception {
        java.io.File imgFile = new java.io.File("data/sample/sample_parking_lot.png");
        if (!imgFile.exists()) {
            com.parknchillai.utils.SampleDataGenerator.ensureSampleImageExists("data/sample/sample_parking_lot.png");
        }
        java.awt.image.BufferedImage img = javax.imageio.ImageIO.read(imgFile);

        java.util.List<ParkingSlot> slots = com.parknchillai.utils.ConfigLoader.loadSlotsFromCsv("data/config/slots.csv");
        AnalysisSummary summary = service.analyze(img, slots, false);
        assertEquals(12, summary.getTotalSlots());
        assertEquals(4, summary.getOccupiedSlots(), "Demo lot must have exactly 4 OCCUPIED slots (#1, #6, #7, #12)");
        assertEquals(8, summary.getAvailableSlots(), "Demo lot must have exactly 8 AVAILABLE slots");
        assertEquals(0, summary.getNeedsReviewSlots());
        assertEquals(33.3333, summary.getOccupancyRate(), 0.01);

        Set<Integer> expectedOccupied = Set.of(1, 6, 7, 12);
        for (SlotAnalysisResult r : summary.getSlotResults()) {
            SlotStatus expected = expectedOccupied.contains(r.getId())
                ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
            assertEquals(expected, r.getStatus(),
                "Slot #" + r.getId() + " expected " + expected + " but got " + r.getStatus());
        }

        LayoutDetectionResult autoLayout = service.autoDetectLayout(img);
        assertEquals(12, autoLayout.getDetectedSlotCount());
        AnalysisSummary autoSummary = service.analyze(img, autoLayout.getSlots(), false);
        assertEquals(4, autoSummary.getOccupiedSlots());
        assertEquals(8, autoSummary.getAvailableSlots());
    }

    @Test
    void activeLearningAppendSampleAddsToTrainPoolPreservesFrozenTestSnapshotAndBlocksTestGroups() throws Exception {
        Path tempDataset = Files.createTempFile("active_learning_test", ".csv");
        Path tempManifest = Path.of(tempDataset.toString() + ".manifest.csv");
        Files.copy(Path.of(ParkingAnalysisService.DEFAULT_DATASET_PATH), tempDataset, StandardCopyOption.REPLACE_EXISTING);
        Files.copy(Path.of(ParkingAnalysisService.DEFAULT_MANIFEST_PATH), tempManifest, StandardCopyOption.REPLACE_EXISTING);

        ParkingAnalysisService customService = ParkingAnalysisService.createWithDataset(tempDataset.toString(), 4, 2);
        int trainBefore = customService.getLastTrainSize();
        int testBefore  = customService.getLastTestSize();
        double[][] testFeaturesBefore = customService.getLastTestDataset().features;
        ParkingAnalysisService.TestSnapshot snapBefore = customService.getFrozenTestSnapshot();
        assertNotNull(snapBefore);

        FeatureVector newSample = new FeatureVector(new double[]{85.0, 0.45, 0.50, 0.49, 0.41, 95.0, 0.44, 0.39});
        customService.appendLabeledSampleAndRetrain(
            "custom_lot.png", "LOT_USER_1", new ParkingSlot(1, 10, 10, 80, 100),
            newSample, SlotStatus.OCCUPIED, null, false);
        customService.appendLabeledSampleAndRetrain(
            "custom_lot.png", "LOT_USER_1", new ParkingSlot(1, 10, 10, 80, 100),
            newSample, SlotStatus.OCCUPIED, null, false); // second identical click updates in place

        assertEquals(trainBefore + 1, customService.getLastTrainSize(),
            "New active-learning sample must be added to the TRAIN partition");
        assertEquals(testBefore, customService.getLastTestSize(),
            "Held-out TEST partition size must remain strictly unchanged during active learning");
        assertSame(testFeaturesBefore, customService.getLastTestDataset().features,
            "Held-out TEST partition instance must be preserved during active learning");
        assertEquals(snapBefore.snapshotSha256(), customService.getFrozenTestSnapshot().snapshotSha256(),
            "Frozen TEST snapshot SHA-256 hash must remain invariant across active-learning retraining");
        assertEquals(snapBefore.testGroups(), customService.getFrozenTestSnapshot().testGroups());

        // Attempting to label with a blank reviewer ID must be rejected (PNC-REM-04)
        assertThrows(IllegalArgumentException.class, () ->
            customService.appendLabeledSampleAndRetrain(
                "custom_lot.png", "LOT_USER_1", new ParkingSlot(2, 10, 10, 80, 100),
                newSample, SlotStatus.OCCUPIED, "   ", null, false)
        );

        // Attempting to label a slot from an existing VALIDATION or TEST capture group must be refused (PNC-REM-06)
        assertThrows(IllegalStateException.class, () ->
            customService.appendLabeledSampleAndRetrain(
                "new_frame.png", "LOT_VAL_OCC_CALIB", new ParkingSlot(8, 10, 10, 80, 100),
                newSample, SlotStatus.OCCUPIED, "REVIEWER_ALICE", null, false)
        );
        assertThrows(IllegalStateException.class, () ->
            customService.appendLabeledSampleAndRetrain(
                "new_frame.png", "LOT_TEST_OCC_HOLD", new ParkingSlot(8, 10, 10, 80, 100),
                newSample, SlotStatus.OCCUPIED, "REVIEWER_ALICE", null, false)
        );

        Files.deleteIfExists(tempDataset);
        Files.deleteIfExists(tempManifest);
    }
}
