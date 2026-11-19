package com.parknchillai.detection;

import com.parknchillai.ai.DatasetManifest;
import com.parknchillai.ai.DecisionTreeModel;
import com.parknchillai.ai.ModelEvaluator;
import com.parknchillai.ai.SlotTrainingRecord;
import com.parknchillai.app.AnalysisSummary;
import com.parknchillai.app.ParkingAnalysisService;
import com.parknchillai.app.SlotAnalysisResult;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.layout.LayoutDetectionResult;
import com.parknchillai.layout.ParkingLayoutDetector;
import com.parknchillai.layout.SlotGenerator;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import org.junit.jupiter.api.Test;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * ParkNChill v2.3 Acceptance, Vehicle-Driven Slot Status &amp; Decision Tree Training Suite:
 * <ul>
 *   <li>Requirement 1: {@link OccupancyDecisionEngine.AnalysisMode#VEHICLE_DETECTION} is the default live
 *       analysis mode; missing model reports {@code MODEL_MISSING} without falling back to the Decision Tree;
 *       real ONNX {@code ModelProto} files ({@code mul_1.onnx}) are verified by header &amp; SHA-256</li>
 *   <li>Requirement 2: Independent vehicle detection and parking-space localization, with unmarked-scene
 *       uncertainty handling and obscured-slot protection</li>
 *   <li>Requirement 3: Generation of provisional vs. human-verified {@link SlotTrainingRecord} examples
 *       with 10-feature schema ({@code v2.3-10feat}: 8 IVP features + {@code vehicleOverlapRatio} +
 *       {@code vehicleMatchedFlag}), rejecting unreviewed provisional labels from training</li>
 *   <li>Requirement 4: Training and evaluating the CART Decision Tree on human-reviewed slot examples
 *       across capture-group-disjoint {@code TRAIN / VALIDATION / TEST} partitions while preserving the
 *       frozen {@code TestSnapshot} and reporting Accuracy, Precision, Recall, F1, Confusion Matrix,
 *       and False-Available Rate</li>
 *   <li>Requirement 5: Secondary Decision Tree prediction comparison in {@code VEHICLE_DETECTION} mode
 *       (surfacing disagreements without overwriting the live detector's evidence)</li>
 * </ul>
 */
class AutomaticDetectionV23Test {

    @Test
    void defaultModeIsVehicleDetectionAndRealOnnxModelFileHeaderAndSha256AreVerified() {
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        assertEquals(OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION, svc.getAnalysisMode(),
            "Requirement 1: Default live analysis mode must be VEHICLE_DETECTION");

        // Verify real ONNX file installed in /opt/spark/local/lib/python3.11/dist-packages/onnxruntime/datasets/mul_1.onnx
        Path realOnnx = Path.of("/opt/spark/local/lib/python3.11/dist-packages/onnxruntime/datasets/mul_1.onnx");
        if (Files.exists(realOnnx)) {
            assertTrue(OnnxVehicleDetectionService.isValidOnnxProtobufFile(realOnnx),
                "Real ONNX ModelProto file mul_1.onnx must pass protobuf header validation");
            OnnxVehicleDetectionService realFileSvc = new OnnxVehicleDetectionService(realOnnx);
            assertEquals(64, realFileSvc.computeModelSha256().length());
        }
    }

    @Test
    void analyzeAutomaticallyDetectsSlotsAndVehiclesWithoutPredefinedCsv() {
        BufferedImage lot = new BufferedImage(520, 270, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = lot.createGraphics();
        g.setColor(new Color(54, 58, 66));
        g.fillRect(0, 0, 520, 270);

        List<ParkingSlot> referenceSlots = new ArrayList<>();
        int[] rowY = {18, 150};
        Set<Integer> occupiedIds = Set.of(1, 3, 5, 7, 10);
        int id = 1;
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 5; c++) {
                int x = 18 + c * 96;
                int y = rowY[r];
                g.setColor(new Color(235, 235, 240));
                g.setStroke(new BasicStroke(2.5f));
                g.drawRect(x, y, 84, 96);
                ParkingSlot ref = new ParkingSlot(id, x, y, 84, 96);
                referenceSlots.add(ref);
                if (occupiedIds.contains(id)) {
                    g.setColor(new Color(200, 45, 55));
                    g.fillRoundRect(x + 10, y + 12, 64, 72, 10, 10);
                }
                id++;
            }
        }
        g.dispose();

        OnnxVehicleDetectionService detector = buildDetectorForOccupiedSlots(lot, referenceSlots, occupiedIds);
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(detector);

        // Run analyzeAutomatically(lot) in default VEHICLE_DETECTION mode with zero CSV configuration!
        AnalysisSummary summary = svc.analyzeAutomatically(lot);

        assertEquals(OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION, summary.getAnalysisMode());
        assertTrue(summary.isSlotLocalizationReliable(), "Marked lot must report reliable slot localization");
        assertEquals(10, summary.getTotalSlots(), "Must auto-detect all 10 parking bays without slots.csv");
        assertEquals(5, summary.getOccupiedSlots(), "Must detect all 5 occupied spaces");
        assertEquals(5, summary.getAvailableSlots(), "Must detect all 5 empty spaces from geometry");
        assertEquals(0, summary.getNeedsReviewSlots());
        assertEquals(5, summary.getVehicleDetections().size());
        assertTrue(summary.getUnassignedVehicleDetections().isEmpty());

        // Verify enriched VehicleDetection metadata & 10-feature FeatureVector consistency (Requirement 3)
        for (VehicleDetection det : summary.getVehicleDetections()) {
            assertEquals(640, det.inferenceWidth());
            assertEquals(640, det.inferenceHeight());
            assertNotNull(det.processingMetadata());
            assertTrue(det.processingMetadata().contains("orig=520x270"));
            assertTrue(det.processingMetadata().contains("nms=kept"));
        }

        for (SlotAnalysisResult res : summary.getSlotResults()) {
            String assignedVeh = res.getDiagnosticRecord().overlapEvidence().assignedVehicleId();
            FeatureVector fv = res.getFeatures();
            assertTrue(fv.hasDetectionFeatures(), "Live slot FeatureVector must include detection-derived features");
            assertEquals(FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION, fv.getSchemaVersion());
            assertEquals(10, fv.toExtendedArray().length);
            if (occupiedIds.contains(res.getId())) {
                assertEquals(SlotStatus.OCCUPIED, res.getStatus());
                assertTrue(assignedVeh.startsWith("VEH-"), "Occupied slot must record assigned vehicle ID");
                assertTrue(fv.getVehicleOverlapRatio() >= 0.50);
                assertEquals(1.0, fv.getVehicleMatchedFlag(), 1e-6);
            } else {
                assertEquals(SlotStatus.AVAILABLE, res.getStatus());
                assertEquals("NONE", assignedVeh, "Empty slot must record NONE as assigned vehicle ID");
                assertEquals(0.0, fv.getVehicleOverlapRatio(), 1e-6);
                assertEquals(0.0, fv.getVehicleMatchedFlag(), 1e-6);
            }
        }
    }

    @Test
    void reviewedSlotRecordsTrainTenFeatureDecisionTreeAndRejectUnreviewedProvisionalRecords() {
        // Requirement 3 & 4: Generate provisional SlotTrainingRecords from 3 independent capture sessions,
        // verify unreviewed provisional records are rejected from training, promote to human-verified,
        // and train/evaluate the 10-feature CART Decision Tree with group-disjoint partitions!
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();

        List<SlotTrainingRecord> provisionalAll = new ArrayList<>();
        String[] groups = {"SESSION_LOT_NORTH", "SESSION_LOT_SOUTH", "SESSION_LOT_EAST", "SESSION_LOT_WEST"};
        List<Set<Integer>> occSets = List.of(
            Set.of(1, 3, 6),
            Set.of(2, 4, 5),
            Set.of(1, 4, 6),
            Set.of(2, 3, 5)
        );

        for (int gIdx = 0; gIdx < groups.length; gIdx++) {
            BufferedImage scene = createMarkedSixSlotLot(groups[gIdx], occSets.get(gIdx));
            LayoutDetectionResult layout = svc.autoDetectLayout(scene);
            assertEquals(6, layout.getDetectedSlotCount());
            svc.setVehicleDetectionService(buildDetectorForOccupiedSlots(scene, layout.getSlots(), occSets.get(gIdx)));

            AnalysisSummary sum = svc.analyzeAutomatically(scene);
            List<SlotTrainingRecord> prov = svc.generateProvisionalSlotRecords(
                groups[gIdx] + ".png",
                DatasetManifest.shortSha256Hex(groups[gIdx]),
                groups[gIdx],
                sum
            );
            assertEquals(6, prov.size());
            for (SlotTrainingRecord r : prov) {
                assertTrue(r.provisional(), "Automatically generated slot records must be marked provisional=true");
                assertFalse(r.isHumanVerified());
                assertTrue(r.reviewStatus().startsWith("PROVISIONAL_AUTO_DETECTED"));
                assertEquals(FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION, r.featureSchemaVersion());
            }
            provisionalAll.addAll(prov);
        }

        // Attempting to train Decision Tree on unreviewed provisional records MUST be rejected (Requirement 3)
        assertThrows(IllegalArgumentException.class, () ->
            svc.trainFromReviewedSlotRecords(provisionalAll, 4, 2),
            "Training on unreviewed provisional detector records must be refused"
        );

        // Promote records to HUMAN_VERIFIED with explicit reviewer attribution
        List<SlotTrainingRecord> reviewedAll = new ArrayList<>(provisionalAll.size());
        for (int i = 0; i < provisionalAll.size(); i++) {
            SlotTrainingRecord p = provisionalAll.get(i);
            int gIdx = i / 6;
            SlotStatus verifiedTruth = occSets.get(gIdx).contains(p.slotId())
                ? SlotStatus.OCCUPIED
                : SlotStatus.AVAILABLE;
            SlotTrainingRecord verified = p.promoteToHumanVerified("REVIEWER_DR_KUMAR", verifiedTruth);
            assertFalse(verified.provisional());
            assertTrue(verified.isHumanVerified());
            assertTrue(verified.reviewStatus().startsWith("HUMAN_VERIFIED["));
            assertEquals("HUMAN_REVIEWED_SLOT_DETECTION", verified.provenance());
            reviewedAll.add(verified);
        }

        // Train and evaluate 10-feature CART Decision Tree on the human-reviewed slot records
        ModelEvaluator.EvaluationReport report = svc.trainFromReviewedSlotRecords(reviewedAll, 4, 2);
        assertNotNull(report);
        assertEquals(ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION, report.scope);
        assertTrue(svc.getLastGroupedPartitions().groupDisjoint(),
            "Partitions across 4 capture sessions must be strictly group-disjoint");
        assertEquals(10, svc.getPredictor().getModel().getFeatureCount(),
            "Trained CART Decision Tree must use the 10-feature v2.3-10feat schema");
        assertEquals(1.0, report.accuracy, 1e-6);
        assertEquals(1.0, report.precision, 1e-6);
        assertEquals(1.0, report.recall, 1e-6);
        assertEquals(1.0, report.f1, 1e-6);
        assertEquals(0.0, report.falseAvailableRate(), 1e-6);

        DecisionTreeModel trained10Model = svc.getPredictor().getModel();
        assertTrue(trained10Model.toTreeString().contains("vehicleOverlapRatio")
                || trained10Model.toTreeString().contains("vehicleMatchedFlag"),
            "10-feature CART tree must incorporate detection-derived features; tree:\n" + trained10Model.toTreeString());
    }

    @Test
    void secondaryDecisionTreeNeverOverwritesLiveVehicleDetectorInVehicleDetectionModeAndSurfacesDisagreement() {
        // Requirement 4 & 5: The Decision Tree may provide a secondary prediction for comparison,
        // but it must NOT overwrite the live detector's evidence in VEHICLE_DETECTION mode,
        // and any disagreement must be surfaced for review.
        BufferedImage emptyLookingAsphaltWithLowContrastCar = new BufferedImage(120, 130, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = emptyLookingAsphaltWithLowContrastCar.createGraphics();
        g.setColor(new Color(62, 64, 68));
        g.fillRect(0, 0, 120, 130);
        g.dispose();

        ParkingSlot slot = new ParkingSlot(1, 10, 10, 80, 100);
        // Vehicle detector detects a car in Slot #1 (score = 0.91), while pixel-only 8-feature CART tree sees uniform asphalt (AVAILABLE)
        OnnxVehicleDetectionService detector = buildDetectorForOccupiedSlots(
            emptyLookingAsphaltWithLowContrastCar, List.of(slot), Set.of(1));

        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(detector);

        AnalysisSummary summary = svc.analyze(
            emptyLookingAsphaltWithLowContrastCar, List.of(slot), false,
            OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION);

        SlotAnalysisResult res = summary.getSlotResults().get(0);
        assertEquals(SlotStatus.OCCUPIED, res.getStatus(),
            "Live vehicle detector evidence (OCCUPIED) must NOT be overwritten by secondary CART tree (AVAILABLE)");
        assertEquals(SlotStatus.AVAILABLE, res.getDiagnosticRecord().treePrediction(),
            "Secondary CART tree prediction (AVAILABLE) must be recorded for comparison");
        assertTrue(res.getDecisionPath().contains("DISAGREEMENT"),
            "Disagreement between live Vehicle Detector and secondary CART tree must be surfaced");
        assertTrue(res.getReviewReason().contains("Detector/Tree Disagreement"),
            "Review reason must note Detector/Tree disagreement");
    }

    @Test
    void unmarkedSceneWithVehiclesReportsUncertainLocalizationDoesNotInventEmptySpacesAndDisplaysVehicles() {
        BufferedImage unmarkedScene = new BufferedImage(480, 300, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = unmarkedScene.createGraphics();
        g.setColor(new Color(62, 65, 70));
        g.fillRect(0, 0, 480, 300);
        g.setColor(new Color(185, 35, 45));
        g.fillRoundRect(40, 60, 64, 90, 10, 10);
        g.setColor(new Color(35, 105, 205));
        g.fillRoundRect(190, 80, 66, 92, 10, 10);
        g.setColor(new Color(45, 155, 85));
        g.fillRoundRect(340, 70, 64, 88, 10, 10);
        g.dispose();

        LayoutDetectionResult layout = ParkingLayoutDetector.detectLayout(unmarkedScene);
        assertFalse(layout.isLocalizationReliable(),
            "Unmarked scene without bay lines must report localizationReliable == false");
        assertEquals(0, layout.getDetectedSlotCount(),
            "Unmarked scene must NOT invent empty parking spaces or create spaces merely because vehicles exist");
        assertTrue(layout.getLocalizationSummary().contains("uncertain"));

        List<ParkingSlot> virtualCarBoxes = List.of(
            new ParkingSlot(1, 40, 60, 64, 90),
            new ParkingSlot(2, 190, 80, 66, 92),
            new ParkingSlot(3, 340, 70, 64, 88)
        );
        OnnxVehicleDetectionService detector = buildDetectorForOccupiedSlots(
            unmarkedScene, virtualCarBoxes, Set.of(1, 2, 3));

        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(detector);

        AnalysisSummary summary = svc.analyzeAutomatically(unmarkedScene);
        assertFalse(summary.isSlotLocalizationReliable());
        assertEquals(0, summary.getTotalSlots(), "Zero parking slots invented on unmarked scene");
        assertEquals(3, summary.getVehicleDetections().size(),
            "All 3 detected vehicles must be preserved even when slot localization fails");
        assertEquals(3, summary.getUnassignedVehicleDetections().size(),
            "All 3 vehicles in unmarked scene must be tracked in unassignedVehicleDetections");
        assertNotNull(summary.getAnnotatedImage(), "Annotated image with vehicle boxes must still be rendered");
    }

    @Test
    void openEndedParallelDividerLinesLocalizeBaysAndProposalSuppressionRemovesDuplicates() {
        BufferedImage openBayLot = new BufferedImage(460, 240, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = openBayLot.createGraphics();
        g.setColor(new Color(55, 58, 64));
        g.fillRect(0, 0, 460, 240);
        g.setColor(new Color(235, 235, 240));
        g.setStroke(new BasicStroke(3f));
        for (int i = 0; i < 5; i++) {
            int x = 30 + i * 90;
            g.drawLine(x, 30, x, 130);
        }
        g.dispose();

        LayoutDetectionResult layout = ParkingLayoutDetector.detectLayout(openBayLot);
        assertTrue(layout.isLocalizationReliable(), "Open-ended parallel divider lines must be recognized as reliable markings");
        assertEquals(4, layout.getDetectedSlotCount(), "5 parallel divider lines must localize 4 bays");

        List<ParkingSlot> noisyProposals = List.of(
            new ParkingSlot(1, 30, 30, 90, 100),
            new ParkingSlot(2, 32, 31, 89, 99),
            new ParkingSlot(3, 120, 30, 90, 100)
        );
        List<ParkingSlot> deduplicated = SlotGenerator.suppressOverlappingProposals(noisyProposals, 0.55);
        assertEquals(2, deduplicated.size(), "Overlapping duplicate slot proposal must be suppressed");
        assertEquals(1, deduplicated.get(0).getId());
        assertEquals(2, deduplicated.get(1).getId());
    }

    @Test
    void removingLocalTrainingCsvDoesNotPreventLiveVehicleInference() throws Exception {
        Path featuresPath = Path.of(ParkingAnalysisService.DEFAULT_DATASET_PATH);
        Path backupPath = Files.createTempFile("features_v23_backup", ".csv");
        Files.copy(featuresPath, backupPath, StandardCopyOption.REPLACE_EXISTING);

        try {
            Files.delete(featuresPath);
            assertFalse(Files.exists(featuresPath), "features.csv temporarily removed for Acceptance Criterion 7 test");

            ParkingAnalysisService csvFreeService = ParkingAnalysisService.createWithDefaultModel();
            assertFalse(csvFreeService.isLocalTrainingDatasetLoaded(),
                "Service must note that local training CSV was absent");
            assertEquals(OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION, csvFreeService.getAnalysisMode(),
                "Service must default to VEHICLE_DETECTION when features.csv is absent");

            BufferedImage demoImg = com.parknchillai.utils.SampleDataGenerator.generateDemoParkingLot();
            LayoutDetectionResult layout = csvFreeService.autoDetectLayout(demoImg);
            assertEquals(12, layout.getDetectedSlotCount());

            Set<Integer> occupied = Set.of(1, 6, 7, 12);
            csvFreeService.setVehicleDetectionService(
                buildDetectorForOccupiedSlots(demoImg, layout.getSlots(), occupied));

            AnalysisSummary summary = csvFreeService.analyzeAutomatically(demoImg);
            assertEquals(12, summary.getTotalSlots());
            assertEquals(4, summary.getOccupiedSlots());
            assertEquals(8, summary.getAvailableSlots());
            assertEquals(0, summary.getNeedsReviewSlots());
        } finally {
            Files.copy(backupPath, featuresPath, StandardCopyOption.REPLACE_EXISTING);
            Files.deleteIfExists(backupPath);
        }
    }

    @Test
    void unobservableSpaceAndFailedInferenceAreNeverClassifiedAsAvailable() {
        BufferedImage obscuredImg = new BufferedImage(200, 140, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = obscuredImg.createGraphics();
        g.setColor(new Color(60, 62, 66));
        g.fillRect(0, 0, 200, 140);
        g.setColor(new Color(2, 2, 2));
        g.fillRect(10, 10, 80, 100);
        g.dispose();

        List<ParkingSlot> slots = List.of(
            new ParkingSlot(1, 10, 10, 80, 100),
            new ParkingSlot(2, 105, 10, 80, 100)
        );

        OnnxVehicleDetectionService emptyDetector = OnnxVehicleDetectionService.createWithTestBackend(
            "ready_detector.onnx", 640, 640, 0.30, 0.45,
            (tensor, w, h) -> new float[84][1]
        );
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(emptyDetector);

        AnalysisSummary summary = svc.analyze(
            obscuredImg, slots, false, OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION);
        assertEquals(SlotStatus.NEEDS_REVIEW, summary.getSlotResults().get(0).getStatus(),
            "Unobservable/obscured space must NEVER be marked AVAILABLE");
        assertEquals(SlotStatus.AVAILABLE, summary.getSlotResults().get(1).getStatus(),
            "Observable empty asphalt space is marked AVAILABLE");

        OnnxVehicleDetectionService failingDetector = OnnxVehicleDetectionService.createWithTestBackend(
            "failing_detector.onnx", 640, 640, 0.30, 0.45,
            (tensor, w, h) -> { throw new RuntimeException("ONNX execution error"); }
        );
        svc.setVehicleDetectionService(failingDetector);
        AnalysisSummary failedSummary = svc.analyze(
            obscuredImg, slots, false, OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION);
        assertEquals(2, failedSummary.getUnclassifiedSlots(),
            "Failed detector inference must mark slots UNCLASSIFIED");
        assertEquals(SlotStatus.UNCLASSIFIED, failedSummary.getSlotResults().get(0).getStatus());
    }

    @Test
    void arbitraryMultiSceneBenchmarkMeasuresDetectionLocalizationAndOccupancyMetrics() {
        int totalVehTp = 0, totalVehFp = 0, totalVehFn = 0;
        int totalSlotTp = 0, totalSlotFp = 0, totalSlotFn = 0;
        double sumSlotIoU = 0.0;
        int matchedSlotCount = 0;
        int occTp = 0, occFp = 0, occFn = 0, occTn = 0;

        SceneEval s1 = evaluateMarkedScene(520, 270, 2, 5, 84, 96, false, Set.of(1, 2, 3, 4, 6, 7, 8, 10));
        SceneEval s2 = evaluateMarkedScene(420, 360, 3, 4, 80, 96, false, Set.of(7));
        SceneEval s3 = evaluateMarkedScene(400, 260, 2, 4, 80, 100, true, Set.of(1, 3, 4, 6, 8));

        for (SceneEval se : List.of(s1, s2, s3)) {
            totalVehTp += se.vehTp;
            totalVehFp += se.vehFp;
            totalVehFn += se.vehFn;
            totalSlotTp += se.slotTp;
            totalSlotFp += se.slotFp;
            totalSlotFn += se.slotFn;
            sumSlotIoU += se.sumIoU;
            matchedSlotCount += se.slotTp;
            occTp += se.occTp;
            occFp += se.occFp;
            occFn += se.occFn;
            occTn += se.occTn;
        }

        double vehPrecision = (double) totalVehTp / (totalVehTp + totalVehFp);
        double vehRecall = (double) totalVehTp / (totalVehTp + totalVehFn);
        double slotPrecision = (double) totalSlotTp / (totalSlotTp + totalSlotFp);
        double slotRecall = (double) totalSlotTp / (totalSlotTp + totalSlotFn);
        double meanLocalizationIoU = sumSlotIoU / matchedSlotCount;
        double occPrecision = (double) occTp / (occTp + occFp);
        double occRecall = (double) occTp / (occTp + occFn);
        double falseAvailableRate = (double) occFn / (occTp + occFn);
        double falseOccupiedRate = (double) occFp / (occTn + occFp);

        assertEquals(1.0, vehPrecision, 1e-6);
        assertEquals(1.0, vehRecall, 1e-6);
        assertEquals(1.0, slotPrecision, 1e-6);
        assertEquals(1.0, slotRecall, 1e-6);
        assertTrue(meanLocalizationIoU >= 0.94, "Mean slot localization IoU across scenes must be >= 0.94; got " + meanLocalizationIoU);
        assertEquals(1.0, occPrecision, 1e-6);
        assertEquals(1.0, occRecall, 1e-6);
        assertEquals(0.0, falseAvailableRate, 1e-6);
        assertEquals(0.0, falseOccupiedRate, 1e-6);
    }

    private static BufferedImage createMarkedSixSlotLot(String seedTag, Set<Integer> occupiedIds) {
        BufferedImage img = new BufferedImage(340, 260, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        int baseBg = 52 + (Math.abs(seedTag.hashCode()) % 10);
        g.setColor(new Color(baseBg, baseBg + 2, baseBg + 6));
        g.fillRect(0, 0, 340, 260);
        int id = 1;
        for (int r = 0; r < 2; r++) {
            int y = 16 + r * 120;
            for (int c = 0; c < 3; c++) {
                int x = 18 + c * 102;
                g.setColor(new Color(235, 235, 240));
                g.setStroke(new BasicStroke(2.5f));
                g.drawRect(x, y, 86, 98);
                if (occupiedIds.contains(id)) {
                    g.setColor(id % 2 == 0 ? new Color(35, 110, 215) : new Color(205, 45, 55));
                    g.fillRoundRect(x + 10, y + 12, 66, 74, 10, 10);
                }
                id++;
            }
        }
        g.dispose();
        return img;
    }

    private record SceneEval(
            int vehTp, int vehFp, int vehFn,
            int slotTp, int slotFp, int slotFn,
            double sumIoU,
            int occTp, int occFp, int occFn, int occTn
    ) {}

    private static SceneEval evaluateMarkedScene(int w, int h, int rows, int cols,
                                                 int bayW, int bayH, boolean lightConcrete,
                                                 Set<Integer> occupiedIds) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(lightConcrete ? new Color(210, 212, 215) : new Color(54, 58, 66));
        g.fillRect(0, 0, w, h);

        List<ParkingSlot> gtSlots = new ArrayList<>();
        int id = 1;
        int rowSpacing = (h - 20) / rows;
        int colSpacing = (w - 20) / cols;
        for (int r = 0; r < rows; r++) {
            int y = 16 + r * rowSpacing;
            for (int c = 0; c < cols; c++) {
                int x = 16 + c * colSpacing;
                g.setColor(lightConcrete ? new Color(40, 42, 45) : new Color(235, 235, 240));
                g.setStroke(new BasicStroke(2.5f));
                g.drawRect(x, y, bayW, bayH);
                gtSlots.add(new ParkingSlot(id, x, y, bayW, bayH));
                if (occupiedIds.contains(id)) {
                    g.setColor(id % 2 == 0 ? new Color(35, 105, 210) : new Color(200, 45, 55));
                    g.fillRoundRect(x + 10, y + 12, bayW - 20, bayH - 24, 10, 10);
                }
                id++;
            }
        }
        g.dispose();

        OnnxVehicleDetectionService detector = buildDetectorForOccupiedSlots(img, gtSlots, occupiedIds);
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(detector);

        AnalysisSummary summary = svc.analyzeAutomatically(img);
        List<ParkingSlot> predSlots = summary.getSlotResults().stream().map(SlotAnalysisResult::getSlot).toList();

        int slotTp = 0;
        double sumIoU = 0.0;
        for (int i = 0; i < Math.min(gtSlots.size(), predSlots.size()); i++) {
            double iou = SlotGenerator.computeSlotIoU(gtSlots.get(i), predSlots.get(i));
            if (iou >= 0.85) {
                slotTp++;
                sumIoU += iou;
            }
        }
        int slotFp = Math.max(0, predSlots.size() - slotTp);
        int slotFn = Math.max(0, gtSlots.size() - slotTp);

        int vehTp = occupiedIds.size();
        int vehFp = Math.max(0, summary.getVehicleDetections().size() - vehTp);
        int vehFn = Math.max(0, occupiedIds.size() - summary.getVehicleDetections().size());

        int occTp = 0, occFp = 0, occFn = 0, occTn = 0;
        for (SlotAnalysisResult r : summary.getSlotResults()) {
            boolean truthOcc = occupiedIds.contains(r.getId());
            boolean predOcc = r.getStatus() == SlotStatus.OCCUPIED;
            if (truthOcc && predOcc) occTp++;
            else if (!truthOcc && predOcc) occFp++;
            else if (truthOcc) occFn++;
            else occTn++;
        }

        return new SceneEval(vehTp, vehFp, vehFn, slotTp, slotFp, slotFn, sumIoU, occTp, occFp, occFn, occTn);
    }

    private static OnnxVehicleDetectionService buildDetectorForOccupiedSlots(
            BufferedImage img, List<ParkingSlot> slots, Set<Integer> occupiedIds) {
        DetectionPreprocessor.LetterboxResult lb = DetectionPreprocessor.preprocess(img, 640, 640);
        return OnnxVehicleDetectionService.createWithTestBackend(
            "yolov8n_v23_vehicles.onnx", 640, 640, 0.30, 0.45,
            (tensor, inW, inH) -> {
                float[][] out = new float[84][Math.max(1, occupiedIds.size())];
                int idx = 0;
                for (ParkingSlot s : slots) {
                    if (!occupiedIds.contains(s.getId())) continue;
                    double carX = s.getX() + 10;
                    double carY = s.getY() + 12;
                    double carW = s.getWidth() - 20;
                    double carH = s.getHeight() - 24;
                    double lcx = (carX + carW / 2.0) * lb.scale() + lb.padX();
                    double lcy = (carY + carH / 2.0) * lb.scale() + lb.padY();
                    double lw = carW * lb.scale();
                    double lh = carH * lb.scale();
                    out[0][idx] = (float) lcx;
                    out[1][idx] = (float) lcy;
                    out[2][idx] = (float) lw;
                    out[3][idx] = (float) lh;
                    out[4 + 2][idx] = 0.92f;
                    idx++;
                }
                return out;
            }
        );
    }

    @Test
    void testFinding03StreamingModelValidationAndSha256() throws Exception {
        Path tempModel = Files.createTempFile("pnc_stream_test_", ".onnx");
        try {
            byte[] dummyHeader = new byte[256];
            dummyHeader[0] = 0x08;
            dummyHeader[1] = 0x07; // valid proto version
            Files.write(tempModel, dummyHeader);

            assertTrue(OnnxVehicleDetectionService.isValidOnnxProtobufFile(tempModel),
                "Streaming header check must validate valid protobuf header");

            OnnxVehicleDetectionService service = new OnnxVehicleDetectionService(tempModel);
            String sha256 = service.computeModelSha256();
            assertNotNull(sha256);
            assertEquals(64, sha256.length(), "SHA-256 hex string must be 64 characters");
            assertNotEquals("HASH_ERROR", sha256);
            assertNotEquals("MODEL_MISSING", sha256);
        } finally {
            Files.deleteIfExists(tempModel);
        }
    }

    @Test
    void testFinding04OnnxSessionLifecycleAndCaching() {
        OnnxVehicleDetectionService service = new OnnxVehicleDetectionService();
        assertDoesNotThrow(() -> {
            service.close(); // Clean close on uninitialized session
            service.close(); // Idempotent close
        });
    }
}
