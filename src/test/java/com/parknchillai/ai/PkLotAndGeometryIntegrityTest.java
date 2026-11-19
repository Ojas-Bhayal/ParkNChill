package com.parknchillai.ai;

import com.parknchillai.app.AppState;
import com.parknchillai.app.ParkingAnalysisService;
import com.parknchillai.detection.OnnxVehicleDetectionService;
import com.parknchillai.detection.SlotOverlapAnalyzer;
import com.parknchillai.detection.VehicleCategory;
import com.parknchillai.detection.VehicleDetection;
import com.parknchillai.detection.VehicleDetectionService;
import com.parknchillai.features.FeatureExtractor;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.RoiExtractor;
import com.parknchillai.parking.SlotStatus;
import com.parknchillai.ui.MainWindow;
import com.parknchillai.utils.SampleDataGenerator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Deterministic Unit &amp; Fixture Test Suite verifying the 7 ParkNChill v2.3 Integrity Fixes:
 * <ol>
 *   <li>First-run ONNX model setup &amp; header/SHA-256 verification ({@link OnnxVehicleDetectionService#installAndVerifyModel})</li>
 *   <li>Partition-aware PKLot training consuming saved {@code TRAIN}, {@code VALIDATION}, and frozen {@code TEST}
 *       assignments, validation hyperparameter selection, and separate User-Reviewed vs PKLot pools</li>
 *   <li>Explicit {@code IMPORTED_GROUND_TRUTH} provenance, prohibition of fabricated {@code PKLOT_ANNOTATION}
 *       reviewer IDs, and auditable user label corrections preserving {@code originalSourceLabel}</li>
 *   <li>Genuinely asynchronous background PKLot import with UI responsiveness, cancellation, and
 *       protection of previously valid datasets</li>
 *   <li>Preservation of actual PKLot polygon / angled quadrilateral {@code <contour>} and {@code <rotatedRect>}
 *       geometry, Sutherland-Hodgman polygon vehicle overlap, and perspective-rectified ROI feature extraction</li>
 * </ol>
 */
class PkLotAndGeometryIntegrityTest {
    static {
        initJavaFxToolkit();
    }

    private static synchronized void initJavaFxToolkit() {
        try {
            System.setProperty("prism.order", "sw");
            System.setProperty("prism.text", "t2k");
            javafx.application.Platform.startup(() -> {});
            javafx.application.Platform.setImplicitExit(false);
        } catch (Throwable ignored) {
            // Platform already initialized or running under headless stub
        }
    }



    @TempDir
    Path tempDir;

    private static VehicleDetectionService createFixtureDetector() {
        return new VehicleDetectionService() {
            @Override
            public DetectorStatus getStatus() {
                return DetectorStatus.READY;
            }

            @Override
            public String describeConfiguration() {
                return "fixture-detector-v23";
            }

            @Override
            public DetectionBatchResult detectVehicles(BufferedImage fullImage) {
                List<VehicleDetection> dets = List.of(
                    new VehicleDetection("V1", VehicleCategory.CAR, 20, 26, 62, 84, 0.94, "fixture-v23", 640, 640, "test"),
                    new VehicleDetection("V2", VehicleCategory.CAR, 108, 136, 64, 84, 0.91, "fixture-v23", 640, 640, "test")
                );
                return new DetectionBatchResult(DetectorStatus.READY, "fixture-detector-v23", "Apache-2.0", "OK", dets, dets, 2L);
            }
        };
    }


    @Test
    void pklotPartitionAssignmentDistributesWholeCameraDateGroupsAcrossAllThreePartitions() {
        List<String> groups = List.of(
            "PKLOT_PUCPR_2012-10-12",
            "PKLOT_PUCPR_2012-10-13",
            "PKLOT_PUCPR_2012-10-14",
            "PKLOT_UFPR04_2012-12-07",
            "PKLOT_UFPR04_2012-12-08",
            "PKLOT_UFPR05_2012-12-15"
        );
        Map<String, String> assignments = PkLotDatasetImporter.assignCameraDatePartitions(groups);
        assertEquals(6, assignments.size());
        assertEquals(2, assignments.values().stream().filter("TRAIN"::equals).count());
        assertEquals(2, assignments.values().stream().filter("VALIDATION"::equals).count());
        assertEquals(2, assignments.values().stream().filter("TEST"::equals).count());
        assertEquals("TRAIN", assignments.get(groups.get(0)));
        assertEquals("VALIDATION", assignments.get(groups.get(1)));
        assertEquals("TEST", assignments.get(groups.get(2)));
    }

    @Test
    void requirement1FirstRunOnnxSetupVerifiesProtobufHeaderComputesSha256AndRejectsInvalidFiles() throws Exception {
        // 1. Corrupt / non-ONNX file must be rejected without claiming readiness
        Path badFile = tempDir.resolve("not_an_onnx.onnx");
        Files.writeString(badFile, "THIS_IS_PLAIN_TEXT_NOT_ONNX_PROTOBUF".repeat(10));
        Path targetPath = tempDir.resolve("models/vehicle_detector.onnx");

        OnnxVehicleDetectionService.ModelSetupResult badRes =
            OnnxVehicleDetectionService.installAndVerifyModel(badFile, targetPath);
        assertFalse(badRes.validOnnxHeader());
        assertEquals(VehicleDetectionService.DetectorStatus.MODEL_INCOMPATIBLE, badRes.resultingStatus());
        assertFalse(Files.exists(targetPath), "Invalid ONNX file must not be copied to target model path");

        // 2. Valid ONNX ModelProto header (tag 0x08, ir_version = 8) installs and computes 64-hex SHA-256
        Path validHeaderFile = tempDir.resolve("valid_header.onnx");
        byte[] protoBytes = new byte[256];
        protoBytes[0] = 0x08; // field 1 ir_version varint tag
        protoBytes[1] = 0x08; // ir_version = 8
        for (int i = 2; i < protoBytes.length; i++) {
            protoBytes[i] = (byte) (i & 0x7F);
        }
        Files.write(validHeaderFile, protoBytes);

        OnnxVehicleDetectionService.ModelSetupResult goodRes =
            OnnxVehicleDetectionService.installAndVerifyModel(validHeaderFile, targetPath);
        assertTrue(goodRes.validOnnxHeader());
        assertTrue(Files.exists(targetPath));
        assertEquals(256L, goodRes.fileSizeBytes());
        assertEquals(64, goodRes.sha256().length());
    }

    @Test
    void requirement2And3CheckedInPkLotFixturesPreservePartitionsEnforceImportedProvenanceAndAuditUserCorrections() throws Exception {
        Path checkedInFixtures = Path.of("data/fixtures/pklot");
        assertTrue(Files.exists(checkedInFixtures), "Checked-in PKLot fixtures must exist at data/fixtures/pklot");

        PkLotDatasetImporter.PkLotImportResult importRes = PkLotDatasetImporter.importDirectory(
            checkedInFixtures,
            createFixtureDetector(),
            SlotOverlapAnalyzer.OverlapPolicy.defaultValidationTunedPolicy(),
            false,
            p -> {},
            () -> false
        );

        assertEquals(3, importRes.validImageCount());
        assertEquals(12, importRes.validSlotCount());
        assertEquals(6, importRes.occupiedCount());
        assertEquals(6, importRes.availableCount());
        assertEquals(0, importRes.detectorFailureCount());

        // Verify saved camera/date group partitions across PUCPR, UFPR04, UFPR05
        Map<String, String> partMap = importRes.groupPartitionMap();
        assertEquals(Set.of("PKLOT_PUCPR_2012-10-12", "PKLOT_UFPR04_2012-12-07", "PKLOT_UFPR05_2012-12-15"),
            partMap.keySet());
        assertTrue(partMap.containsValue("TRAIN"));
        assertTrue(partMap.containsValue("VALIDATION"));
        assertTrue(partMap.containsValue("TEST"));

        // Verify Requirement 3: Every imported record has IMPORTED_GROUND_TRUTH, empty reviewerId, and XML path
        for (SlotTrainingRecord rec : importRes.eligibleRecords()) {
            assertEquals(SlotTrainingRecord.PROVENANCE_IMPORTED_GROUND_TRUTH, rec.provenance());
            assertEquals("PKLot", rec.sourceDataset());
            assertTrue(rec.annotationFilePath().endsWith(".xml"), "Must record XML annotation file path");
            assertEquals("", rec.reviewerId(), "Imported records must NOT have a fabricated reviewerId");
            assertFalse(rec.isHumanVerified(), "Unedited imported records must NOT be marked HUMAN_VERIFIED");
            assertTrue(rec.isImportedGroundTruth(), "Must be recognized as validated IMPORTED_GROUND_TRUTH");
            assertFalse(rec.reviewStatus().startsWith("HUMAN_VERIFIED"));
            assertTrue(rec.isEligibleForTraining(
                SlotTrainingRecord.TrainingEligibilityPolicy.HUMAN_VERIFIED_AND_IMPORTED_GROUND_TRUTH));
            assertFalse(rec.isEligibleForTraining(
                SlotTrainingRecord.TrainingEligibilityPolicy.HUMAN_VERIFIED_ONLY));
        }

        // Verify fabricating PKLOT_ANNOTATION reviewerId is explicitly rejected
        SlotTrainingRecord firstRec = importRes.eligibleRecords().get(0);
        assertThrows(IllegalArgumentException.class,
            () -> firstRec.withAuditedUserCorrection("PKLOT_ANNOTATION", SlotStatus.AVAILABLE, "bad reviewer"),
            "Fabricated reviewerId PKLOT_ANNOTATION must be rejected");

        // Verify auditable user correction preserves originalSourceLabel without overwriting it
        SlotStatus originalLbl = firstRec.originalSourceLabel();
        SlotStatus flippedLbl = (originalLbl == SlotStatus.OCCUPIED) ? SlotStatus.AVAILABLE : SlotStatus.OCCUPIED;
        SlotTrainingRecord correctedRec = firstRec.withAuditedUserCorrection("AUDITOR_42", flippedLbl, "Shadow inspection");
        assertEquals(originalLbl, correctedRec.originalSourceLabel(),
            "Original ground-truth label must be preserved when user applies an auditable correction");
        assertEquals(flippedLbl, correctedRec.occupancyLabel());
        assertEquals("AUDITOR_42", correctedRec.reviewerId());
        assertTrue(correctedRec.userCorrectionAuditEvent().contains("USER_CORRECTION[by=AUDITOR_42"));
        assertTrue(correctedRec.isHumanVerified());

        // Verify Requirement 2: Main training action routes through saved PKLot partitions,
        // selects hyperparameters on VALIDATION, and freezes TEST partition metadata
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(createFixtureDetector());
        AppState state = new AppState(svc);
        state.registerPkLotImportResult(importRes);
        assertEquals(AppState.ActiveDatasetMode.PKLOT_IMPORTED_DATASET, state.getActiveDatasetMode());
        assertEquals(12, state.getEligibleImportedGroundTruthCount());
        assertEquals(0, state.getEligibleHumanVerifiedCount());

        ModelEvaluator.EvaluationReport testRep = state.trainTenFeatureDecisionTree(4, 1);
        assertNotNull(testRep);
        assertEquals(ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION, testRep.scope);
        assertTrue(testRep.provenanceNote.contains("12 IMPORTED_GROUND_TRUTH, 0 HUMAN_VERIFIED"));

        ParkingAnalysisService.PkLotTrainingSummary pkSummary = svc.getLastPkLotTrainingSummary();
        assertNotNull(pkSummary);
        assertEquals(4, pkSummary.trainCount());
        assertEquals(4, pkSummary.validationCount());
        assertEquals(4, pkSummary.testCount());
        assertEquals(Set.of("PKLOT_PUCPR_2012-10-12"), pkSummary.trainGroups());
        assertEquals(Set.of("PKLOT_UFPR04_2012-12-07"), pkSummary.validationGroups());
        assertEquals(Set.of("PKLOT_UFPR05_2012-12-15"), pkSummary.testGroups());

        // Clean up generated meta file in fixtures dir if written
        Files.deleteIfExists(checkedInFixtures.resolve("pklot_frozen_test.meta"));
    }

    @Test
    void requirement4AsyncPkLotImportRunsOnBackgroundThreadSupportsCancellationAndPreservesPreviousDataset() throws Exception {
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(createFixtureDetector());
        MainWindow window = new MainWindow(svc, false);
        AppState state = window.getAppState();

        // First import valid checked-in fixtures asynchronously and wait for completion
        Path checkedInFixtures = Path.of("data/fixtures/pklot");
        CompletableFuture<PkLotDatasetImporter.PkLotImportResult> future1 =
            window.getModelDetailsView().importPkLotDirectoryAsync(checkedInFixtures);
        PkLotDatasetImporter.PkLotImportResult res1 = future1.get(5, TimeUnit.SECONDS);
        assertNotNull(res1);
        assertEquals(12, state.getEligibleReviewedRecords().size());
        Files.deleteIfExists(checkedInFixtures.resolve("pklot_frozen_test.meta"));

        // Now configure a slow detector so we can observe background execution and cancel mid-import
        svc.setVehicleDetectionService(new VehicleDetectionService() {
            @Override
            public DetectorStatus getStatus() { return DetectorStatus.READY; }
            @Override
            public String describeConfiguration() { return "slow-detector"; }
            @Override
            public DetectionBatchResult detectVehicles(BufferedImage fullImage) {
                try { Thread.sleep(120); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
                return createFixtureDetector().detectVehicles(fullImage);
            }
        });

        CompletableFuture<PkLotDatasetImporter.PkLotImportResult> slowFuture =
            window.getModelDetailsView().importPkLotDirectoryAsync(checkedInFixtures);

        // Verify UI remains responsive on the calling thread and duplicate import is disabled while running
        assertTrue(window.getModelDetailsView().isImportInProgress(),
            "Import must execute asynchronously on a background thread");
        assertTrue(window.getModelDetailsView().getBtnImportPkLot().isDisabled(),
            "Import PKLot button must be disabled while an import is running");
        assertTrue(window.getModelDetailsView().getBtnCancelImport().isVisible(),
            "Cancel Import button must be visible while an import is running");

        // Cancel the in-progress import and verify the previous 12-record dataset is preserved intact!
        window.getModelDetailsView().cancelActiveImport();
        assertThrows(Exception.class, () -> slowFuture.get(5, TimeUnit.SECONDS));
        assertFalse(window.getModelDetailsView().isImportInProgress());
        assertFalse(window.getModelDetailsView().getBtnImportPkLot().isDisabled(),
            "Import PKLot button must be re-enabled after cancellation");
        assertEquals(12, state.getEligibleReviewedRecords().size(),
            "Cancelled import must NEVER partially replace or clear the previously valid dataset");
    }

    @Test
    void requirement5AngledQuadrilateralAndRotatedRectContoursComputeExactPolygonOverlapAndRectifiedFeatures() throws Exception {
        // Parse the checked-in angled XML fixture
        Path xmlPath = Path.of("data/fixtures/pklot/PUCPR/Sunny/2012-10-12/2012-10-12_08_00_00.xml");
        List<PkLotDatasetImporter.ParsedSlotAnnotation> parsed = PkLotDatasetImporter.parsePkLotXml(xmlPath);
        assertEquals(4, parsed.size());

        // Space #1 has 4-point angled contour: (18,24)->(84,30)->(80,116)->(14,110)
        ParkingSlot angledSlot1 = parsed.get(0).toParkingSlot();
        assertTrue(angledSlot1.hasCustomPolygon(), "Angled 4-point PKLot contour must be preserved as a custom polygon");
        assertEquals(4, angledSlot1.getPolygonVertices().size());
        assertEquals(12.0, angledSlot1.getRotationAngleDegrees(), 1e-6);

        // Space #3 has only <rotatedRect center=(230,70) size=(68,88) angle=15> -> must compute 4 rotated corners!
        ParkingSlot rotatedSlot3 = parsed.get(3).toParkingSlot();
        assertTrue(rotatedSlot3.hasCustomPolygon(), "Rotated rectangle with angle=15 deg must preserve 4 rotated corners");
        assertEquals(4, rotatedSlot3.getPolygonVertices().size());
        assertEquals(15.0, rotatedSlot3.getRotationAngleDegrees(), 1e-6);

        // Create a strongly angled parallelogram bay: top at x=[10..50], y=10; bottom at x=[60..100], y=110
        // Notice its axis-aligned bounding box is [10..100] x [10..110] (area = 9000),
        // whereas its true polygon area by Shoelace is base(40) * height(100) = 4000!
        ParkingSlot parallelogramSlot = ParkingSlot.fromPolygon(
            10,
            List.of(
                new double[]{10.0, 10.0},
                new double[]{50.0, 10.0},
                new double[]{100.0, 110.0},
                new double[]{60.0, 110.0}
            ),
            ParkingSlot.GeometryOrigin.PKLOT_POLYGON,
            26.5
        );
        assertTrue(parallelogramSlot.hasCustomPolygon());
        assertEquals(4000.0, parallelogramSlot.computeExactArea(), 1e-4,
            "Shoelace formula must compute exact parallelogram area (4000), not bounding-box area (9000)");

        // Place a vehicle in the bottom-left corner [12, 75, 30x30] (inside the axis-aligned bounding box [10..100, 10..110],
        // but completely OUTSIDE the angled parallelogram!)
        VehicleDetection neighborCarInBoundingBoxCorner = new VehicleDetection(
            "V_NEIGHBOR", VehicleCategory.CAR, 12, 75, 30, 30, 0.92, "onnx"
        );
        assertFalse(neighborCarInBoundingBoxCorner.isCenterInside(parallelogramSlot),
            "Ray-casting point-in-polygon must recognize that (27, 90) is outside the angled parallelogram");
        assertEquals(0.0, neighborCarInBoundingBoxCorner.intersectionArea(parallelogramSlot), 1e-4,
            "Sutherland-Hodgman polygon clipping must report 0 overlap with a car in the empty bounding-box corner");

        // Place a vehicle centered inside the parallelogram at (55, 60) [40, 45, 30x30]
        VehicleDetection carInsideParallelogram = new VehicleDetection(
            "V_INSIDE", VehicleCategory.CAR, 40, 45, 30, 30, 0.95, "onnx"
        );
        assertTrue(carInsideParallelogram.isCenterInside(parallelogramSlot));
        assertTrue(carInsideParallelogram.intersectionOverSlotArea(parallelogramSlot) >= 0.20);

        // Verify RoiExtractor rectifies the 4-corner angled slot and extracts valid 8 IVP features
        File demoFile = SampleDataGenerator.ensureSampleImageExists(SampleDataGenerator.DEFAULT_SAMPLE_IMAGE_PATH);
        BufferedImage img = ImageIO.read(demoFile);
        BufferedImage rectifiedRoi = RoiExtractor.extract(img, angledSlot1);
        assertNotNull(rectifiedRoi);
        FeatureVector fv = FeatureExtractor.extract(rectifiedRoi);
        assertTrue(fv.hasAllValuesWithinValidBounds());
    }
}