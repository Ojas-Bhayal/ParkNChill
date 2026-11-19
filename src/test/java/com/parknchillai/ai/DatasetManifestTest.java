package com.parknchillai.ai;

import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive regression suite for Phases 1, 2, 3, 4, and 5 of the v2.2 Remaining Fixes Plan:
 * <ul>
 *   <li>{@code PNC-REM-01}: Fail-closed missing-manifest policy, explicit initializers, and legacy import</li>
 *   <li>{@code PNC-REM-02}: State-tracked 3-target transaction ({@code features.csv + manifest.csv + crop.png}),
 *       crop backup restoration, crash recovery ({@code PREPARED / COMMITTING / COMMITTED}), and {@code RECOVERY_REQUIRED}</li>
 *   <li>{@code PNC-REM-03}: Schema-aware deterministic {@code sampleId} and collision rejection</li>
 *   <li>{@code PNC-REM-04}: Explicit reviewer attribution (no {@code OPERATOR}/{@code QA_REVIEWER} fallback)</li>
 *   <li>{@code PNC-REM-06}: Validation and Test group protection and explicit correction flows</li>
 * </ul>
 */
class DatasetManifestTest {

    private Path tempDir;
    private Path featuresFile;
    private Path manifestFile;

    @BeforeEach
    void setUp() throws IOException {
        DatasetManifest.resetTestingHooks();
        tempDir = Files.createTempDirectory("pnc_manifest_rem_test_");
        featuresFile = tempDir.resolve("features.csv");
        manifestFile = tempDir.resolve("manifest.csv");
        Files.copy(Path.of("data/labels/features.csv"), featuresFile, StandardCopyOption.REPLACE_EXISTING);
        Files.copy(Path.of("data/labels/manifest.csv"), manifestFile, StandardCopyOption.REPLACE_EXISTING);
    }

    @AfterEach
    void tearDown() throws IOException {
        DatasetManifest.resetTestingHooks();
        if (tempDir != null && Files.exists(tempDir)) {
            Files.walk(tempDir)
                .sorted((a, b) -> b.compareTo(a))
                .forEach(p -> {
                    try { Files.deleteIfExists(p); } catch (IOException ignored) {}
                });
        }
    }

    // --- Phase 1 (PNC-REM-01): Stop inventing provenance when a manifest is missing ---

    @Test
    void existingFeatureCsvWithMissingManifestFailsClosedWithoutFabricatingManifestOrChangingFeatures() throws IOException {
        Files.deleteIfExists(manifestFile);
        byte[] origFeaturesBytes = Files.readAllBytes(featuresFile);

        IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
            DatasetManifest.loadOrSync(featuresFile.toString(), manifestFile.toString())
        );
        assertTrue(ex.getMessage().contains("Missing provenance manifest"));
        assertFalse(Files.exists(manifestFile), "loadOrSync must NEVER fabricate a manifest file when missing");
        assertArrayEquals(origFeaturesBytes, Files.readAllBytes(featuresFile));
    }

    @Test
    void initializeNewEmptyDatasetCreatesValidSchemaFiles() throws IOException {
        Path emptyFeatures = tempDir.resolve("new_features.csv");
        Path emptyManifest = tempDir.resolve("new_manifest.csv");

        DatasetManifest.initializeNewEmptyDataset(emptyFeatures.toString(), emptyManifest.toString());
        assertTrue(Files.exists(emptyFeatures));
        assertTrue(Files.exists(emptyManifest));

        String fContent = Files.readString(emptyFeatures, StandardCharsets.UTF_8);
        String mContent = Files.readString(emptyManifest, StandardCharsets.UTF_8);
        assertTrue(fContent.contains(FeatureVector.csvHeader()));
        assertTrue(mContent.contains(DatasetManifest.HEADER));
    }

    @Test
    void explicitImportLegacyDatasetProducesBackupAndMarksRowsUnverifiedLegacyWithUngroupedCaptureGroup() throws IOException {
        Files.deleteIfExists(manifestFile);
        DatasetManifest.MigrationReport report = DatasetManifest.importLegacyDataset(
            featuresFile.toString(), manifestFile.toString()
        );

        assertNotNull(report.featuresBackupPath());
        assertTrue(Files.exists(report.featuresBackupPath()));
        assertEquals(48, report.totalRecords());
        assertEquals(0, report.preservedRecords());
        assertEquals(48, report.inferredRecords());

        List<DatasetManifest.ManifestRecord> imported = DatasetManifest.loadManifest(manifestFile.toString());
        assertEquals(48, imported.size());
        for (DatasetManifest.ManifestRecord r : imported) {
            assertEquals("UNVERIFIED_LEGACY", r.reviewStatus());
            assertEquals("LEGACY_IMPORTED", r.provenance());
            assertEquals("LEGACY_UNGROUPED", r.captureGroup());
            assertNotEquals("VERIFIED_REAL_CAMERA", r.provenance());
        }

        // DatasetValidator must report legacyImportedSamples = 48 and groupDisjointEvaluationPossible = false
        DatasetLoader.Dataset ds = DatasetLoader.load(featuresFile.toString());
        DatasetValidator.ValidationSummary vs = DatasetValidator.validate(ds, imported);
        assertEquals(48, vs.legacyImportedSamples());
        assertEquals(0, vs.realCameraVerifiedSamples());
        assertFalse(vs.groupDisjointEvaluationPossible(),
            "Legacy imported rows with unknown capture groups must disable group-disjoint evaluation");
    }

    @Test
    void initializeBundledDemoDatasetNeverMarksAnyRowAsVerifiedRealCamera() throws IOException {
        Path initManifest = tempDir.resolve("bundled_init_manifest.csv");
        List<DatasetManifest.ManifestRecord> records = DatasetManifest.initializeBundledDemoDataset(
            featuresFile.toString(), initManifest.toString()
        );
        assertEquals(48, records.size());
        for (DatasetManifest.ManifestRecord r : records) {
            assertNotEquals("VERIFIED_REAL_CAMERA", r.provenance());
            assertTrue("DEMO_LOT_EXTRACTED".equals(r.provenance()) || "PROCEDURAL_SYNTHETIC".equals(r.provenance()));
        }
    }

    // --- Phase 2 (PNC-REM-03): Schema-aware sample identity & collision rejection ---

    @Test
    void computeSampleIdIsDeterministicAndDistinguishesDifferentSchemaVersions() {
        String idV22a = DatasetManifest.computeSampleId(
            "cam_north.png", "LOT_NORTH", 3, 10, 20, 80, 100, "v2.2-8feat");
        String idV22b = DatasetManifest.computeSampleId(
            "cam_north.png", "LOT_NORTH", 3, 10, 20, 80, 100, "v2.2-8feat");
        String idV20  = DatasetManifest.computeSampleId(
            "cam_north.png", "LOT_NORTH", 3, 10, 20, 80, 100, "v2.0-5feat");

        assertEquals(idV22a, idV22b, "Same canonical tuple and schemaVersion must produce identical sampleId");
        assertNotEquals(idV22a, idV20, "Different schemaVersion must produce a different sampleId");

        assertThrows(IllegalArgumentException.class, () ->
            DatasetManifest.computeSampleId("cam.png", "LOT_1", 1, 0, 0, 80, 100, ""));
        assertThrows(IllegalArgumentException.class, () ->
            DatasetManifest.computeSampleId("cam.png", "LOT_1", 1, 0, 0, 80, 100, "v9.9-unknown"));
    }

    @Test
    void simulatedSampleIdCollisionFailsClosedWithoutChangingEitherFile() throws IOException {
        byte[] origFeaturesBytes = Files.readAllBytes(featuresFile);
        byte[] origManifestBytes = Files.readAllBytes(manifestFile);

        // Force computeSampleId to return "SMP-001", which already belongs to LOT_A_DEMO slot #1
        DatasetManifest.setForcedCollisionSampleIdForTesting("SMP-001");
        FeatureVector fv = new FeatureVector(new double[]{88.0, 0.44, 0.51, 0.46, 0.40, 110.0, 0.52, 0.35});
        ParkingSlot slot = new ParkingSlot(9, 40, 50, 80, 100);

        IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
            DatasetManifest.recordUserLabeledSample(
                featuresFile.toString(), manifestFile.toString(),
                "distinct_image.png", "LOT_DISTINCT", slot, fv, SlotStatus.OCCUPIED, "REV_1", null, false
            )
        );
        assertTrue(ex.getMessage().contains("collision"));
        assertArrayEquals(origFeaturesBytes, Files.readAllBytes(featuresFile));
        assertArrayEquals(origManifestBytes, Files.readAllBytes(manifestFile));
    }

    @Test
    void newSampleAndInPlaceRelabelingPreserveAlignmentAndDistinctCaptures() throws IOException {
        FeatureVector fv1 = new FeatureVector(new double[]{88.0, 0.44, 0.51, 0.46, 0.40, 110.0, 0.52, 0.35});
        FeatureVector fv2 = new FeatureVector(new double[]{64.0, 0.08, 0.82, 0.17, 0.04, 18.0, 0.00, 0.02});
        ParkingSlot slot = new ParkingSlot(4, 25, 30, 80, 100);

        DatasetManifest.ManifestRecord r1 = DatasetManifest.recordUserLabeledSample(
            featuresFile.toString(), manifestFile.toString(),
            "cam_east_lot.png", "LOT_EAST_CAM1", slot, fv1, SlotStatus.OCCUPIED, "REVIEWER_ALICE", null, false
        );
        assertEquals("REVIEWER_ALICE", r1.reviewerId());
        assertEquals(49, DatasetLoader.load(featuresFile.toString()).size());

        // Re-label same source and ROI -> updates in place at index 48
        DatasetManifest.recordUserLabeledSample(
            featuresFile.toString(), manifestFile.toString(),
            "cam_east_lot.png", "LOT_EAST_CAM1", slot, fv2, SlotStatus.AVAILABLE, "REVIEWER_ALICE", null, false
        );
        assertEquals(49, DatasetLoader.load(featuresFile.toString()).size());

        // Different source image with identical feature vector fv2 -> appends a 50th distinct sample
        DatasetManifest.recordUserLabeledSample(
            featuresFile.toString(), manifestFile.toString(),
            "cam_west_lot.png", "LOT_WEST_CAM2", slot, fv2, SlotStatus.AVAILABLE, "REVIEWER_BOB", null, false
        );
        DatasetLoader.Dataset ds = DatasetLoader.load(featuresFile.toString());
        List<DatasetManifest.ManifestRecord> man = DatasetManifest.loadManifest(manifestFile.toString());
        assertEquals(50, ds.size());
        assertEquals(50, man.size());
        assertNotEquals(man.get(48).sampleId(), man.get(49).sampleId());
        DatasetManifest.verifyStrictAlignment(ds, man);
    }

    // --- Phase 3 (PNC-REM-04): Explicit reviewer attribution & no silent fallback ---

    @Test
    void blankReviewerIdIsRejectedAndNoImplicitOperatorOrQaReviewerFallbackIsInserted() {
        FeatureVector fv = new FeatureVector(new double[]{88.0, 0.44, 0.51, 0.46, 0.40, 110.0, 0.52, 0.35});
        ParkingSlot slot = new ParkingSlot(1, 10, 10, 80, 100);

        // 1. Bare USER_REVIEWED status without by=<reviewer> must return empty reviewerId() and isHumanReviewed() == false
        DatasetManifest.ManifestRecord bareStatus = new DatasetManifest.ManifestRecord(
            "SMP-BARE", "cam.png", "GRP_1", 1, 10, 10, 80, 100,
            fv, SlotStatus.OCCUPIED, FeatureVector.SCHEMA_VERSION,
            "USER_REVIEWED", "USER_LABELLED", "TRAIN", "2026-10-21T00:00:00Z"
        );
        assertEquals("", bareStatus.reviewerId(), "Must NOT substitute 'OPERATOR' when by= is absent");
        assertFalse(bareStatus.isHumanReviewed());

        // 2. recordUserLabeledSample rejects blank reviewerId
        assertThrows(IllegalArgumentException.class, () ->
            DatasetManifest.recordUserLabeledSample(
                featuresFile.toString(), manifestFile.toString(),
                "cam.png", "LOT_NEW", slot, fv, SlotStatus.OCCUPIED, "   ", null, false
            )
        );
    }

    // --- Phase 4 (PNC-REM-02): State-tracked 3-target transaction & crop backup/recovery ---

    private BufferedImage createTestCrop(Color c) {
        BufferedImage img = new BufferedImage(80, 100, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(c);
        g.fillRect(0, 0, 80, 100);
        g.dispose();
        return img;
    }

    @Test
    void faultBeforeAndAfterCsvReplacementsRollsBackBothCsvFilesByteForByte() throws IOException {
        byte[] origFeaturesBytes = Files.readAllBytes(featuresFile);
        byte[] origManifestBytes = Files.readAllBytes(manifestFile);
        FeatureVector fv = new FeatureVector(new double[]{88.0, 0.44, 0.51, 0.46, 0.40, 110.0, 0.52, 0.35});
        ParkingSlot slot = new ParkingSlot(5, 10, 10, 80, 100);

        for (DatasetManifest.FaultInjectionStage stage : List.of(
                DatasetManifest.FaultInjectionStage.BEFORE_REPLACEMENT,
                DatasetManifest.FaultInjectionStage.AFTER_FEATURES_REPLACEMENT,
                DatasetManifest.FaultInjectionStage.AFTER_MANIFEST_REPLACEMENT)) {
            DatasetManifest.setFaultInjectionStageForTesting(stage);
            assertThrows(IllegalStateException.class, () ->
                DatasetManifest.recordUserLabeledSample(
                    featuresFile.toString(), manifestFile.toString(),
                    "rollback_lot.png", "LOT_ROLLBACK", slot, fv, SlotStatus.OCCUPIED, "REV_1", null, false
                )
            );
            assertArrayEquals(origFeaturesBytes, Files.readAllBytes(featuresFile), "Failed at stage " + stage);
            assertArrayEquals(origManifestBytes, Files.readAllBytes(manifestFile), "Failed at stage " + stage);
            assertFalse(Files.exists(Path.of(manifestFile.toString() + ".tx.journal")));
        }
    }

    @Test
    void faultDuringCropReplacementRollsBackCsvsAndRestoresOrDeletesCrop() throws IOException {
        byte[] origFeaturesBytes = Files.readAllBytes(featuresFile);
        byte[] origManifestBytes = Files.readAllBytes(manifestFile);
        FeatureVector fv1 = new FeatureVector(new double[]{88.0, 0.44, 0.51, 0.46, 0.40, 110.0, 0.52, 0.35});
        FeatureVector fv2 = new FeatureVector(new double[]{62.0, 0.07, 0.85, 0.16, 0.03, 19.0, 0.00, 0.02});
        ParkingSlot slot = new ParkingSlot(6, 10, 10, 80, 100);
        BufferedImage redCrop  = createTestCrop(Color.RED);
        BufferedImage blueCrop = createTestCrop(Color.BLUE);

        String expectedSampleId = DatasetManifest.computeSampleId(
            "crop_tx_lot.png", "LOT_CROP_TX", 6, 10, 10, 80, 100, FeatureVector.SCHEMA_VERSION);
        Path cropFile = tempDir.resolve("crops").resolve(expectedSampleId + ".png");

        // Case A: Fault during final crop replacement when NO prior crop existed
        DatasetManifest.setFaultInjectionStageForTesting(DatasetManifest.FaultInjectionStage.DURING_CROP_REPLACEMENT);
        assertThrows(IllegalStateException.class, () ->
            DatasetManifest.recordUserLabeledSample(
                featuresFile.toString(), manifestFile.toString(),
                "crop_tx_lot.png", "LOT_CROP_TX", slot, fv1, SlotStatus.OCCUPIED, "REV_1", redCrop, true
            )
        );
        assertArrayEquals(origFeaturesBytes, Files.readAllBytes(featuresFile));
        assertArrayEquals(origManifestBytes, Files.readAllBytes(manifestFile));
        assertFalse(Files.exists(cropFile), "No orphan crop may remain when initial crop move fails");

        // Case B: Succeed once so cropFile exists with redCrop
        DatasetManifest.resetTestingHooks();
        DatasetManifest.recordUserLabeledSample(
            featuresFile.toString(), manifestFile.toString(),
            "crop_tx_lot.png", "LOT_CROP_TX", slot, fv1, SlotStatus.OCCUPIED, "REV_1", redCrop, true
        );
        assertTrue(Files.exists(cropFile));
        byte[] committedFeaturesAfterFirst = Files.readAllBytes(featuresFile);
        byte[] committedManifestAfterFirst = Files.readAllBytes(manifestFile);
        byte[] committedRedCropBytes       = Files.readAllBytes(cropFile);

        // Case C: Fault during crop replacement when overwriting existing redCrop with blueCrop
        DatasetManifest.setFaultInjectionStageForTesting(DatasetManifest.FaultInjectionStage.DURING_CROP_REPLACEMENT);
        assertThrows(IllegalStateException.class, () ->
            DatasetManifest.recordUserLabeledSample(
                featuresFile.toString(), manifestFile.toString(),
                "crop_tx_lot.png", "LOT_CROP_TX", slot, fv2, SlotStatus.AVAILABLE, "REV_1", blueCrop, true
            )
        );
        assertArrayEquals(committedFeaturesAfterFirst, Files.readAllBytes(featuresFile));
        assertArrayEquals(committedManifestAfterFirst, Files.readAllBytes(manifestFile));
        assertArrayEquals(committedRedCropBytes, Files.readAllBytes(cropFile),
            "Previous crop bytes must be restored when overwriting crop fails mid-transaction");
    }

    @Test
    void crashRecoveryForCommittedStateCompletesCleanupWithoutRollingBackNewData() throws IOException {
        FeatureVector fv = new FeatureVector(new double[]{88.0, 0.44, 0.51, 0.46, 0.40, 110.0, 0.52, 0.35});
        ParkingSlot slot = new ParkingSlot(7, 10, 10, 80, 100);
        BufferedImage greenCrop = createTestCrop(Color.GREEN);

        DatasetManifest.ManifestRecord rec = DatasetManifest.recordUserLabeledSample(
            featuresFile.toString(), manifestFile.toString(),
            "committed_recovery.png", "LOT_COMMIT_REC", slot, fv, SlotStatus.OCCUPIED, "REV_1", greenCrop, true
        );
        byte[] committedFeatures = Files.readAllBytes(featuresFile);
        byte[] committedManifest = Files.readAllBytes(manifestFile);

        // Simulate a leftover COMMITTED journal + stale .bak files from a crash right before final cleanup
        Path journal = Path.of(manifestFile.toString() + ".tx.journal");
        Path fBak    = Path.of(featuresFile.toString() + ".bak");
        Path mBak    = Path.of(manifestFile.toString() + ".bak");
        Files.writeString(fBak, "OLD_STALE_BACKUP", StandardCharsets.UTF_8);
        Files.writeString(mBak, "OLD_STALE_BACKUP", StandardCharsets.UTF_8);
        String journalContent = String.join("\n",
            "state=COMMITTED",
            "fPath=" + featuresFile.toAbsolutePath(),
            "fTmp=" + featuresFile.toAbsolutePath() + ".tmp",
            "fBak=" + fBak.toAbsolutePath(),
            "fExisted=true",
            "mPath=" + manifestFile.toAbsolutePath(),
            "mTmp=" + manifestFile.toAbsolutePath() + ".tmp",
            "mBak=" + mBak.toAbsolutePath(),
            "mExisted=true",
            "cropDst=" + tempDir.resolve("crops").resolve(rec.sampleId() + ".png").toAbsolutePath(),
            "cropTmp=",
            "cropBak=",
            "cropExisted=false"
        );
        Files.writeString(journal, journalContent, StandardCharsets.UTF_8);

        // Calling loadOrSync must see state=COMMITTED, keep the 49-row committed files, and clean up .bak and journal!
        List<DatasetManifest.ManifestRecord> loaded = DatasetManifest.loadOrSync(
            featuresFile.toString(), manifestFile.toString());
        assertEquals(49, loaded.size());
        assertArrayEquals(committedFeatures, Files.readAllBytes(featuresFile));
        assertArrayEquals(committedManifest, Files.readAllBytes(manifestFile));
        assertFalse(Files.exists(journal));
        assertFalse(Files.exists(fBak));
        assertFalse(Files.exists(mBak));
    }

    @Test
    void rollbackFailureRetainsJournalAndBackupsAndReportsRecoveryRequired() {
        DatasetManifest.setFaultInjectionStageForTesting(DatasetManifest.FaultInjectionStage.AFTER_FEATURES_REPLACEMENT);
        DatasetManifest.setSimulateRollbackFailureForTesting(true);

        FeatureVector fv = new FeatureVector(new double[]{88.0, 0.44, 0.51, 0.46, 0.40, 110.0, 0.52, 0.35});
        ParkingSlot slot = new ParkingSlot(8, 10, 10, 80, 100);

        IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
            DatasetManifest.recordUserLabeledSample(
                featuresFile.toString(), manifestFile.toString(),
                "rb_fail.png", "LOT_RB_FAIL", slot, fv, SlotStatus.OCCUPIED, "REV_1", null, false
            )
        );
        assertTrue(ex.getMessage().contains("RECOVERY_REQUIRED"));
        assertTrue(Files.exists(Path.of(manifestFile.toString() + ".tx.journal")),
            "Journal must be retained when rollback itself fails");
        assertTrue(Files.exists(Path.of(featuresFile.toString() + ".bak")),
            "Features backup must be retained when rollback itself fails");
    }

    // --- Phase 5 (PNC-REM-06): Validation & Test group protection and explicit correction flows ---

    @Test
    void validationAndTestGroupsAreProtectedFromTrainingAndSupportExplicitCorrectionFlows() {
        FeatureVector fv = new FeatureVector(new double[]{85.0, 0.40, 0.50, 0.45, 0.36, 100.0, 0.45, 0.31});
        ParkingSlot slot = new ParkingSlot(2, 10, 10, 80, 100);

        // 1. Adding a sample from an existing TEST group is rejected
        assertThrows(IllegalStateException.class, () ->
            DatasetManifest.recordUserLabeledSample(
                featuresFile.toString(), manifestFile.toString(),
                "frame1.png", "LOT_TEST_OCC_HOLD", slot, fv, SlotStatus.OCCUPIED, "REV_1", null, false
            )
        );

        // 2. Adding a sample from an existing VALIDATION group to training is rejected
        assertThrows(IllegalStateException.class, () ->
            DatasetManifest.recordUserLabeledSample(
                featuresFile.toString(), manifestFile.toString(),
                "frame2.png", "LOT_VAL_OCC_CALIB", slot, fv, SlotStatus.OCCUPIED, "REV_1", null, false
            )
        );

        // 3. Explicit validation correction updates SMP-008 in place while keeping partition == VALIDATION
        DatasetManifest.ManifestRecord valCorrected = DatasetManifest.correctValidationSampleLabel(
            featuresFile.toString(), manifestFile.toString(), "SMP-008", SlotStatus.AVAILABLE, "REV_QA"
        );
        assertEquals("VALIDATION", valCorrected.partition());
        assertEquals(SlotStatus.AVAILABLE, valCorrected.label());
        assertEquals("REV_QA", valCorrected.reviewerId());

        // 4. Explicit test correction updates SMP-006 in place, bumps splitId, and returns warning
        DatasetManifest.TestCorrectionResult testCorr = DatasetManifest.correctTestSampleLabelWithVersionBump(
            featuresFile.toString(), manifestFile.toString(), "SMP-006", SlotStatus.AVAILABLE, "REV_CHIEF"
        );
        assertEquals("TEST", testCorr.updatedRecord().partition());
        assertTrue(testCorr.newSplitId().startsWith("SPLIT-CORRECTED-"));
        assertTrue(testCorr.warningMessage().contains("invalidates"));
    }

    @Test
    void loadOrSyncIsIdempotentAndSchemaHeaderWidthsMatchNineAndTwentyTwoColumns() throws IOException {
        List<DatasetManifest.ManifestRecord> first = DatasetManifest.loadOrSync(
            featuresFile.toString(), manifestFile.toString());
        byte[] afterFirstBytes = Files.readAllBytes(manifestFile);

        List<DatasetManifest.ManifestRecord> second = DatasetManifest.loadOrSync(
            featuresFile.toString(), manifestFile.toString());
        byte[] afterSecondBytes = Files.readAllBytes(manifestFile);

        assertEquals(first, second);
        assertArrayEquals(afterFirstBytes, afterSecondBytes);
        assertEquals(9, FeatureVector.csvHeader().split(",", -1).length);
        assertEquals(22, DatasetManifest.HEADER.split(",", -1).length);
    }
}
