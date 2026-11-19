package com.parknchillai.ai;

import com.parknchillai.features.FeatureVector;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression suite for Phase 3 (PNC-REM-04) and Phase 2 (PNC-FIX-03) evidence-based
 * provenance validation, explicit reviewer attribution, ROI bounds verification inside
 * decoded source images, mixed-group exclusion, and single-class metric unavailability.
 */
class DatasetValidatorTest {

    private Path tempDir;

    @BeforeEach
    void setUp() throws IOException {
        tempDir = Files.createTempDirectory("pnc_validator_rem_test_");
    }

    @AfterEach
    void tearDown() throws IOException {
        if (tempDir != null && Files.exists(tempDir)) {
            Files.walk(tempDir)
                .sorted((a, b) -> b.compareTo(a))
                .forEach(p -> {
                    try { Files.deleteIfExists(p); } catch (IOException ignored) {}
                });
        }
    }

    private Path createRealCameraTestImage(String name, Color tint) throws IOException {
        BufferedImage img = new BufferedImage(120, 120, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(tint);
        g.fillRect(0, 0, 120, 120);
        g.dispose();
        Path p = tempDir.resolve(name);
        ImageIO.write(img, "png", p.toFile());
        return p;
    }

    @Test
    void createVerifiedRealCameraRecordRejectsBlankReviewerMissingCorruptOrOutOfBoundsRoiImage() throws IOException {
        Path validImg = createRealCameraTestImage("valid.png", new Color(70, 80, 90));
        Path corruptImg = tempDir.resolve("corrupt.png");
        Files.writeString(corruptImg, "NOT_A_VALID_PNG_FILE", StandardCharsets.UTF_8);
        FeatureVector fv = new FeatureVector(new double[]{80.0, 0.35, 0.50, 0.40, 0.32, 90.0, 0.40, 0.28});

        // 1. Blank reviewer ID is rejected (never substituted with QA_REVIEWER)
        assertThrows(IllegalArgumentException.class, () ->
            DatasetManifest.ManifestRecord.createVerifiedRealCameraRecord(
                "R-1", validImg.toString(), "REAL_GRP_1", 1, 10, 10, 80, 100,
                fv, SlotStatus.OCCUPIED, "   ", "TRAIN", "2026-10-21T00:00:00Z"
            )
        );

        // 2. Missing image is rejected
        assertThrows(IllegalArgumentException.class, () ->
            DatasetManifest.ManifestRecord.createVerifiedRealCameraRecord(
                "R-2", tempDir.resolve("missing.png").toString(), "REAL_GRP_1", 1, 10, 10, 80, 100,
                fv, SlotStatus.OCCUPIED, "AUDITOR_1", "TRAIN", "2026-10-21T00:00:00Z"
            )
        );

        // 3. Corrupt/non-image file is rejected
        assertThrows(IllegalArgumentException.class, () ->
            DatasetManifest.ManifestRecord.createVerifiedRealCameraRecord(
                "R-3", corruptImg.toString(), "REAL_GRP_1", 1, 10, 10, 80, 100,
                fv, SlotStatus.OCCUPIED, "AUDITOR_1", "TRAIN", "2026-10-21T00:00:00Z"
            )
        );

        // 4. ROI extending outside the 120x120 decoded image is rejected
        assertThrows(IllegalArgumentException.class, () ->
            DatasetManifest.ManifestRecord.createVerifiedRealCameraRecord(
                "R-4", validImg.toString(), "REAL_GRP_1", 1, 60, 60, 80, 100,
                fv, SlotStatus.OCCUPIED, "AUDITOR_1", "TRAIN", "2026-10-21T00:00:00Z"
            )
        );

        // 5. User-declared real camera starts as USER_DECLARED_REAL_CAMERA and promotes to VERIFIED_REAL_CAMERA
        DatasetManifest.ManifestRecord declared = DatasetManifest.ManifestRecord.createUserDeclaredRealCameraRecord(
            "R-5", validImg.toString(), "REAL_GRP_1", 1, 10, 10, 80, 100,
            fv, SlotStatus.OCCUPIED, "AUDITOR_1", "TRAIN", "2026-10-21T00:00:00Z"
        );
        assertEquals("USER_DECLARED_REAL_CAMERA", declared.provenance());
        assertEquals("AUDITOR_1", declared.reviewerId());

        DatasetManifest.ManifestRecord promoted = DatasetManifest.promoteToVerifiedRealCamera(declared);
        assertEquals("VERIFIED_REAL_CAMERA", promoted.provenance());
        assertEquals("AUDITOR_1", promoted.reviewerId());
    }

    @Test
    void tenRealLabeledRecordsFromOneGroupDoNotSatisfyThreeGroupCriterion() throws IOException {
        Path imgPath = createRealCameraTestImage("lot1_cam.png", new Color(70, 80, 90));
        double[][] feats = new double[10][8];
        SlotStatus[] labels = new SlotStatus[10];
        List<DatasetManifest.ManifestRecord> manifest = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            feats[i] = new double[]{80.0 + i, 0.35, 0.50, 0.40, 0.32, 90.0, 0.40, 0.28};
            labels[i] = (i % 2 == 0) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
            manifest.add(DatasetManifest.ManifestRecord.createVerifiedRealCameraRecord(
                "REAL-" + (i + 1), imgPath.toString(), "SINGLE_REAL_GROUP",
                i + 1, 10, 10, 80, 100, new FeatureVector(feats[i]), labels[i],
                "AUDITOR_1", "TRAIN", "2026-10-21T00:00:00Z"
            ));
        }

        DatasetLoader.Dataset ds = new DatasetLoader.Dataset(feats, labels);
        DatasetValidator.ValidationSummary summary = DatasetValidator.validate(ds, manifest);

        assertEquals(10, summary.realCameraVerifiedSamples());
        assertEquals(1, summary.verifiedRealCameraGroups());
        assertFalse(summary.realImageSamplesPresent());
        assertFalse(summary.realWorldGeneralizationEstablished());
    }

    @Test
    void threeDistinctRealGroupsSatisfyInventoryOnlyWhileMixedGroupsAreExcluded() throws IOException {
        Path img1 = createRealCameraTestImage("cam_lot_1.png", new Color(60, 70, 80));
        Path img2 = createRealCameraTestImage("cam_lot_2.png", new Color(80, 90, 100));
        Path img3 = createRealCameraTestImage("cam_lot_3.png", new Color(100, 110, 120));
        Path[] imgs = {img1, img2, img3};
        String[] grps = {"REAL_LOT_1", "REAL_LOT_2", "REAL_LOT_3"};
        String[] parts = {"TRAIN", "VALIDATION", "TEST"};

        double[][] feats = new double[12][8];
        SlotStatus[] labels = new SlotStatus[12];
        List<DatasetManifest.ManifestRecord> manifest = new ArrayList<>();

        for (int i = 0; i < 12; i++) {
            feats[i] = new double[]{75.0 + i, 0.30 + (i * 0.01), 0.50, 0.38, 0.30, 85.0, 0.35, 0.25};
            labels[i] = (i % 2 == 0) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
            int gIdx = i % 3;
            manifest.add(DatasetManifest.ManifestRecord.createVerifiedRealCameraRecord(
                "REAL-" + (i + 1), imgs[gIdx].toString(), grps[gIdx],
                i + 1, 10, 10, 80, 100, new FeatureVector(feats[i]), labels[i],
                "AUDITOR_1", parts[gIdx], "2026-10-21T00:00:00Z"
            ));
        }

        DatasetLoader.Dataset ds = new DatasetLoader.Dataset(feats, labels);
        DatasetValidator.ValidationSummary summary = DatasetValidator.validate(ds, manifest);

        assertEquals(12, summary.realCameraVerifiedSamples());
        assertEquals(3, summary.verifiedRealCameraGroups());
        assertTrue(summary.realImageSamplesPresent());
        assertFalse(summary.realWorldGeneralizationEstablished(),
            "File hash and inventory alone must NEVER set realWorldGeneralizationEstablished");

        // Now contaminate REAL_LOT_3 by mixing a synthetic record into the same captureGroup REAL_LOT_3
        List<DatasetManifest.ManifestRecord> mixedManifest = new ArrayList<>(manifest);
        DatasetManifest.ManifestRecord last = mixedManifest.get(11);
        mixedManifest.set(11, new DatasetManifest.ManifestRecord(
            last.sampleId(), "procedural://scene_mixed", "REAL_LOT_3",
            last.slotId(), last.x(), last.y(), last.width(), last.height(),
            last.features(), last.label(), last.schemaVersion(),
            "UNVERIFIED_SYNTHETIC", "PROCEDURAL_SYNTHETIC", "TEST", last.timestamp()
        ));

        DatasetValidator.ValidationSummary mixedSummary = DatasetValidator.validate(ds, mixedManifest);
        assertEquals(2, mixedSummary.verifiedRealCameraGroups(),
            "Mixed capture group REAL_LOT_3 must be excluded from verifiedRealCameraGroups");
        assertEquals(8, mixedSummary.realCameraVerifiedSamples(),
            "Samples in mixed capture group REAL_LOT_3 must be excluded from realCameraVerifiedSamples");
        assertFalse(mixedSummary.realImageSamplesPresent());
    }

    @Test
    void missingSourceFilesAndTamperedHashesAreRejectedFromVerifiedRealSubset() throws IOException {
        Path validImg = createRealCameraTestImage("valid_cam.png", new Color(70, 75, 80));
        Path tamperedImg = createRealCameraTestImage("tampered_cam.png", new Color(90, 95, 100));

        double[][] feats = {
            {80.0, 0.40, 0.50, 0.45, 0.35, 100.0, 0.45, 0.30},
            {82.0, 0.42, 0.48, 0.46, 0.36, 102.0, 0.46, 0.31},
            {60.0, 0.06, 0.85, 0.16, 0.02, 20.0, 0.00, 0.02},
            {62.0, 0.07, 0.84, 0.17, 0.03, 22.0, 0.00, 0.02}
        };
        SlotStatus[] labels = {
            SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE
        };

        DatasetManifest.ManifestRecord validRec = DatasetManifest.ManifestRecord.createVerifiedRealCameraRecord(
            "R-1", validImg.toString(), "GRP_REAL_1", 1, 10, 10, 80, 100,
            new FeatureVector(feats[0]), labels[0], "QA_REVIEWER_1", "TRAIN", "2026-10-21T00:00:00Z"
        );
        DatasetManifest.ManifestRecord tamperedRec = DatasetManifest.ManifestRecord.createVerifiedRealCameraRecord(
            "R-2", tamperedImg.toString(), "GRP_REAL_2", 2, 10, 10, 80, 100,
            new FeatureVector(feats[1]), labels[1], "QA_REVIEWER_1", "VALIDATION", "2026-10-21T00:00:00Z"
        );
        // Modify tamperedImg bytes after hash was recorded
        createRealCameraTestImage("tampered_cam.png", new Color(200, 10, 10));

        DatasetManifest.ManifestRecord missingFileRec = new DatasetManifest.ManifestRecord(
            "R-3", tempDir.resolve("non_existent.png").toString(), "GRP_REAL_3",
            3, 10, 10, 80, 100, new FeatureVector(feats[2]), labels[2],
            FeatureVector.SCHEMA_VERSION,
            "HUMAN_VERIFIED[sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef;by=QA_1]",
            "VERIFIED_REAL_CAMERA", "TEST", "2026-10-21T00:00:00Z"
        );
        DatasetManifest.ManifestRecord synthRec = new DatasetManifest.ManifestRecord(
            "R-4", "procedural://scene_4", "GRP_SYNTH_4",
            4, 10, 10, 80, 100, new FeatureVector(feats[3]), labels[3],
            FeatureVector.SCHEMA_VERSION, "UNVERIFIED_SYNTHETIC",
            "PROCEDURAL_SYNTHETIC", "TRAIN", "2026-10-21T00:00:00Z"
        );

        DatasetLoader.Dataset ds = new DatasetLoader.Dataset(feats, labels);
        DatasetValidator.ValidationSummary summary = DatasetValidator.validate(
            ds, List.of(validRec, tamperedRec, missingFileRec, synthRec)
        );

        assertEquals(1, summary.realCameraVerifiedSamples());
        assertEquals(1, summary.verifiedRealCameraGroups());
        assertFalse(summary.realImageSamplesPresent());
        assertFalse(summary.realWorldGeneralizationEstablished());
    }

    @Test
    void bundledDatasetReportsFiveDemoFortyThreeSyntheticAndZeroVerifiedRealCamera() throws IOException {
        DatasetLoader.Dataset ds = DatasetLoader.load("data/labels/features.csv");
        List<DatasetManifest.ManifestRecord> man = DatasetManifest.loadManifest("data/labels/manifest.csv");
        DatasetValidator.ValidationSummary summary = DatasetValidator.validate(ds, man);

        assertTrue(summary.validForTraining());
        assertEquals(48, summary.totalSamples());
        assertEquals(5, summary.demoExtractedSamples());
        assertEquals(43, summary.proceduralSyntheticSamples());
        assertEquals(0, summary.legacyImportedSamples());
        assertEquals(0, summary.userDeclaredSamples());
        assertEquals(0, summary.realCameraVerifiedSamples());
        assertEquals(0, summary.verifiedRealCameraGroups());
        assertFalse(summary.realImageSamplesPresent());
        assertTrue(summary.groupDisjointEvaluationPossible());
        assertFalse(summary.realWorldGeneralizationEstablished());
        assertTrue(summary.formatProvenanceSummary().contains("NOT ESTABLISHED"));
    }

    @Test
    void metricsAreMarkedUnavailableIfEvaluationSplitLacksOneOfRequiredClasses() {
        ModelEvaluator.EvaluationReport singleClassReport = ModelEvaluator.buildReportFromCounts(
            0, 0, 0, 5, ModelEvaluator.EvaluationScope.VALIDATION_PARTITION, "Single-class AVAILABLE split"
        );
        assertFalse(singleClassReport.metricsAvailable);
        assertTrue(Double.isNaN(singleClassReport.precision));
        assertTrue(Double.isNaN(singleClassReport.recall));
        assertTrue(Double.isNaN(singleClassReport.f1));
        assertEquals("UNAVAILABLE", singleClassReport.formatMetric(singleClassReport.f1));
        assertTrue(singleClassReport.toString().contains("UNAVAILABLE"));
    }
}
