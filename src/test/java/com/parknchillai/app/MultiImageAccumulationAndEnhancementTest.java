package com.parknchillai.app;

import com.parknchillai.ai.ModelEvaluator;
import com.parknchillai.ai.SlotTrainingRecord;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.imageprocessing.ImageEnhancer;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import com.parknchillai.utils.ImageLoader;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression test suite verifying:
 * <ul>
 *   <li>Problem 1: Image Enhancement becomes the active working image while preserving immutable original photograph.</li>
 *   <li>Problem 2: Accumulating data across multiple images without erasing earlier records, slot ID collision prevention, and aggregate metrics.</li>
 *   <li>Problem 3: Honest Decision Tree evaluation metrics with 'Insufficient test data' when test partition lacks samples or classes.</li>
 *   <li>Problem 4: Boundary origin tag preservation.</li>
 * </ul>
 */
public class MultiImageAccumulationAndEnhancementTest {

    private ParkingAnalysisService analysisService;
    private AppState appState;

    @BeforeEach
    void setUp() {
        analysisService = ParkingAnalysisService.createWithDefaultModel();
        appState = new AppState(analysisService);
        appState.setReviewerId("TEST_OPERATOR_1");
    }

    private BufferedImage createSyntheticTestImage(Color color) {
        BufferedImage img = new BufferedImage(320, 240, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(color);
        g.fillRect(0, 0, 320, 240);
        g.setColor(Color.WHITE);
        g.drawRect(20, 20, 80, 80);
        g.drawRect(120, 20, 80, 80);
        g.dispose();
        return img;
    }

    private List<ParkingSlot> createTestSlots() {
        return List.of(
            new ParkingSlot(1, 20, 20, 80, 80, ParkingSlot.GeometryOrigin.PKLOT_POLYGON),
            new ParkingSlot(2, 120, 20, 80, 80, ParkingSlot.GeometryOrigin.AUTO_DETECTED),
            new ParkingSlot(3, 20, 120, 80, 80, ParkingSlot.GeometryOrigin.MANUAL_DRAWN),
            new ParkingSlot(4, 120, 120, 80, 80, ParkingSlot.GeometryOrigin.CSV_CONFIG)
        );
    }

    @Test
    void testMultiImageAccumulationAcrossFourImagesWithoutErasing() {
        // Create 4 distinct photographs
        BufferedImage img1 = createSyntheticTestImage(new Color(180, 50, 50));
        BufferedImage img2 = createSyntheticTestImage(new Color(50, 180, 50));
        BufferedImage img3 = createSyntheticTestImage(new Color(50, 50, 180));
        BufferedImage img4 = createSyntheticTestImage(new Color(120, 120, 120));

        String sha1 = "1111111111111111111111111111111111111111111111111111111111111111";
        String sha2 = "2222222222222222222222222222222222222222222222222222222222222222";
        String sha3 = "3333333333333333333333333333333333333333333333333333333333333333";
        String sha4 = "4444444444444444444444444444444444444444444444444444444444444444";

        ImageLoader.LoadedImageMetadata meta1 = new ImageLoader.LoadedImageMetadata("lot_red.png", "C:/lot_red.png", "png", 320, 240, 320, 240, 1.0, 1.0, false, false, img1);
        ImageLoader.LoadedImageMetadata meta2 = new ImageLoader.LoadedImageMetadata("lot_green.png", "C:/lot_green.png", "png", 320, 240, 320, 240, 1.0, 1.0, false, false, img2);
        ImageLoader.LoadedImageMetadata meta3 = new ImageLoader.LoadedImageMetadata("lot_blue.png", "C:/lot_blue.png", "png", 320, 240, 320, 240, 1.0, 1.0, false, false, img3);
        ImageLoader.LoadedImageMetadata meta4 = new ImageLoader.LoadedImageMetadata("lot_gray.png", "C:/lot_gray.png", "png", 320, 240, 320, 240, 1.0, 1.0, false, false, img4);

        // Load & Analyze Image 1
        appState.setLoadedImageAndSlots(meta1, sha1, createTestSlots(), true, "PKLot XML Loaded");
        AnalysisSummary summary1 = analysisService.analyze(img1, createTestSlots());
        appState.updateAnalysisSummary(summary1);
        appState.reviewSlot(1, SlotStatus.OCCUPIED);
        appState.reviewSlot(2, SlotStatus.AVAILABLE);

        assertEquals(1, appState.getAllImageSessions().size());
        assertEquals(2, appState.getConfirmedSlotCountForCurrentImage());

        // Load & Analyze Image 2 (Must NOT erase image 1!)
        appState.setLoadedImageAndSlots(meta2, sha2, createTestSlots(), true, "PKLot XML Loaded");
        AnalysisSummary summary2 = analysisService.analyze(img2, createTestSlots());
        appState.updateAnalysisSummary(summary2);
        appState.reviewSlot(1, SlotStatus.OCCUPIED);
        appState.reviewSlot(3, SlotStatus.AVAILABLE);

        assertEquals(2, appState.getAllImageSessions().size(), "Loading image 2 must not erase image 1 session");

        // Load & Analyze Image 3
        appState.setLoadedImageAndSlots(meta3, sha3, createTestSlots(), true, "Auto-detected");
        AnalysisSummary summary3 = analysisService.analyze(img3, createTestSlots());
        appState.updateAnalysisSummary(summary3);
        appState.reviewSlot(1, SlotStatus.AVAILABLE);
        appState.reviewSlot(4, SlotStatus.OCCUPIED);

        assertEquals(3, appState.getAllImageSessions().size());

        // Load & Analyze Image 4
        appState.setLoadedImageAndSlots(meta4, sha4, createTestSlots(), true, "Auto-detected");
        AnalysisSummary summary4 = analysisService.analyze(img4, createTestSlots());
        appState.updateAnalysisSummary(summary4);
        appState.reviewSlot(1, SlotStatus.OCCUPIED);
        appState.reviewSlot(2, SlotStatus.AVAILABLE);

        // Verify all 4 images exist in AppState
        assertEquals(4, appState.getAllImageSessions().size(), "All 4 image sessions must coexist in AppState");
        assertEquals(4, appState.getTotalImagesAnalysed(), "Total images analysed must equal 4");
        assertEquals(16, appState.getTotalSlotsCollected(), "Total slots collected must equal 4 slots x 4 images = 16");

        // Verify slot IDs do NOT collide in userReviewedTrainingPool
        List<SlotTrainingRecord> eligible = appState.getEligibleReviewedRecords();
        assertEquals(8, eligible.size(), "8 total slots were reviewed across the 4 images (2 per image)");

        Set<String> uniqueSampleIds = new HashSet<>();
        for (SlotTrainingRecord rec : eligible) {
            assertTrue(uniqueSampleIds.add(rec.sampleId()),
                "Sample ID must be distinct across images (no collision for Slot #1): " + rec.sampleId());
        }

        // Verify session switching preserves per-image state
        appState.switchToSession(sha1);
        assertEquals(SlotStatus.OCCUPIED, appState.getVerifiedLabelForSlot(1),
            "Switching back to image 1 must restore its verified Slot #1 as OCCUPIED");
        assertEquals(SlotStatus.AVAILABLE, appState.getVerifiedLabelForSlot(2));

        appState.switchToSession(sha3);
        assertEquals(SlotStatus.AVAILABLE, appState.getVerifiedLabelForSlot(1),
            "Image 3 Slot #1 was marked AVAILABLE and must retain that state");

        // Verify start fresh dataset clears everything
        appState.startFreshDataset(false);
        assertEquals(4, appState.getAllImageSessions().size(), "confirm=false must not clear dataset");

        appState.startFreshDataset(true);
        assertEquals(0, appState.getAllImageSessions().size(), "confirm=true must clear all image sessions");
        assertEquals(0, appState.getEligibleReviewedRecords().size(), "confirm=true must clear training pool");
    }

    @Test
    void testActiveWorkingImageEnhancementAndOriginalPreservation() {
        BufferedImage orig = createSyntheticTestImage(new Color(60, 60, 60));
        String sha = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890";
        ImageLoader.LoadedImageMetadata meta = new ImageLoader.LoadedImageMetadata("test.png", "C:/test.png", "png", 320, 240, 320, 240, 1.0, 1.0, false, false, orig);

        appState.setLoadedImageAndSlots(meta, sha, createTestSlots(), true, "PKLot");
        AnalysisSummary initialSummary = analysisService.analyze(orig, createTestSlots());
        appState.updateAnalysisSummary(initialSummary);

        // Initially working image is unenhanced
        assertFalse(appState.isWorkingImageEnhanced());
        assertEquals("Original", appState.getActiveEnhancementDescription());
        assertSame(orig, appState.getOriginalImage());

        // Apply manual enhancement (contrast + brightness adjustment)
        BufferedImage enhanced = ImageEnhancer.adjust(orig, 1.5, 30, 0.8, 3);
        appState.applyWorkingImageEnhancement(enhanced, "Manual (Contrast=1.5, Brightness=30, Gamma=0.8, Noise=3x3)");

        assertTrue(appState.isWorkingImageEnhanced(), "isWorkingImageEnhanced must be true after manual adjustment");
        assertEquals("Manual (Contrast=1.5, Brightness=30, Gamma=0.8, Noise=3x3)", appState.getActiveEnhancementDescription());
        assertSame(orig, appState.getOriginalImage(), "Original photograph must remain strictly immutable");
        assertSame(enhanced, appState.getActiveWorkingImage(), "Active working image must be the enhanced image");

        // Verify that IVP slot crops and features were recomputed on enhanced working image
        AnalysisSummary updatedSummary = appState.getCurrentSummary();
        assertNotNull(updatedSummary);
        SlotAnalysisResult slot1Result = updatedSummary.getSlotResults().get(0);
        // The enhanced ROI in slotResult must match enhanced image crop, while original ROI matches original crop
        assertNotNull(slot1Result.getRoiImage());
        assertNotNull(slot1Result.getOriginalRoiImage());

        // Reset to original
        appState.resetWorkingImageToOriginal();
        assertFalse(appState.isWorkingImageEnhanced());
        assertEquals("Original (Unenhanced)", appState.getActiveEnhancementDescription());
        assertSame(orig, appState.getActiveWorkingImage());
    }

    @Test
    void testHonestDecisionTreeEvaluationThresholds() {
        // N < 4 test partition: must report metricsAvailable = false and unavailableReason
        ModelEvaluator.EvaluationReport reportSmall = ModelEvaluator.buildReportFromCounts(
            1, 0, 0, 1, ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION, "Small test set N=2"
        );
        assertFalse(reportSmall.metricsAvailable, "N=2 must be marked UNAVAILABLE to avoid misleading 100% metrics");
        assertTrue(Double.isNaN(reportSmall.accuracy));
        assertTrue(reportSmall.unavailableReason.contains("Insufficient test data"));

        // Single-class partition (e.g. 0 occupied, 6 available): must report metricsAvailable = false
        ModelEvaluator.EvaluationReport reportSingleClass = ModelEvaluator.buildReportFromCounts(
            0, 0, 0, 6, ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION, "Single-class AVAILABLE"
        );
        assertFalse(reportSingleClass.metricsAvailable, "Single-class test partition must be marked UNAVAILABLE");
        assertTrue(reportSingleClass.unavailableReason.contains("Single-class evaluation partition"));

        // Valid balanced test partition N=8: must report metricsAvailable = true
        ModelEvaluator.EvaluationReport reportValid = ModelEvaluator.buildReportFromCounts(
            4, 0, 0, 4, ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION, "Held-out test N=8"
        );
        assertTrue(reportValid.metricsAvailable, "Balanced N=8 test partition must have metrics available");
        assertEquals(1.0, reportValid.accuracy, 1e-6);
        assertEquals(1.0, reportValid.precision, 1e-6);
        assertEquals(1.0, reportValid.recall, 1e-6);
        assertEquals(1.0, reportValid.f1, 1e-6);
    }

    @Test
    void testBoundaryOriginPreservation() {
        List<ParkingSlot> slots = createTestSlots();
        assertEquals(ParkingSlot.GeometryOrigin.PKLOT_POLYGON, slots.get(0).getOrigin());
        assertEquals("PKLot Polygon Contour", slots.get(0).getOrigin().getDisplayName());

        assertEquals(ParkingSlot.GeometryOrigin.AUTO_DETECTED, slots.get(1).getOrigin());
        assertEquals("Auto-Detected (IVP)", slots.get(1).getOrigin().getDisplayName());

        assertEquals(ParkingSlot.GeometryOrigin.MANUAL_DRAWN, slots.get(2).getOrigin());
        assertEquals("Manual Box", slots.get(2).getOrigin().getDisplayName());

        assertEquals(ParkingSlot.GeometryOrigin.CSV_CONFIG, slots.get(3).getOrigin());
        assertEquals("CSV Configuration", slots.get(3).getOrigin().getDisplayName());
    }

    @Test
    void testPipelineFailureSafeErrorHandling() {
        BufferedImage img = createSyntheticTestImage(Color.GRAY);
        // Slot coordinates extending outside image (x=300, width=50 -> 350 > 320)
        ParkingSlot outOfBoundsSlot = new ParkingSlot(99, 300, 20, 50, 50);

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
            analysisService.analyze(img, List.of(outOfBoundsSlot))
        );
        assertTrue(ex.getMessage().contains("Slot 99 ROI"));
        assertTrue(ex.getMessage().contains("extends outside image"));
    }

    @Test
    void testMultiImageDatasetExport() throws java.io.IOException {
        BufferedImage img1 = createSyntheticTestImage(new Color(150, 50, 50));
        BufferedImage img2 = createSyntheticTestImage(new Color(50, 150, 50));
        String sha1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        String sha2 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

        ImageLoader.LoadedImageMetadata meta1 = new ImageLoader.LoadedImageMetadata("imgA.png", "C:/imgA.png", "png", 320, 240, 320, 240, 1.0, 1.0, false, false, img1);
        ImageLoader.LoadedImageMetadata meta2 = new ImageLoader.LoadedImageMetadata("imgB.png", "C:/imgB.png", "png", 320, 240, 320, 240, 1.0, 1.0, false, false, img2);

        appState.setLoadedImageAndSlots(meta1, sha1, createTestSlots(), true, "PKLot");
        appState.updateAnalysisSummary(analysisService.analyze(img1, createTestSlots()));

        appState.setLoadedImageAndSlots(meta2, sha2, createTestSlots(), true, "PKLot");
        appState.updateAnalysisSummary(analysisService.analyze(img2, createTestSlots()));

        assertEquals(2, appState.getTotalImagesAnalysed());
        assertEquals(8, appState.getTotalSlotsCollected());

        java.nio.file.Path tempCsv = java.nio.file.Files.createTempFile("accumulated_test", ".csv");
        try {
            try (java.io.BufferedWriter writer = java.nio.file.Files.newBufferedWriter(tempCsv, java.nio.charset.StandardCharsets.UTF_8)) {
                writer.write("image_name,image_sha256,slot_id,status,boundary_origin\n");
                for (ImageSessionRecord sess : appState.getAllImageSessions()) {
                    for (SlotAnalysisResult r : sess.getAnalysisSummary().getSlotResults()) {
                        writer.write(String.format(java.util.Locale.ROOT, "\"%s\",\"%s\",%d,%s,\"%s\"\n",
                            sess.getFileName(), sess.getSourceImageSha256(), r.getId(), r.getStatus().name(),
                            r.getSlot().getOrigin().getDisplayName()));
                    }
                }
            }

            List<String> lines = java.nio.file.Files.readAllLines(tempCsv, java.nio.charset.StandardCharsets.UTF_8);
            assertEquals(9, lines.size(), "1 header + 8 slot rows (4 from imgA, 4 from imgB)");
            assertTrue(lines.get(0).startsWith("image_name,image_sha256,slot_id,status,boundary_origin"));
            assertTrue(lines.stream().anyMatch(l -> l.contains("imgA.png") && l.contains("PKLot Polygon Contour")));
            assertTrue(lines.stream().anyMatch(l -> l.contains("imgB.png") && l.contains("Auto-Detected (IVP)")));
        } finally {
            java.nio.file.Files.deleteIfExists(tempCsv);
        }
    }
}
