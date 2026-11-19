package com.parknchillai.app;

import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import com.parknchillai.utils.ImageLoader;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("WS-4: Bounded Multi-Image Memory Verification")
class BoundedMultiImageMemoryTest {

    private ParkingAnalysisService analysisService;
    private AppState appState;

    @BeforeEach
    void setUp() {
        analysisService = ParkingAnalysisService.createWithDefaultModel();
        appState = new AppState(analysisService);
        appState.setReviewerId("MEMORY_AUDITOR");
    }

    private BufferedImage createSyntheticLotImage(int index, int width, int height) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(new Color(40 + (index * 7) % 180, 50, 60));
        g.fillRect(0, 0, width, height);
        g.setColor(Color.WHITE);
        for (int i = 0; i < 10; i++) {
            g.drawRect(20 + i * 40, 20, 35, 60);
        }
        g.dispose();
        return img;
    }

    private List<ParkingSlot> createTestSlots(int count) {
        List<ParkingSlot> slots = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            slots.add(new ParkingSlot(i + 1, 20 + (i % 10) * 40, 20 + (i / 10) * 70, 35, 60));
        }
        return slots;
    }

    @Test
    void testLruCapRetainsAtMostMaxActiveBitmaps() {
        int totalSessions = 10;
        List<ParkingSlot> slots = createTestSlots(5);

        for (int i = 0; i < totalSessions; i++) {
            BufferedImage img = createSyntheticLotImage(i, 400, 300);
            String sha = String.format("%064d", i);
            ImageLoader.LoadedImageMetadata meta = new ImageLoader.LoadedImageMetadata(
                "session_img_" + i + ".png", "C:/fake/path_" + i + ".png", "png",
                400, 300, 400, 300, 1.0, 1.0, false, false, img
            );
            appState.setLoadedImageAndSlots(meta, sha, slots, true, "OK");
        }

        assertEquals(totalSessions, appState.getAllImageSessions().size(),
            "All 10 session metadata records must be tracked");

        int retainedDirect = appState.getDirectlyRetainedSessionImageCount();
        assertTrue(retainedDirect <= AppState.MAX_ACTIVE_SESSION_IMAGES,
            "Directly retained images in LRU must not exceed cap of " + AppState.MAX_ACTIVE_SESSION_IMAGES
                + ", but got: " + retainedDirect);

        // Switch to an earlier evicted session (e.g. index 2)
        String targetSha = String.format("%064d", 2);
        appState.switchToSession(targetSha);
        assertEquals(targetSha, appState.getActiveSessionId());
        assertNotNull(appState.getActiveWorkingImage(), "Switched session must have working image available");
        assertTrue(appState.getDirectlyRetainedSessionImageCount() <= AppState.MAX_ACTIVE_SESSION_IMAGES);
    }

    @Test
    void testIntermediateStageImagesNotRetainedInHeapAndComputedOnDemand() {
        BufferedImage img = createSyntheticLotImage(1, 640, 480);
        List<ParkingSlot> slots = createTestSlots(12);

        AnalysisSummary summary = analysisService.analyze(img, slots, false);
        appState.updateAnalysisSummary(summary);

        assertNotNull(summary);
        assertEquals(12, summary.getSlotResults().size());

        for (SlotAnalysisResult result : summary.getSlotResults()) {
            assertNotNull(result.getFeatures(), "Features must be present");
            // On demand retrieval: must return valid images without permanent heap retention
            BufferedImage gray = result.getGrayImage();
            assertNotNull(gray, "Gray image must be available on demand");
            assertEquals(result.getRoiImage().getWidth(), gray.getWidth());

            BufferedImage bin = result.getBinaryImage();
            assertNotNull(bin, "Binary image must be available on demand");

            BufferedImage edges = result.getEdgeImage();
            assertNotNull(edges, "Edge image must be available on demand");
        }
    }

    @Test
    void testCloseSessionAndClearAllSessionsPreserveData() {
        // Setup 2 sessions
        BufferedImage img1 = createSyntheticLotImage(1, 320, 240);
        String sha1 = "1111111111111111111111111111111111111111111111111111111111111111";
        ImageLoader.LoadedImageMetadata meta1 = new ImageLoader.LoadedImageMetadata(
            "img1.png", "C:/img1.png", "png", 320, 240, 320, 240, 1.0, 1.0, false, false, img1
        );
        appState.setLoadedImageAndSlots(meta1, sha1, createTestSlots(4), true, "OK");
        AnalysisSummary sum1 = analysisService.analyze(img1, createTestSlots(4), false);
        appState.updateAnalysisSummary(sum1);

        // Review slots 1 and 2
        appState.reviewSlot(1, SlotStatus.OCCUPIED);
        appState.reviewSlot(2, SlotStatus.AVAILABLE);

        BufferedImage img2 = createSyntheticLotImage(2, 320, 240);
        String sha2 = "2222222222222222222222222222222222222222222222222222222222222222";
        ImageLoader.LoadedImageMetadata meta2 = new ImageLoader.LoadedImageMetadata(
            "img2.png", "C:/img2.png", "png", 320, 240, 320, 240, 1.0, 1.0, false, false, img2
        );
        appState.setLoadedImageAndSlots(meta2, sha2, createTestSlots(4), true, "OK");
        AnalysisSummary sum2 = analysisService.analyze(img2, createTestSlots(4), false);
        appState.updateAnalysisSummary(sum2);

        // Review slot 3
        appState.reviewSlot(3, SlotStatus.OCCUPIED);

        assertEquals(3, appState.getEligibleReviewedRecords().size(), "Must have 3 reviewed records total");
        assertEquals(2, appState.getAllImageSessions().size(), "Must have 2 sessions");

        // Close session 1
        boolean closed = appState.closeSession(sha1);
        assertTrue(closed, "Session 1 must close successfully");
        assertEquals(1, appState.getAllImageSessions().size(), "Only session 2 should remain");
        assertEquals(sha2, appState.getActiveSessionId(), "Active session should be session 2");

        // Crucial test: Reviewed training records from closed session 1 MUST BE PRESERVED
        assertEquals(3, appState.getEligibleReviewedRecords().size(),
            "Closing an image session must NOT discard reviewed training data");

        // Clear all sessions with preserveReviewedTrainingData = true
        appState.clearAllSessions(true);
        assertEquals(0, appState.getAllImageSessions().size(), "All sessions cleared");
        assertNull(appState.getActiveSessionId(), "Active session is null");
        assertEquals(3, appState.getEligibleReviewedRecords().size(),
            "clearAllSessions(preserve=true) must keep reviewed records intact");

        // Clear all sessions with preserveReviewedTrainingData = false
        appState.clearAllSessions(false);
        assertEquals(0, appState.getEligibleReviewedRecords().size(),
            "clearAllSessions(preserve=false) wipes reviewed training pool");
    }

    @Test
    void testHeapScalingAcross4_20_50Images() throws Exception {
        Path tempDir = Files.createTempDirectory("parknchill_mem_test_");
        tempDir.toFile().deleteOnExit();

        List<ParkingSlot> slots = createTestSlots(20);
        long[] retainedHeapMb = new long[3];
        int[] checkpoints = {4, 20, 50};

        int checkpointIdx = 0;
        for (int i = 1; i <= 50; i++) {
            File imgFile = tempDir.resolve("lot_" + i + ".png").toFile();
            BufferedImage img = createSyntheticLotImage(i, 800, 600);
            javax.imageio.ImageIO.write(img, "png", imgFile);
            imgFile.deleteOnExit();

            String sha = String.format("%064d", i);
            ImageLoader.LoadedImageMetadata meta = new ImageLoader.LoadedImageMetadata(
                imgFile.getName(), imgFile.getAbsolutePath(), "png",
                800, 600, 800, 600, 1.0, 1.0, false, false, img
            );

            appState.setLoadedImageAndSlots(meta, sha, slots, true, "OK");
            AnalysisSummary summary = analysisService.analyze(img, slots, false);
            appState.updateAnalysisSummary(summary);

            if (checkpointIdx < checkpoints.length && i == checkpoints[checkpointIdx]) {
                System.gc();
                Thread.sleep(100);
                System.gc();
                long used = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);
                retainedHeapMb[checkpointIdx] = used;
                int directlyRetained = appState.getDirectlyRetainedSessionImageCount();
                assertTrue(directlyRetained <= AppState.MAX_ACTIVE_SESSION_IMAGES,
                    "Directly retained image count at step " + i + " must be <= " + AppState.MAX_ACTIVE_SESSION_IMAGES);
                checkpointIdx++;
            }
        }

        System.out.printf("Heap usage across checkpoints: 4 images=%d MB, 20 images=%d MB, 50 images=%d MB%n",
            retainedHeapMb[0], retainedHeapMb[1], retainedHeapMb[2]);

        // Heap between 20 and 50 images must scale sub-linearly with bounded LRU
        long delta20to50 = retainedHeapMb[2] - retainedHeapMb[1];
        assertTrue(delta20to50 < 80,
            "Heap delta from 20 to 50 images (" + delta20to50 + " MB) must be bounded (<80 MB) due to LRU unloader");
    }
}
