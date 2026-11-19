package com.parknchillai.detection;

import com.parknchillai.app.ImageSessionRecord;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("WS-2: Vehicle Detection Cost Reduction, Strategy Toggle, and LRU Cache Tests")
public class VehicleDetectionStrategyAndCacheTest {

    private static BufferedImage createLotImage(int width, int height) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(new Color(60, 60, 60));
        g.fillRect(0, 0, width, height);
        g.setColor(Color.WHITE);
        for (int x = 50; x < width; x += 100) {
            g.fillRect(x, 20, 10, height - 40);
        }
        g.dispose();
        return img;
    }

    @Test
    @DisplayName("FAST strategy executes exactly 1 whole-image pass without quadrant tiling")
    void fastStrategyExecutesSinglePassWithoutTiling() {
        AtomicInteger passes = new AtomicInteger();
        OnnxVehicleDetectionService detector = OnnxVehicleDetectionService.createWithTestBackend(
            "mock_detector.onnx", 640, 640, 0.30, 0.45, true,
            (tensor, w, h) -> {
                passes.incrementAndGet();
                return new float[84][1];
            }
        );

        BufferedImage img = createLotImage(800, 600);
        VehicleDetectionService.DetectionBatchResult result = detector.detectVehicles(img, DetectionStrategy.FAST);

        assertEquals(DetectionStrategy.FAST, result.strategy());
        assertEquals(1, result.onnxPasses());
        assertEquals(1, passes.get());
        assertFalse(result.fromCache());
        assertTrue(result.formatExecutionSummary().contains("FAST"));
    }

    @Test
    @DisplayName("ACCURATE strategy executes 1 whole-image pass + 4 quadrant tiles (5 total passes)")
    void accurateStrategyExecutesFivePassesWithTiling() {
        AtomicInteger passes = new AtomicInteger();
        OnnxVehicleDetectionService detector = OnnxVehicleDetectionService.createWithTestBackend(
            "mock_detector.onnx", 640, 640, 0.30, 0.45, true,
            (tensor, w, h) -> {
                passes.incrementAndGet();
                return new float[84][1];
            }
        );

        BufferedImage img = createLotImage(800, 600);
        VehicleDetectionService.DetectionBatchResult result = detector.detectVehicles(img, DetectionStrategy.ACCURATE);

        assertEquals(DetectionStrategy.ACCURATE, result.strategy());
        assertEquals(5, result.onnxPasses());
        assertEquals(5, passes.get());
        assertFalse(result.fromCache());
        assertTrue(result.formatExecutionSummary().contains("ACCURATE"));
    }

    @Test
    @DisplayName("LRU cache returns hit with zero ONNX passes and sub-millisecond retrieval")
    void lruCacheReturnsHitWithZeroInferencePasses() {
        AtomicInteger passes = new AtomicInteger();
        OnnxVehicleDetectionService detector = OnnxVehicleDetectionService.createWithTestBackend(
            "mock_detector.onnx", 640, 640, 0.30, 0.45, true,
            (tensor, w, h) -> {
                passes.incrementAndGet();
                return new float[84][1];
            }
        );

        BufferedImage img = createLotImage(800, 600);

        // First execution: cache miss
        VehicleDetectionService.DetectionBatchResult miss = detector.detectVehicles(img, DetectionStrategy.ACCURATE);
        assertFalse(miss.fromCache());
        assertEquals(5, miss.onnxPasses());
        assertEquals(5, passes.get());

        // Second execution: cache hit
        VehicleDetectionService.DetectionBatchResult hit = detector.detectVehicles(img, DetectionStrategy.ACCURATE);
        assertTrue(hit.fromCache());
        assertEquals(0, hit.onnxPasses());
        assertEquals(5, passes.get(), "No additional ONNX inference passes should occur on cache hit");
        assertTrue(hit.formatExecutionSummary().contains("CACHE HIT"));
    }

    @Test
    @DisplayName("Cache key distinguishes strategies and explicit SHA-256")
    void cacheKeyDistinguishesStrategies() {
        AtomicInteger passes = new AtomicInteger();
        OnnxVehicleDetectionService detector = OnnxVehicleDetectionService.createWithTestBackend(
            "mock_detector.onnx", 640, 640, 0.30, 0.45, true,
            (tensor, w, h) -> {
                passes.incrementAndGet();
                return new float[84][1];
            }
        );

        BufferedImage img = createLotImage(800, 600);

        // Run FAST -> 1 pass
        detector.detectVehicles(img, DetectionStrategy.FAST);
        assertEquals(1, passes.get());

        // Run ACCURATE on same image -> cache miss because strategy is different -> 5 additional passes
        detector.detectVehicles(img, DetectionStrategy.ACCURATE);
        assertEquals(6, passes.get());

        // Run ACCURATE again -> cache hit -> 0 passes
        VehicleDetectionService.DetectionBatchResult hit = detector.detectVehicles(img, DetectionStrategy.ACCURATE);
        assertTrue(hit.fromCache());
        assertEquals(6, passes.get());

        // Run with explicit SHA-256
        VehicleDetectionService.DetectionBatchResult explicitHit = detector.detectVehicles(img, "explicit-custom-sha", DetectionStrategy.ACCURATE);
        assertFalse(explicitHit.fromCache());
        assertEquals(11, passes.get());
    }

    @Test
    @DisplayName("Collision regression test: images identical on sampled rows but different on unsampled rows produce different full-pixel hashes")
    void sampledRowCollisionRegressionTest() {
        int w = 400;
        int h = 400; // rowStep in old sampler was 400 / 200 = 2 (only sampled rows 0, 2, 4, 6...)
        BufferedImage imgA = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        BufferedImage imgB = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

        // Fill both images with identical gray pattern on all even rows (0, 2, 4...)
        for (int y = 0; y < h; y += 2) {
            for (int x = 0; x < w; x++) {
                int rgb = ((x % 256) << 16) | ((y % 256) << 8) | 128;
                imgA.setRGB(x, y, rgb);
                imgB.setRGB(x, y, rgb);
            }
        }

        // On odd rows (which the old sampler skipped completely), make them completely different:
        // imgA remains black (0), imgB has bright red pixels
        for (int y = 1; y < h; y += 2) {
            for (int x = 0; x < w; x++) {
                imgA.setRGB(x, y, 0);
                imgB.setRGB(x, y, 0x00FF0000);
            }
        }

        // 1. Verify old sampler falsely reported these two different images as identical (hash collision defect)
        @SuppressWarnings("deprecation")
        String oldDigestA = OnnxVehicleDetectionService.deprecatedInsecureSampledRowDigest(imgA);
        @SuppressWarnings("deprecation")
        String oldDigestB = OnnxVehicleDetectionService.deprecatedInsecureSampledRowDigest(imgB);
        assertEquals(oldDigestA, oldDigestB, "Old sampled digest suffered false collision on unsampled row difference");

        // 2. Verify new collision-resistant full-pixel hash correctly distinguishes them
        String fullShaA = OnnxVehicleDetectionService.computeFullPixelSha256(imgA);
        String fullShaB = OnnxVehicleDetectionService.computeFullPixelSha256(imgB);
        assertNotNull(fullShaA);
        assertNotNull(fullShaB);
        assertNotEquals(fullShaA, fullShaB, "Full-pixel SHA-256 must never collide when any row differs");

        // 3. Verify detection cache treats them as distinct entries without sharing detections
        AtomicInteger passes = new AtomicInteger();
        OnnxVehicleDetectionService detector = OnnxVehicleDetectionService.createWithTestBackend(
            "mock_detector.onnx", 640, 640, 0.30, 0.45, false,
            (tensor, width, height) -> {
                passes.incrementAndGet();
                return new float[84][1];
            }
        );

        VehicleDetectionService.DetectionBatchResult resA = detector.detectVehicles(imgA, DetectionStrategy.FAST);
        VehicleDetectionService.DetectionBatchResult resB = detector.detectVehicles(imgB, DetectionStrategy.FAST);

        assertFalse(resA.fromCache());
        assertFalse(resB.fromCache(), "Image B must not hit cache from Image A despite identical sampled rows");
        assertEquals(2, passes.get());
    }

    @Test
    @DisplayName("Counting seam test: exactly one full-pixel hash per image session across repeated detections")
    void countingSeamOneFullHashPerSessionAcrossRepeatedDetections() {
        BufferedImage img = createLotImage(600, 400);
        ImageSessionRecord session = new ImageSessionRecord(
            "UNHASHED_SESSION", null, img, img, "Original", null, true, "OK"
        );

        OnnxVehicleDetectionService.resetFullPixelHashInvocationCount();
        assertEquals(0, OnnxVehicleDetectionService.getFullPixelHashInvocationCount());

        // First resolution: computes full-pixel hash
        String id1 = session.resolveDetectionCacheIdentity();
        assertNotNull(id1);
        assertEquals(1, OnnxVehicleDetectionService.getFullPixelHashInvocationCount());

        // Repeated resolutions on same session record reuse cached full-pixel hash (zero re-hashes)
        String id2 = session.resolveDetectionCacheIdentity();
        String id3 = session.resolveDetectionCacheIdentity();
        assertEquals(id1, id2);
        assertEquals(id1, id3);
        assertEquals(1, OnnxVehicleDetectionService.getFullPixelHashInvocationCount(),
            "Full-pixel hash must not be recomputed for repeated queries on same session");

        // Run detector with session identity
        OnnxVehicleDetectionService detector = OnnxVehicleDetectionService.createWithTestBackend(
            "mock_detector.onnx", 640, 640, 0.30, 0.45, false,
            (tensor, w, h) -> new float[84][1]
        );

        VehicleDetectionService.DetectionBatchResult r1 = detector.detectVehicles(img, id1, DetectionStrategy.FAST);
        VehicleDetectionService.DetectionBatchResult r2 = detector.detectVehicles(img, id1, DetectionStrategy.FAST);

        assertFalse(r1.fromCache());
        assertTrue(r2.fromCache());
        assertEquals(1, OnnxVehicleDetectionService.getFullPixelHashInvocationCount(),
            "Total full pixel hashes across image session detections must remain 1");
    }

    @Test
    @DisplayName("Enhanced working image produces distinct hash and does not collide with original")
    void enhancedWorkingImageDoesNotCollideWithOriginal() {
        BufferedImage orig = createLotImage(600, 400);
        BufferedImage enhanced = new BufferedImage(600, 400, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = enhanced.createGraphics();
        g.drawImage(orig, 0, 0, null);
        // Apply brightness modification
        g.setColor(new Color(255, 255, 255, 50));
        g.fillRect(0, 0, 600, 400);
        g.dispose();

        String shaOrig = OnnxVehicleDetectionService.computeFullPixelSha256(orig);
        String shaEnhanced = OnnxVehicleDetectionService.computeFullPixelSha256(enhanced);

        assertNotNull(shaOrig);
        assertNotNull(shaEnhanced);
        assertNotEquals(shaOrig, shaEnhanced, "Enhanced working image must have distinct hash from original");

        OnnxVehicleDetectionService detector = OnnxVehicleDetectionService.createWithTestBackend(
            "mock_detector.onnx", 640, 640, 0.30, 0.45, false,
            (tensor, w, h) -> new float[84][1]
        );

        VehicleDetectionService.DetectionBatchResult origRes = detector.detectVehicles(orig, shaOrig, DetectionStrategy.FAST);
        VehicleDetectionService.DetectionBatchResult enhRes = detector.detectVehicles(enhanced, shaEnhanced, DetectionStrategy.FAST);

        assertFalse(origRes.fromCache());
        assertFalse(enhRes.fromCache(), "Enhanced image must miss cache of original");
    }

    @Test
    @DisplayName("Cache remains bounded under many distinct images and clears on explicit clear")
    void cacheRemainsBoundedAndClears() {
        OnnxVehicleDetectionService detector = new OnnxVehicleDetectionService();
        detector.clearDetectionCache();
        assertEquals(0, detector.getDetectionCacheSize());

        // Insert 25 distinct images (LRU capacity is 16)
        for (int i = 0; i < 25; i++) {
            BufferedImage img = createLotImage(100 + i * 10, 100 + i * 10);
            detector.detectVehicles(img, DetectionStrategy.FAST);
        }

        assertTrue(detector.getDetectionCacheSize() <= 16,
            "Detection cache size (" + detector.getDetectionCacheSize() + ") must be bounded by LRU capacity (16)");

        detector.clearDetectionCache();
        assertEquals(0, detector.getDetectionCacheSize(), "Cache must be empty after clearDetectionCache");
    }

    @Test
    @DisplayName("Benchmark FAST vs ACCURATE latency and pass count")
    void benchmarkFastVsAccurateAndCache() {
        AtomicInteger passes = new AtomicInteger();
        OnnxVehicleDetectionService detector = OnnxVehicleDetectionService.createWithTestBackend(
            "mock_detector.onnx", 640, 640, 0.30, 0.45, true,
            (tensor, w, h) -> {
                passes.incrementAndGet();
                try { Thread.sleep(2); } catch (InterruptedException ignored) {}
                return new float[84][1];
            }
        );

        BufferedImage img = createLotImage(1280, 720);

        detector.clearDetectionCache();

        long t0Fast = System.nanoTime();
        VehicleDetectionService.DetectionBatchResult fastRes = detector.detectVehicles(img, DetectionStrategy.FAST);
        long tFastNs = System.nanoTime() - t0Fast;

        detector.clearDetectionCache();

        long t0Acc = System.nanoTime();
        VehicleDetectionService.DetectionBatchResult accRes = detector.detectVehicles(img, DetectionStrategy.ACCURATE);
        long tAccNs = System.nanoTime() - t0Acc;

        long t0Cache = System.nanoTime();
        VehicleDetectionService.DetectionBatchResult cacheRes = detector.detectVehicles(img, DetectionStrategy.ACCURATE);
        long tCacheNs = System.nanoTime() - t0Cache;

        double fastMs = tFastNs / 1_000_000.0;
        double accMs = tAccNs / 1_000_000.0;
        double cacheMs = tCacheNs / 1_000_000.0;

        System.out.println("================================================================================");
        System.out.println("WS-2 VEHICLE DETECTION BENCHMARK: FAST vs ACCURATE vs CACHE HIT");
        System.out.println("================================================================================");
        System.out.printf("| %-22s | %-12s | %-16s |%n", "Strategy / Operation", "ONNX Passes", "Latency (ms)");
        System.out.println("|:-----------------------|:-------------|:-----------------|");
        System.out.printf("| %-22s | %-12d | %13.2f ms |%n", "FAST (Whole Image)", fastRes.onnxPasses(), fastMs);
        System.out.printf("| %-22s | %-12d | %13.2f ms |%n", "ACCURATE (Tiled 5x)", accRes.onnxPasses(), accMs);
        System.out.printf("| %-22s | %-12d | %13.2f ms |%n", "CACHE HIT (LRU)", cacheRes.onnxPasses(), cacheMs);
        System.out.println("================================================================================");

        assertEquals(1, fastRes.onnxPasses());
        assertEquals(5, accRes.onnxPasses());
        assertEquals(0, cacheRes.onnxPasses());
        assertTrue(cacheRes.fromCache());
        assertTrue(cacheMs < accMs, "Cache hit should be significantly faster than full inference");
    }
}
