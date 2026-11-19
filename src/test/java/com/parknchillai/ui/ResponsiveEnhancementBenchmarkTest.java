package com.parknchillai.ui;

import com.parknchillai.imageprocessing.ImageEnhancer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("WS-3: Responsive Enhancement Preview and Apply Benchmark")
class ResponsiveEnhancementBenchmarkTest {

    @Test
    void testPreviewDownscalingPreservesBoundsAndAspectRatio() {
        BufferedImage bigImage = new BufferedImage(4000, 3000, BufferedImage.TYPE_INT_RGB);
        BufferedImage preview = ImageEnhancementDialog.createPreviewScale(bigImage, 640, 480);

        assertNotNull(preview);
        assertTrue(preview.getWidth() <= 640, "Preview width must be <= 640; got " + preview.getWidth());
        assertTrue(preview.getHeight() <= 480, "Preview height must be <= 480; got " + preview.getHeight());
        assertEquals(640, preview.getWidth());
        assertEquals(480, preview.getHeight());

        BufferedImage smallImage = new BufferedImage(200, 150, BufferedImage.TYPE_INT_RGB);
        BufferedImage smallPreview = ImageEnhancementDialog.createPreviewScale(smallImage, 640, 480);
        assertSame(smallImage, smallPreview, "Images smaller than preview bounds must not be re-allocated");
    }

    @Test
    void benchmarkPreviewLatencyAndFullApplyTime() {
        System.out.println("================================================================================");
        System.out.println("WS-3 ENHANCEMENT BENCHMARK: Downscaled Interactive Preview vs Full Resolution");
        System.out.println("================================================================================");

        // 1. Synthetic 1280x720 image
        BufferedImage img720p = createBenchImage(1280, 720);
        BufferedImage preview720p = ImageEnhancementDialog.createPreviewScale(img720p, 640, 480);

        // 2. Synthetic 4000x3000 (12 Megapixel) image
        BufferedImage img4k = createBenchImage(4000, 3000);
        BufferedImage preview4k = ImageEnhancementDialog.createPreviewScale(img4k, 640, 480);

        // Warmup
        for (int i = 0; i < 3; i++) {
            ImageEnhancer.adjust(preview720p, 1.2, 10.0, 0.9, 0);
            ImageEnhancer.adjust(img720p, 1.2, 10.0, 0.9, 0);
        }

        // Measure Downscaled Preview Latency (720p source -> 640x360 downscaled)
        double previewMs720 = measureAverageTimeMs(() -> ImageEnhancer.adjust(preview720p, 1.2, 10.0, 0.9, 0), 10);
        // Measure Full 1280x720 Apply
        double applyMs720 = measureAverageTimeMs(() -> ImageEnhancer.adjust(img720p, 1.2, 10.0, 0.9, 0), 10);

        // Measure Downscaled Preview Latency (4K source -> 640x480 downscaled)
        double previewMs4k = measureAverageTimeMs(() -> ImageEnhancer.adjust(preview4k, 1.2, 10.0, 0.9, 0), 10);
        // Measure Full 4000x3000 Apply
        double applyMs4k = measureAverageTimeMs(() -> ImageEnhancer.adjust(img4k, 1.2, 10.0, 0.9, 0), 5);

        System.out.printf(Locale.ROOT, "| Image Size | Downscaled Preview (ms) | Full-Res Apply (ms) | Preview Speedup |%n");
        System.out.println("|:---:|:---:|:---:|:---:|");
        System.out.printf(Locale.ROOT, "| 1280x720 (720p) | %.2f ms | %.2f ms | **%.1fx** |%n",
            previewMs720, applyMs720, applyMs720 / Math.max(0.01, previewMs720));
        System.out.printf(Locale.ROOT, "| 4000x3000 (12 MP) | %.2f ms | %.2f ms | **%.1fx** |%n",
            previewMs4k, applyMs4k, applyMs4k / Math.max(0.01, previewMs4k));

        assertTrue(previewMs720 < 100.0, "Downscaled preview latency on 720p must be < 100ms (interactive range)");
        assertTrue(previewMs4k < 100.0, "Downscaled preview latency on 4K must be < 100ms (interactive range)");
    }

    private static double measureAverageTimeMs(Runnable r, int iterations) {
        long totalNs = 0;
        for (int i = 0; i < iterations; i++) {
            long t0 = System.nanoTime();
            r.run();
            long t1 = System.nanoTime();
            totalNs += (t1 - t0);
        }
        return (totalNs / (double) iterations) / 1_000_000.0;
    }

    private static BufferedImage createBenchImage(int w, int h) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(new Color(60, 65, 70));
        g.fillRect(0, 0, w, h);
        g.setColor(Color.WHITE);
        g.drawRect(50, 50, w - 100, h - 100);
        g.dispose();
        return img;
    }
}
