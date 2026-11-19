package com.parknchillai.imageprocessing;

import com.parknchillai.utils.ImageNormalizer;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

class PerspectiveTransformerTest {

    @Test
    void normalizeResolutionDownscalesOversizedImagePreservingAspectRatio() {
        BufferedImage huge = new BufferedImage(2400, 1800, BufferedImage.TYPE_INT_RGB);
        BufferedImage normalized = ImageNormalizer.normalizeResolution(huge, 1280);
        assertEquals(1280, normalized.getWidth());
        assertEquals(960, normalized.getHeight());

        BufferedImage small = new BufferedImage(380, 350, BufferedImage.TYPE_INT_RGB);
        assertSame(small, ImageNormalizer.normalizeResolution(small, 1280));
    }

    @Test
    void identityPerspectiveWarpPreservesImageContent() {
        BufferedImage src = new BufferedImage(40, 30, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = src.createGraphics();
        g.setColor(new Color(200, 100, 50));
        g.fillRect(0, 0, 40, 30);
        g.dispose();

        double[][] quad = {
            {0, 0},
            {39, 0},
            {39, 29},
            {0, 29}
        };
        BufferedImage warped = PerspectiveTransformer.warpPerspective(src, quad, 40, 30);
        assertEquals(40, warped.getWidth());
        assertEquals(30, warped.getHeight());

        int rgb = warped.getRGB(20, 15);
        assertEquals(200, (rgb >> 16) & 0xFF, 1);
        assertEquals(100, (rgb >> 8) & 0xFF, 1);
        assertEquals(50, rgb & 0xFF, 1);
    }

    @Test
    void collinearQuadThrowsException() {
        BufferedImage src = new BufferedImage(20, 20, BufferedImage.TYPE_BYTE_GRAY);
        double[][] collinear = {
            {0, 0}, {5, 5}, {10, 10}, {15, 15}
        };
        assertThrows(IllegalArgumentException.class,
            () -> PerspectiveTransformer.warpPerspective(src, collinear, 20, 20));
    }
}
