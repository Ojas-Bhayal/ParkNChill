package com.parknchillai.imageprocessing;

import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link EdgeDetector}.
 */
class EdgeDetectorTest {

    @Test
    void uniformImageHasNoEdges() {
        // A uniform image has zero gradient everywhere
        BufferedImage uniform = uniformGray(20, 20, 128);
        BufferedImage edges   = EdgeDetector.applySobel(uniform);
        for (int y = 1; y < 19; y++) {
            for (int x = 1; x < 19; x++) {
                assertEquals(0, edges.getRaster().getSample(x, y, 0),
                    "Uniform image should have zero edge magnitude at (" + x + "," + y + ")");
            }
        }
    }

    @Test
    void sharpVerticalEdgeDetected() {
        // Left half black, right half white → strong horizontal gradient
        BufferedImage img = new BufferedImage(20, 20, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < 20; y++) {
            for (int x = 0; x < 20; x++) {
                int v = x < 10 ? 0 : 255;
                img.getRaster().setSample(x, y, 0, v);
            }
        }
        BufferedImage edges = EdgeDetector.applySobel(img);
        // The boundary column should have a high edge magnitude
        int edgeMag = edges.getRaster().getSample(10, 10, 0);
        assertTrue(edgeMag > 100, "Edge magnitude at boundary should be high; got " + edgeMag);
    }

    @Test
    void outputDimensionsMatchInput() {
        BufferedImage img   = uniformGray(15, 25, 100);
        BufferedImage edges = EdgeDetector.applySobel(img);
        assertEquals(15, edges.getWidth());
        assertEquals(25, edges.getHeight());
    }

    @Test
    void nullInputThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> EdgeDetector.applySobel(null));
    }

    // --- Helper -----------------------------------------------------------

    private static BufferedImage uniformGray(int w, int h, int intensity) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                img.getRaster().setSample(x, y, 0, intensity);
            }
        }
        return img;
    }
}
