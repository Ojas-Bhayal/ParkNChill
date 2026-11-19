package com.parknchillai.imageprocessing;

import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link MorphologyProcessor}.
 */
class MorphologyProcessorTest {

    @Test
    void erosionEliminatesIsolatedWhitePixel() {
        BufferedImage img = binaryImage(7, 7, 0);
        // Single isolated foreground pixel at center
        img.getRaster().setSample(3, 3, 0, 255);

        BufferedImage eroded = MorphologyProcessor.erode(img, 3);
        // An isolated pixel surrounded by background must be eliminated by 3x3 erosion
        assertEquals(0, eroded.getRaster().getSample(3, 3, 0),
            "Single pixel should be eroded to background");
    }

    @Test
    void dilationExpandsForegroundPixel() {
        BufferedImage img = binaryImage(7, 7, 0);
        img.getRaster().setSample(3, 3, 0, 255);

        BufferedImage dilated = MorphologyProcessor.dilate(img, 3);
        // Center and all 8-neighbours should now be white
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                assertEquals(255, dilated.getRaster().getSample(3 + dx, 3 + dy, 0),
                    "Neighbor (" + (3 + dx) + "," + (3 + dy) + ") should be dilated to foreground");
            }
        }
    }

    @Test
    void closingFillsSingleHole() {
        BufferedImage img = binaryImage(7, 7, 255);
        // Single black hole at center
        img.getRaster().setSample(3, 3, 0, 0);

        BufferedImage closed = MorphologyProcessor.close(img, 3);
        assertEquals(255, closed.getRaster().getSample(3, 3, 0),
            "Morphological closing should fill small 1-pixel hole");
    }

    @Test
    void openingRemovesIsolatedNoiseBlob() {
        BufferedImage img = binaryImage(9, 9, 0);
        img.getRaster().setSample(4, 4, 0, 255);

        BufferedImage opened = MorphologyProcessor.open(img, 3);
        assertEquals(0, opened.getRaster().getSample(4, 4, 0),
            "Morphological opening should remove small 1-pixel noise blob");
    }

    @Test
    void invalidKernelThrowsException() {
        BufferedImage img = binaryImage(5, 5, 0);
        assertThrows(IllegalArgumentException.class,
            () -> MorphologyProcessor.erode(img, 2));
        assertThrows(IllegalArgumentException.class,
            () -> MorphologyProcessor.dilate(img, 1));
    }

    // --- Helper -----------------------------------------------------------

    private static BufferedImage binaryImage(int w, int h, int fillVal) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                img.getRaster().setSample(x, y, 0, fillVal);
            }
        }
        return img;
    }
}
