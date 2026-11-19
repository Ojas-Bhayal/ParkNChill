package com.parknchillai.imageprocessing;

import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

class FilterProcessorTest {

    @Test
    void boxFilterAveragesUniformImage() {
        BufferedImage img = uniformGray(10, 10, 100);
        BufferedImage filtered = FilterProcessor.applyBoxFilter(img, 3);
        assertEquals(10, filtered.getWidth());
        assertEquals(10, filtered.getHeight());
        assertEquals(100, filtered.getRaster().getSample(5, 5, 0));
    }

    @Test
    void gaussianFilterPreservesUniformImage() {
        BufferedImage img = uniformGray(12, 12, 180);
        BufferedImage filtered = FilterProcessor.applyGaussianFilter(img, 5);
        assertEquals(180, filtered.getRaster().getSample(6, 6, 0));
    }

    @Test
    void gaussianFilterSmoothsIsolatedNoisePixel() {
        BufferedImage img = uniformGray(9, 9, 0);
        img.getRaster().setSample(4, 4, 0, 255);

        BufferedImage filtered = FilterProcessor.applyGaussianFilter(img, 5);
        int centerVal = filtered.getRaster().getSample(4, 4, 0);

        assertTrue(centerVal < 200, "Gaussian smoothing should attenuate noise spike; got " + centerVal);
        assertTrue(centerVal > 0, "Noise spike should spread energy; got " + centerVal);
    }

    @Test
    void smallImages1x1And2x2With5x5GaussianDoNotThrow() {
        BufferedImage img1x1 = uniformGray(1, 1, 120);
        BufferedImage out1x1 = FilterProcessor.applyGaussianFilter(img1x1, 5);
        assertEquals(120, out1x1.getRaster().getSample(0, 0, 0));

        BufferedImage img2x2 = uniformGray(2, 2, 90);
        BufferedImage out2x2 = FilterProcessor.applyGaussianFilter(img2x2, 5);
        assertEquals(90, out2x2.getRaster().getSample(1, 1, 0));
    }

    @Test
    void evenKernelSizeThrowsException() {
        BufferedImage img = uniformGray(8, 8, 50);
        assertThrows(IllegalArgumentException.class,
            () -> FilterProcessor.applyBoxFilter(img, 4));
        assertThrows(IllegalArgumentException.class,
            () -> FilterProcessor.applyGaussianFilter(img, 2));
    }

    @Test
    void kernelSizeLessThanThreeThrowsException() {
        BufferedImage img = uniformGray(8, 8, 50);
        assertThrows(IllegalArgumentException.class,
            () -> FilterProcessor.applyBoxFilter(img, 1));
    }

    @Test
    void nullImageThrowsException() {
        assertThrows(IllegalArgumentException.class,
            () -> FilterProcessor.applyGaussianFilter(null, 5));
    }

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
