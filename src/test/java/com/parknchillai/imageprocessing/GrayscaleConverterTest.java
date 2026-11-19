package com.parknchillai.imageprocessing;

import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link GrayscaleConverter}.
 *
 * <p>Tests use programmatically generated images so no external files are required.
 */
class GrayscaleConverterTest {

    @Test
    void pureRedPixelProducesExpectedLuminance() {
        BufferedImage img = singlePixel(255, 0, 0);
        BufferedImage gray = GrayscaleConverter.toGrayscale(img);
        int expected = (int) Math.round(0.299 * 255); // 76
        assertEquals(expected, GrayscaleConverter.getGrayValue(gray, 0, 0), 1,
            "Red pixel luminance should follow BT.601 formula");
    }

    @Test
    void pureGreenPixelProducesExpectedLuminance() {
        BufferedImage img = singlePixel(0, 255, 0);
        BufferedImage gray = GrayscaleConverter.toGrayscale(img);
        int expected = (int) Math.round(0.587 * 255); // 149
        assertEquals(expected, GrayscaleConverter.getGrayValue(gray, 0, 0), 1);
    }

    @Test
    void pureBluePixelProducesExpectedLuminance() {
        BufferedImage img = singlePixel(0, 0, 255);
        BufferedImage gray = GrayscaleConverter.toGrayscale(img);
        int expected = (int) Math.round(0.114 * 255); // 29
        assertEquals(expected, GrayscaleConverter.getGrayValue(gray, 0, 0), 1);
    }

    @Test
    void whitePixelRemainsWhite() {
        BufferedImage img = singlePixel(255, 255, 255);
        BufferedImage gray = GrayscaleConverter.toGrayscale(img);
        assertEquals(255, GrayscaleConverter.getGrayValue(gray, 0, 0));
    }

    @Test
    void blackPixelRemainsBlack() {
        BufferedImage img = singlePixel(0, 0, 0);
        BufferedImage gray = GrayscaleConverter.toGrayscale(img);
        assertEquals(0, GrayscaleConverter.getGrayValue(gray, 0, 0));
    }

    @Test
    void nullInputThrowsException() {
        assertThrows(IllegalArgumentException.class,
            () -> GrayscaleConverter.toGrayscale(null));
    }

    @Test
    void outputTypeIsByteGray() {
        BufferedImage img = singlePixel(100, 150, 200);
        BufferedImage gray = GrayscaleConverter.toGrayscale(img);
        assertEquals(BufferedImage.TYPE_BYTE_GRAY, gray.getType());
    }

    @Test
    void outputDimensionsMatchInput() {
        BufferedImage img = new BufferedImage(30, 20, BufferedImage.TYPE_INT_RGB);
        BufferedImage gray = GrayscaleConverter.toGrayscale(img);
        assertEquals(30, gray.getWidth());
        assertEquals(20, gray.getHeight());
    }

    // --- Helper -----------------------------------------------------------

    private static BufferedImage singlePixel(int r, int g, int b) {
        BufferedImage img = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
        img.setRGB(0, 0, (r << 16) | (g << 8) | b);
        return img;
    }
}
