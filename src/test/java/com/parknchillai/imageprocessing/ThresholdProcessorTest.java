package com.parknchillai.imageprocessing;

import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

class ThresholdProcessorTest {

    @Test
    void fixedThresholdBrightPixelBecomesWhite() {
        BufferedImage img = grayImage(1, 1, 200);
        BufferedImage result = ThresholdProcessor.applyFixedThreshold(img, 128);
        assertEquals(255, result.getRaster().getSample(0, 0, 0));
    }

    @Test
    void fixedThresholdDarkPixelBecomesBlack() {
        BufferedImage img = grayImage(1, 1, 50);
        BufferedImage result = ThresholdProcessor.applyFixedThreshold(img, 128);
        assertEquals(0, result.getRaster().getSample(0, 0, 0));
    }

    @Test
    void pixelExactlyAtThresholdIsBlack() {
        BufferedImage img = grayImage(1, 1, 128);
        BufferedImage result = ThresholdProcessor.applyFixedThreshold(img, 128);
        assertEquals(0, result.getRaster().getSample(0, 0, 0));
    }

    @Test
    void bimodalImageOtsuThresholdLiesBetweenClusters() {
        BufferedImage img = bimodalGrayImage(10, 10);
        int t = ThresholdProcessor.computeOtsuThreshold(img);
        assertTrue(t >= 50 && t < 200,
            "Otsu threshold should lie between the two clusters; got " + t);
    }

    @Test
    void uniformGrayImageBinarisesToAllBackgroundZeros() {
        BufferedImage uniformAsphalt = grayImage(10, 10, 60);
        int t = ThresholdProcessor.computeOtsuThreshold(uniformAsphalt);
        assertEquals(60, t, "Uniform image Otsu threshold should equal uniform intensity");

        BufferedImage binary = ThresholdProcessor.applyOtsuThreshold(uniformAsphalt);
        for (int y = 0; y < binary.getHeight(); y++) {
            for (int x = 0; x < binary.getWidth(); x++) {
                assertEquals(0, binary.getRaster().getSample(x, y, 0),
                    "Uniform featureless image must binarise to 0 (background), not 255");
            }
        }
    }

    @Test
    void adaptiveThresholdSuppressesSmoothShadowGradientOnEmptyAsphalt() {
        // Simulate an 80x100 empty asphalt slot with a smooth illumination gradient from 40 (shadow) to 110 (sunlight)
        BufferedImage shadowedAsphalt = new BufferedImage(80, 100, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < 100; y++) {
            for (int x = 0; x < 80; x++) {
                int val = 40 + (int) Math.round((70.0 * x) / 79.0);
                shadowedAsphalt.getRaster().setSample(x, y, 0, val);
            }
        }

        // Global Otsu splits the gradient roughly in half (~50% false foreground),
        // whereas local integral-image adaptive thresholding produces 0% false foreground!
        BufferedImage adaptiveBinary = ThresholdProcessor.applyAdaptiveThreshold(shadowedAsphalt);
        int whitePixels = 0;
        for (int y = 0; y < 100; y++) {
            for (int x = 0; x < 80; x++) {
                if (adaptiveBinary.getRaster().getSample(x, y, 0) == 255) {
                    whitePixels++;
                }
            }
        }
        assertEquals(0, whitePixels,
            "Smooth shadow gradient across empty asphalt should produce 0 false foreground pixels under adaptive thresholding");
    }

    @Test
    void adaptiveThresholdSegmentsDarkAchromaticVehicleOnMediumAsphalt() {
        // 80x100 medium-dark asphalt (intensity 62) with a dark matte-black vehicle body (intensity 26)
        BufferedImage roi = grayImage(80, 100, 62);
        Graphics2D g = roi.createGraphics();
        g.setColor(new Color(26, 26, 26));
        g.fillRoundRect(12, 12, 56, 76, 12, 12);
        g.setColor(new Color(14, 14, 14));
        g.fillRect(18, 26, 44, 16);
        g.dispose();

        BufferedImage binary = ThresholdProcessor.applyAdaptiveThreshold(roi);
        int whitePixels = 0;
        for (int y = 0; y < 100; y++) {
            for (int x = 0; x < 80; x++) {
                if (binary.getRaster().getSample(x, y, 0) == 255) {
                    whitePixels++;
                }
            }
        }
        double fgRatio = (double) whitePixels / (80 * 100);
        assertTrue(fgRatio >= 0.22,
            "Two-sided adaptive thresholding must segment dark achromatic vehicle structure (fgRatio >= 0.22); got " + fgRatio);
    }

    @Test
    void otsuOutputIsBinary() {
        BufferedImage img = bimodalGrayImage(8, 8);
        BufferedImage binary = ThresholdProcessor.applyOtsuThreshold(img);
        for (int y = 0; y < binary.getHeight(); y++) {
            for (int x = 0; x < binary.getWidth(); x++) {
                int v = binary.getRaster().getSample(x, y, 0);
                assertTrue(v == 0 || v == 255, "Expected binary pixel; got " + v);
            }
        }
    }

    @Test
    void nullInputThrowsException() {
        assertThrows(IllegalArgumentException.class,
            () -> ThresholdProcessor.applyFixedThreshold(null, 128));
        assertThrows(IllegalArgumentException.class,
            () -> ThresholdProcessor.applyOtsuThreshold(null));
        assertThrows(IllegalArgumentException.class,
            () -> ThresholdProcessor.applyAdaptiveThreshold(null));
    }

    @Test
    void invalidThresholdThrowsException() {
        BufferedImage img = grayImage(1, 1, 100);
        assertThrows(IllegalArgumentException.class,
            () -> ThresholdProcessor.applyFixedThreshold(img, -1));
        assertThrows(IllegalArgumentException.class,
            () -> ThresholdProcessor.applyFixedThreshold(img, 256));
        assertThrows(IllegalArgumentException.class,
            () -> ThresholdProcessor.applyAdaptiveThreshold(img, 4, 0.12)); // even windowSize
    }

    static BufferedImage grayImage(int w, int h, int intensity) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                img.getRaster().setSample(x, y, 0, intensity);
            }
        }
        return img;
    }

    static BufferedImage bimodalGrayImage(int w, int h) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < h; y++) {
            int v = y < h / 2 ? 50 : 200;
            for (int x = 0; x < w; x++) {
                img.getRaster().setSample(x, y, 0, v);
            }
        }
        return img;
    }
}
