package com.parknchillai.parking;

import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link RoiExtractor} and {@link ParkingSlot}.
 */
class RoiExtractorTest {

    @Test
    void extractsCorrectSubimage() {
        BufferedImage full = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
        // Paint a distinct color inside the slot region (x=10, y=20, w=30, h=40)
        for (int y = 20; y < 60; y++) {
            for (int x = 10; x < 40; x++) {
                full.setRGB(x, y, 0x123456);
            }
        }

        ParkingSlot slot = new ParkingSlot(1, 10, 20, 30, 40);
        BufferedImage roi = RoiExtractor.extract(full, slot);

        assertEquals(30, roi.getWidth());
        assertEquals(40, roi.getHeight());
        assertEquals(0x123456, roi.getRGB(0, 0) & 0xFFFFFF);
    }

    @Test
    void slotExceedingImageBoundsThrowsException() {
        BufferedImage full = new BufferedImage(50, 50, BufferedImage.TYPE_INT_RGB);
        ParkingSlot outSlot = new ParkingSlot(1, 30, 30, 30, 30); // 30+30 > 50

        assertThrows(IllegalArgumentException.class,
            () -> RoiExtractor.extract(full, outSlot));
    }

    @Test
    void nullArgumentsThrowException() {
        BufferedImage full = new BufferedImage(50, 50, BufferedImage.TYPE_INT_RGB);
        ParkingSlot slot = new ParkingSlot(1, 0, 0, 10, 10);

        assertThrows(IllegalArgumentException.class, () -> RoiExtractor.extract(null, slot));
        assertThrows(IllegalArgumentException.class, () -> RoiExtractor.extract(full, null));
    }

    @Test
    void negativeOrZeroDimensionsThrowIllegalArgumentException() {
        BufferedImage full = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
        assertThrows(IllegalArgumentException.class, () -> RoiExtractor.extract(full, new ParkingSlot(1, -5, 10, 20, 20)));
        assertThrows(IllegalArgumentException.class, () -> RoiExtractor.extract(full, new ParkingSlot(1, 10, -5, 20, 20)));
        assertThrows(IllegalArgumentException.class, () -> RoiExtractor.extract(full, new ParkingSlot(1, 10, 10, 0, 20)));
        assertThrows(IllegalArgumentException.class, () -> RoiExtractor.extract(full, new ParkingSlot(1, 10, 10, 20, -5)));
    }

    @Test
    void testFinding07OrderQuadClockwiseOnSlantedAndRotatedQuads() {
        // Standard rectangle in scrambled order
        java.util.List<double[]> rect = java.util.List.of(
            new double[]{90, 120}, // BR
            new double[]{10, 20},  // TL
            new double[]{10, 120}, // BL
            new double[]{90, 20}   // TR
        );
        double[][] ordered = RoiExtractor.orderQuadClockwise(rect);
        assertArrayEquals(new double[]{10, 20}, ordered[0], 1e-6);  // TL
        assertArrayEquals(new double[]{90, 20}, ordered[1], 1e-6);  // TR
        assertArrayEquals(new double[]{90, 120}, ordered[2], 1e-6); // BR
        assertArrayEquals(new double[]{10, 120}, ordered[3], 1e-6); // BL

        // Rotated diamond
        java.util.List<double[]> diamond = java.util.List.of(
            new double[]{100, 100}, // bottom
            new double[]{0, 50},    // left
            new double[]{200, 50},  // right
            new double[]{100, 0}    // top
        );
        double[][] orderedDiamond = RoiExtractor.orderQuadClockwise(diamond);
        assertNotNull(orderedDiamond);
        assertEquals(4, orderedDiamond.length);
        assertEquals(4, java.util.Arrays.stream(orderedDiamond).distinct().count());
    }
}
