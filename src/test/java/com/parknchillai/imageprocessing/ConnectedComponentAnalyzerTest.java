package com.parknchillai.imageprocessing;

import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

class ConnectedComponentAnalyzerTest {

    @Test
    void emptyBinaryImageHasZeroBlobRatio() {
        BufferedImage empty = new BufferedImage(20, 20, BufferedImage.TYPE_BYTE_GRAY);
        ConnectedComponentAnalyzer.ComponentSummary summary = ConnectedComponentAnalyzer.analyze(empty);
        assertEquals(0, summary.getLargestBlobPixels());
        assertEquals(0.0, summary.getLargestBlobRatio(), 1e-9);
        assertEquals(0, summary.getSignificantBlobCount());
    }

    @Test
    void contiguousVehicleBlobDistinguishedFromScatteredNoiseWithSameTotalForeground() {
        BufferedImage contiguousCar = new BufferedImage(20, 20, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 5; y < 15; y++) {
            for (int x = 5; x < 15; x++) {
                contiguousCar.getRaster().setSample(x, y, 0, 255);
            }
        }

        BufferedImage scatteredSpeckles = new BufferedImage(20, 20, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < 20; y += 2) {
            for (int x = 0; x < 20; x += 2) {
                scatteredSpeckles.getRaster().setSample(x, y, 0, 255);
            }
        }

        double carBlobRatio = ConnectedComponentAnalyzer.computeLargestBlobRatio(contiguousCar);
        double speckleBlobRatio = ConnectedComponentAnalyzer.computeLargestBlobRatio(scatteredSpeckles);

        assertEquals(0.25, carBlobRatio, 1e-6);
        assertEquals(1.0 / 400.0, speckleBlobRatio, 1e-6);
        assertTrue(carBlobRatio > 50 * speckleBlobRatio);
    }

    @Test
    void nullInputThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> ConnectedComponentAnalyzer.analyze(null));
    }
}
