package com.parknchillai.features;

import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

class FeatureExtractorTest {

    @Test
    void meanIntensityOfUniformDarkImage() {
        BufferedImage img = uniformGray(10, 10, 50);
        double mean = FeatureExtractor.computeMeanIntensity(img);
        assertEquals(50.0, mean, 0.5, "Mean intensity should be 50 for uniform-50 image");
    }

    @Test
    void foregroundRatioOfAllWhiteBinaryImage() {
        BufferedImage white = uniformGray(5, 5, 255);
        assertEquals(1.0, FeatureExtractor.computeForegroundRatio(white), 1e-6);
    }

    @Test
    void foregroundRatioOfAllBlackBinaryImage() {
        BufferedImage black = uniformGray(5, 5, 0);
        assertEquals(0.0, FeatureExtractor.computeForegroundRatio(black), 1e-6);
    }

    @Test
    void darkPixelRatioOfBrightImageIsZero() {
        BufferedImage bright = uniformGray(10, 10, 200);
        assertEquals(0.0, FeatureExtractor.computeDarkPixelRatio(bright), 1e-6);
    }

    @Test
    void extractReturnsVectorWithCorrectCountAndIntermediateDimensionsMatchRoi() {
        BufferedImage roi = new BufferedImage(34, 48, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 48; y++)
            for (int x = 0; x < 34; x++)
                roi.setRGB(x, y, (100 << 16) | (80 << 8) | 60);

        FeatureExtractor.ProcessedRoi processed = FeatureExtractor.processWithStages(roi, false);
        FeatureVector fv = processed.getFeatures();
        assertEquals(FeatureVector.FEATURE_COUNT, fv.toArray().length);
        assertTrue(fv.hasAllValuesWithinValidBounds());

        BufferedImage[] stages = {
            processed.getGray(),
            processed.getSaturation(),
            processed.getSmoothed(),
            processed.getBinary(),
            processed.getClosed(),
            processed.getLargestBlobMask(),
            processed.getEdges()
        };
        for (BufferedImage stage : stages) {
            assertNotNull(stage);
            assertEquals(34, stage.getWidth(), "Intermediate stage width must match input ROI");
            assertEquals(48, stage.getHeight(), "Intermediate stage height must match input ROI");
        }
    }

    @Test
    void saturationThresholdConsistencyMatchesSpecAbove90() {
        // Authoritative spec in FeatureVector.SPECS[6] and FeatureExtractor.computeSaturationRatio must both use S > 90
        FeatureVector.FeatureSpec satSpec = FeatureVector.SPECS.get(FeatureVector.IDX_SATURATION_RATIO);
        assertTrue(satSpec.displayName().contains("S > 90"));
        assertTrue(satSpec.formula().contains("90"));

        BufferedImage satMap = new BufferedImage(10, 10, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 10; x++) {
                // Top half S = 80 (<= 90 -> not counted), bottom half S = 120 (> 90 -> counted)
                satMap.getRaster().setSample(x, y, 0, y < 5 ? 80 : 120);
            }
        }
        assertEquals(0.50, FeatureExtractor.computeSaturationRatio(satMap), 1e-6);
    }

    @Test
    void adaptiveThresholdModePreservesUniformColouredVehicleBlobWhileSuppressingShadow() {
        BufferedImage roi = new BufferedImage(80, 100, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 100; y++) {
            for (int x = 0; x < 80; x++) {
                int v = 40 + (x * 50) / 80;
                roi.setRGB(x, y, (v << 16) | (v << 8) | v);
            }
        }
        Graphics2D g = roi.createGraphics();
        g.setColor(new Color(200, 30, 35));
        g.fillRect(20, 25, 40, 50);
        g.dispose();

        FeatureVector adaptiveFv = FeatureExtractor.extract(roi, true);
        assertTrue(adaptiveFv.getLargestBlobRatio() >= 0.24,
            "Adaptive+HSV fusion must preserve solid coloured car roof blob >= 0.24; got " + adaptiveFv.getLargestBlobRatio());
    }

    @Test
    void adaptiveAndOtsuModesBothExtractContiguousBlobForDarkMatteBlackVehicle() {
        BufferedImage roi = new BufferedImage(80, 100, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = roi.createGraphics();
        g.setColor(new Color(60, 62, 66));
        g.fillRect(0, 0, 80, 100);
        g.setColor(new Color(25, 27, 30));
        g.fillRoundRect(12, 12, 56, 76, 12, 12);
        g.setColor(new Color(15, 18, 22));
        g.fillRect(18, 24, 44, 15);
        g.dispose();

        FeatureVector otsuFv = FeatureExtractor.extract(roi, false);
        FeatureVector adaptiveFv = FeatureExtractor.extract(roi, true);

        assertTrue(otsuFv.getLargestBlobRatio() >= 0.22,
            "Otsu mode should extract contiguous blob >= 0.22 for matte-black vehicle; got " + otsuFv.getLargestBlobRatio());
        assertTrue(adaptiveFv.getLargestBlobRatio() >= 0.22,
            "Two-sided Adaptive mode must also extract contiguous blob >= 0.22 for matte-black vehicle; got " + adaptiveFv.getLargestBlobRatio());
    }

    @Test
    void extractReturnsNoNaN() {
        BufferedImage roi = new BufferedImage(15, 15, BufferedImage.TYPE_INT_RGB);
        FeatureVector fv = FeatureExtractor.extract(roi);
        for (double v : fv.toArray()) {
            assertFalse(Double.isNaN(v), "Feature value must not be NaN");
        }
    }


    @Test
    void processingShowsEnhancedRoiAndLeavesOriginalPixelsUnchanged() {
        BufferedImage roi = new BufferedImage(64, 40, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < roi.getHeight(); y++) {
            for (int x = 0; x < roi.getWidth(); x++) {
                int value = 65 + (x % 16) * 4;
                roi.setRGB(x, y, (value << 16) | (value << 8) | value);
            }
        }
        int originalPixel = roi.getRGB(4, 4);
        FeatureExtractor.ProcessedRoi result = FeatureExtractor.processWithStages(roi, false);
        assertNotNull(result.getOriginal());
        assertNotNull(result.getEnhanced());
        assertEquals(roi.getWidth(), result.getEnhanced().getWidth());
        assertEquals(roi.getHeight(), result.getEnhanced().getHeight());
        assertEquals(originalPixel, roi.getRGB(4, 4), "Feature extraction must not mutate the original ROI");
        assertNotEquals(originalPixel, result.getEnhanced().getRGB(4, 4), "A low-contrast ROI should show a real enhancement");
    }

    @Test
    void nullRoiThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> FeatureExtractor.extract(null));
    }

    private static BufferedImage uniformGray(int w, int h, int intensity) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                img.getRaster().setSample(x, y, 0, intensity);
        return img;
    }
}
