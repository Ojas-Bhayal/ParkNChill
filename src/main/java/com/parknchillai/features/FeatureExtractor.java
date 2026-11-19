package com.parknchillai.features;

import com.parknchillai.imageprocessing.*;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Extracts the 8-element multi-modal numerical feature vector from a parking-space ROI image.
 *
 * <p>Addresses perimeter stall-line contamination and unimodal background splitting:
 * <ul>
 *   <li>In {@code useAdaptiveThreshold = true} mode, painted perimeter stall divider lines and
 *       faint achromatic bay-number markings are suppressed prior to local integral-image
 *       thresholding so empty bays with painted borders do not produce a 4-sided perimeter halo
 *       ({@code largestBlobRatio ≈ 0.25..0.31 -> OCCUPIED}).</li>
 *   <li>In {@code useAdaptiveThreshold = false} (Otsu) mode, if bright perimeter stall lines
 *       pull the global Otsu threshold above an interior painted or dark vehicle, or if thick
 *       perimeter lines alone exceed the blob threshold on an otherwise empty interior, interior
 *       chrominance/contrast recovery and border-only line suppression prevent both false-occupied
 *       and false-available errors while preserving exact demo-lot extraction compatibility.</li>
 * </ul>
 */
public final class FeatureExtractor {

    /** Pixel intensity below this value is counted as "dark". */
    private static final int DARK_THRESHOLD = 85;

    /** Sobel magnitude above this value is counted as an edge pixel. */
    private static final int EDGE_THRESHOLD = 30;

    /** HSV saturation value (in [0, 255]) above which a pixel is counted as chromatic/colorful. */
    private static final int SATURATION_THRESHOLD = 90;

    private FeatureExtractor() { /* utility class */ }

    public static final class ProcessedRoi {
        private final BufferedImage original;
        private final BufferedImage enhanced;
        private final BufferedImage gray;
        private final BufferedImage smoothed;
        private final BufferedImage binary;
        private final BufferedImage closed;
        private final BufferedImage edges;
        private final BufferedImage saturation;
        private final BufferedImage largestBlobMask;
        private final FeatureVector features;

        public ProcessedRoi(BufferedImage gray,
                            BufferedImage smoothed,
                            BufferedImage binary,
                            BufferedImage closed,
                            BufferedImage edges,
                            FeatureVector features) {
            this(null, null, gray, smoothed, binary, closed, edges, null, null, features);
        }

        public ProcessedRoi(BufferedImage gray,
                            BufferedImage smoothed,
                            BufferedImage binary,
                            BufferedImage closed,
                            BufferedImage edges,
                            BufferedImage saturation,
                            BufferedImage largestBlobMask,
                            FeatureVector features) {
            this(null, null, gray, smoothed, binary, closed, edges, saturation, largestBlobMask, features);
        }

        public ProcessedRoi(BufferedImage original,
                            BufferedImage enhanced,
                            BufferedImage gray,
                            BufferedImage smoothed,
                            BufferedImage binary,
                            BufferedImage closed,
                            BufferedImage edges,
                            BufferedImage saturation,
                            BufferedImage largestBlobMask,
                            FeatureVector features) {
            this.original = original;
            this.enhanced = enhanced;
            this.gray = gray;
            this.smoothed = smoothed;
            this.binary = binary;
            this.closed = closed;
            this.edges = edges;
            this.saturation = saturation;
            this.largestBlobMask = largestBlobMask;
            this.features = features;
        }

        public BufferedImage getOriginal()        { return original; }
        public BufferedImage getEnhanced()        { return enhanced; }
        public BufferedImage getGray()            { return gray; }
        public BufferedImage getSmoothed()        { return smoothed; }
        public BufferedImage getBinary()          { return binary; }
        public BufferedImage getClosed()          { return closed; }
        public BufferedImage getEdges()           { return edges; }
        public BufferedImage getSaturation()      { return saturation; }
        public BufferedImage getLargestBlobMask() { return largestBlobMask; }
        public FeatureVector getFeatures()        { return features; }
    }

    public static FeatureVector extract(BufferedImage roi) {
        return processWithStages(roi, false).getFeatures();
    }

    public static FeatureVector extract(BufferedImage roi, boolean useAdaptiveThreshold) {
        return processWithStages(roi, useAdaptiveThreshold).getFeatures();
    }

    public static FeatureVector extract(BufferedImage rawRoi, BufferedImage enhancedRoi, boolean useAdaptiveThreshold) {
        return processWithStages(rawRoi, enhancedRoi, useAdaptiveThreshold).getFeatures();
    }

    public static ProcessedRoi processWithStages(BufferedImage roi) {
        return processWithStages(roi, false);
    }

    public static ProcessedRoi processWithStages(BufferedImage roi, boolean useAdaptiveThreshold) {
        if (roi == null) throw new IllegalArgumentException("ROI must not be null.");
        BufferedImage enhanced = ImageEnhancer.autoEnhance(roi);
        return processWithStages(roi, enhanced, useAdaptiveThreshold);
    }

    public static ProcessedRoi processWithStages(BufferedImage rawRoi, BufferedImage enhancedRoi, boolean useAdaptiveThreshold) {
        if (enhancedRoi == null) throw new IllegalArgumentException("ROI must not be null.");
        if (enhancedRoi.getWidth() == 0 || enhancedRoi.getHeight() == 0) {
            throw new IllegalArgumentException("ROI must have positive dimensions.");
        }

        BufferedImage original = (rawRoi != null) ? rawRoi : enhancedRoi;

        // CANONICAL POLICY (ivp-enhance-v1):
        // "What is displayed is what is computed."
        // All 8 classical IVP features and intermediate representations are computed
        // strictly from the ENHANCED slot crop.
        BufferedImage gray = GrayscaleConverter.toGrayscale(enhancedRoi);
        BufferedImage saturation = ColorSpaceConverter.toSaturationMap(enhancedRoi);

        // Step 2 — Gaussian smoothing (5×5) to suppress high-frequency noise before thresholding
        BufferedImage smoothed = FilterProcessor.applyGaussianFilter(gray, 5);

        // Step 3 — Binary segmentation (Otsu or Border-Line-Suppressed Chrominance-Fused Adaptive)
        BufferedImage binary = segmentForeground(smoothed, saturation, useAdaptiveThreshold);

        // Step 4 — Morphological closing (3×3) to consolidate vehicle blobs
        BufferedImage closed = MorphologyProcessor.close(binary, 3);

        // Step 5 — Connected-Component Analysis (8-connected BFS flood-fill on closed mask)
        ConnectedComponentAnalyzer.ComponentSummary ccSummary = ConnectedComponentAnalyzer.analyze(closed);
        BufferedImage largestBlobMask = ccSummary.getLargestBlobMask();

        // Step 6 — Sobel edge detection on the smoothed grayscale
        BufferedImage edges = EdgeDetector.applySobel(smoothed);

        // --- Feature computation (8 features) from enhanced stages ---
        double meanIntensity    = computeMeanIntensity(smoothed);
        double foregroundRatio  = computeForegroundRatio(binary);
        double darkPixelRatio   = computeDarkPixelRatio(smoothed);
        double edgeDensity      = computeEdgeDensity(edges);
        double objectArea       = computeForegroundRatio(closed);
        double meanSaturation   = computeMeanIntensity(saturation);
        double saturationRatio  = computeSaturationRatio(saturation);
        double largestBlobRatio = ccSummary.getLargestBlobRatio();

        FeatureVector features = new FeatureVector(new double[]{
            meanIntensity,
            foregroundRatio,
            darkPixelRatio,
            edgeDensity,
            objectArea,
            meanSaturation,
            saturationRatio,
            largestBlobRatio
        });

        return new ProcessedRoi(original, enhancedRoi, gray, smoothed, binary, closed, edges, saturation, largestBlobMask, features);
    }

    private static BufferedImage segmentForeground(BufferedImage smoothed,
                                                   BufferedImage saturation,
                                                   boolean useAdaptiveThreshold) {
        int w = smoothed.getWidth();
        int h = smoothed.getHeight();
        int marginX = Math.max(2, Math.min(8, w / 10));
        int marginY = Math.max(2, Math.min(8, h / 10));

        if (w < 12 || h < 12) {
            if (useAdaptiveThreshold) {
                BufferedImage adaptiveGray = ThresholdProcessor.applyAdaptiveThreshold(smoothed);
                return fuseAdaptiveWithSaturation(adaptiveGray, saturation, SATURATION_THRESHOLD);
            }
            return ThresholdProcessor.applyOtsuThreshold(smoothed);
        }

        int bgMedian = estimatePavementBackgroundMedian(smoothed, saturation, marginX, marginY);

        if (useAdaptiveThreshold) {
            BufferedImage cleaned = suppressBorderStallLines(smoothed, saturation, marginX, marginY, bgMedian);
            BufferedImage adaptiveGray = ThresholdProcessor.applyAdaptiveThreshold(cleaned);
            BufferedImage fused = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
            for (int y = marginY; y < h - marginY; y++) {
                for (int x = marginX; x < w - marginX; x++) {
                    int b = adaptiveGray.getRaster().getSample(x, y, 0);
                    int s = saturation.getRaster().getSample(x, y, 0);
                    if (b == 255 || s > SATURATION_THRESHOLD) {
                        fused.getRaster().setSample(x, y, 0, 255);
                    }
                }
            }
            return fused;
        }

        BufferedImage otsu = ThresholdProcessor.applyOtsuThreshold(smoothed);
        int white = 0;
        int interiorWhite = 0;
        int interiorTotal = 0;
        int satPixels = 0;
        int devPixels = 0;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                boolean inBorder = (x < marginX || x >= w - marginX || y < marginY || y >= h - marginY);
                int b = otsu.getRaster().getSample(x, y, 0);
                if (b == 255) white++;
                if (!inBorder) {
                    interiorTotal++;
                    if (b == 255) interiorWhite++;
                    int s = saturation.getRaster().getSample(x, y, 0);
                    int g = smoothed.getRaster().getSample(x, y, 0);
                    if (s > SATURATION_THRESHOLD) satPixels++;
                    if (Math.abs(g - bgMedian) > 18) devPixels++;
                }
            }
        }

        double fgRatio = (double) white / (w * h);
        double intFgRatio = interiorTotal > 0 ? (double) interiorWhite / interiorTotal : 0.0;
        double intSatRatio = interiorTotal > 0 ? (double) satPixels / interiorTotal : 0.0;
        double intDevRatio = interiorTotal > 0 ? (double) devPixels / interiorTotal : 0.0;

        // Guard 1: Bright painted border lines pulled Otsu threshold above an interior vehicle
        if (fgRatio < 0.12 && (intSatRatio >= 0.18 || intDevRatio >= 0.22)) {
            BufferedImage recovered = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
            for (int y = marginY; y < h - marginY; y++) {
                for (int x = marginX; x < w - marginX; x++) {
                    int s = saturation.getRaster().getSample(x, y, 0);
                    int g = smoothed.getRaster().getSample(x, y, 0);
                    if (s > SATURATION_THRESHOLD || Math.abs(g - bgMedian) > 18) {
                        recovered.getRaster().setSample(x, y, 0, 255);
                    }
                }
            }
            return recovered;
        }

        // Guard 2: Thick perimeter stall lines exceeded 7% of ROI while interior bay is empty
        if (fgRatio > 0.07 && intFgRatio < 0.035 && intSatRatio < 0.05 && intDevRatio < 0.08) {
            BufferedImage interiorOnly = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
            for (int y = marginY; y < h - marginY; y++) {
                for (int x = marginX; x < w - marginX; x++) {
                    interiorOnly.getRaster().setSample(x, y, 0, otsu.getRaster().getSample(x, y, 0));
                }
            }
            return interiorOnly;
        }

        return otsu;
    }

    private static int estimatePavementBackgroundMedian(BufferedImage smoothed,
                                                        BufferedImage saturation,
                                                        int marginX,
                                                        int marginY) {
        int w = smoothed.getWidth();
        int h = smoothed.getHeight();
        List<Integer> ringVals = new ArrayList<>();
        for (int y = marginY; y < h - marginY; y++) {
            for (int x = marginX; x < w - marginX; x++) {
                boolean nearEdge = (x < marginX + 6 || x >= w - marginX - 6
                    || y < marginY + 6 || y >= h - marginY - 6);
                if (nearEdge && saturation.getRaster().getSample(x, y, 0) < 45) {
                    ringVals.add(smoothed.getRaster().getSample(x, y, 0));
                }
            }
        }
        if (ringVals.isEmpty()) {
            for (int y = marginY; y < h - marginY; y++) {
                for (int x = marginX; x < w - marginX; x++) {
                    ringVals.add(smoothed.getRaster().getSample(x, y, 0));
                }
            }
        }
        if (ringVals.isEmpty()) return 60;
        Collections.sort(ringVals);
        return ringVals.get(ringVals.size() / 2);
    }

    private static BufferedImage suppressBorderStallLines(BufferedImage smoothed,
                                                          BufferedImage saturation,
                                                          int marginX,
                                                          int marginY,
                                                          int bgMedian) {
        int w = smoothed.getWidth();
        int h = smoothed.getHeight();
        BufferedImage cleaned = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int g = smoothed.getRaster().getSample(x, y, 0);
                int s = saturation.getRaster().getSample(x, y, 0);
                boolean inBorder = (x < marginX || x >= w - marginX || y < marginY || y >= h - marginY);
                boolean isFaintMarking = (y >= (h * 3) / 4 && s < 35 && g > bgMedian + 8 && g < bgMedian + 65);
                if ((inBorder && s < 60) || isFaintMarking) {
                    cleaned.getRaster().setSample(x, y, 0, bgMedian);
                } else {
                    cleaned.getRaster().setSample(x, y, 0, g);
                }
            }
        }
        return cleaned;
    }

    /**
     * Fuses the local adaptive intensity binary mask with high-saturation chromatic vehicle pixels
     * ({@code S > satThreshold}) so painted car roofs remain contiguous foreground blobs while
     * achromatic pavement shadows remain 0.
     */
    static BufferedImage fuseAdaptiveWithSaturation(BufferedImage adaptiveBinary,
                                                    BufferedImage saturationMap,
                                                    int satThreshold) {
        int w = adaptiveBinary.getWidth();
        int h = adaptiveBinary.getHeight();
        BufferedImage fused = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int b = adaptiveBinary.getRaster().getSample(x, y, 0);
                int s = saturationMap.getRaster().getSample(x, y, 0);
                if (b == 255 || s > satThreshold) {
                    fused.getRaster().setSample(x, y, 0, 255);
                }
            }
        }
        return fused;
    }

    public static double computeMeanIntensity(BufferedImage gray) {
        long sum = 0;
        int total = gray.getWidth() * gray.getHeight();
        for (int y = 0; y < gray.getHeight(); y++) {
            for (int x = 0; x < gray.getWidth(); x++) {
                sum += gray.getRaster().getSample(x, y, 0);
            }
        }
        return (double) sum / total;
    }

    static double computeForegroundRatio(BufferedImage binary) {
        int white = 0;
        int total = binary.getWidth() * binary.getHeight();
        for (int y = 0; y < binary.getHeight(); y++) {
            for (int x = 0; x < binary.getWidth(); x++) {
                if (binary.getRaster().getSample(x, y, 0) == 255) white++;
            }
        }
        return (double) white / total;
    }

    static double computeDarkPixelRatio(BufferedImage gray) {
        int dark = 0;
        int total = gray.getWidth() * gray.getHeight();
        for (int y = 0; y < gray.getHeight(); y++) {
            for (int x = 0; x < gray.getWidth(); x++) {
                if (gray.getRaster().getSample(x, y, 0) < DARK_THRESHOLD) dark++;
            }
        }
        return (double) dark / total;
    }

    static double computeEdgeDensity(BufferedImage edgeMag) {
        int edgePixels = 0;
        int total = edgeMag.getWidth() * edgeMag.getHeight();
        for (int y = 0; y < edgeMag.getHeight(); y++) {
            for (int x = 0; x < edgeMag.getWidth(); x++) {
                if (edgeMag.getRaster().getSample(x, y, 0) > EDGE_THRESHOLD) edgePixels++;
            }
        }
        return (double) edgePixels / total;
    }

    static double computeSaturationRatio(BufferedImage saturationMap) {
        int colorful = 0;
        int total = saturationMap.getWidth() * saturationMap.getHeight();
        for (int y = 0; y < saturationMap.getHeight(); y++) {
            for (int x = 0; x < saturationMap.getWidth(); x++) {
                if (saturationMap.getRaster().getSample(x, y, 0) > SATURATION_THRESHOLD) colorful++;
            }
        }
        return (double) colorful / total;
    }
}
