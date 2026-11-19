package com.parknchillai.layout;

import com.parknchillai.imageprocessing.EdgeDetector;
import com.parknchillai.imageprocessing.FilterProcessor;
import com.parknchillai.imageprocessing.GrayscaleConverter;

import java.awt.image.BufferedImage;

public final class ParkingBoundaryDetector {

    private static final int EDGE_ENERGY_THRESHOLD = 28;
    private static final int BRIGHT_LINE_THRESHOLD = 195;

    private ParkingBoundaryDetector() { /* utility class */ }

    public static ParkingBoundary detectBoundary(BufferedImage image) {
        if (image == null) {
            throw new IllegalArgumentException("Input image must not be null.");
        }
        int w = image.getWidth();
        int h = image.getHeight();
        if (w < 10 || h < 10) {
            throw new IllegalArgumentException("Image is too small for boundary detection: " + w + "x" + h);
        }

        BufferedImage gray = GrayscaleConverter.ensureGrayscale(image);
        BufferedImage smoothed = FilterProcessor.applyGaussianFilter(gray, 3);
        BufferedImage edges = EdgeDetector.applySobel(smoothed);
        double bgMean = computeMeanIntensity(gray);

        int[] colActivity = new int[w];
        int[] rowActivity = new int[h];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int g = gray.getRaster().getSample(x, y, 0);
                int e = edges.getRaster().getSample(x, y, 0);
                if (isLineMarking(g, bgMean) || e >= EDGE_ENERGY_THRESHOLD) {
                    colActivity[x]++;
                    rowActivity[y]++;
                }
            }
        }

        int minColHits = Math.max(2, (int) Math.round(h * 0.03));
        int minRowHits = Math.max(2, (int) Math.round(w * 0.03));

        int left = 0;
        while (left < w - 1 && colActivity[left] < minColHits) left++;

        int right = w - 1;
        while (right > left && colActivity[right] < minColHits) right--;

        int top = 0;
        while (top < h - 1 && rowActivity[top] < minRowHits) top++;

        int bottom = h - 1;
        while (bottom > top && rowActivity[bottom] < minRowHits) bottom--;

        int bw = right - left + 1;
        int bh = bottom - top + 1;

        if (bw < w * 0.25 || bh < h * 0.25) {
            int marginX = Math.max(2, (int) Math.round(w * 0.03));
            int marginY = Math.max(2, (int) Math.round(h * 0.03));
            return new ParkingBoundary(
                marginX,
                marginY,
                Math.max(1, w - 2 * marginX),
                Math.max(1, h - 2 * marginY)
            );
        }

        return new ParkingBoundary(left, top, bw, bh);
    }

    /**
     * Computes the global mean grayscale intensity across the image to distinguish
     * dark-asphalt parking lots (with bright white/yellow stall markings) from
     * light-concrete parking lots (with dark painted stall markings).
     */
    static double computeMeanIntensity(BufferedImage gray) {
        long sum = 0;
        int w = gray.getWidth();
        int h = gray.getHeight();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                sum += gray.getRaster().getSample(x, y, 0);
            }
        }
        return (double) sum / Math.max(1, w * h);
    }

    /**
     * Returns {@code true} if {@code grayVal} represents a high-contrast painted stall line
     * relative to the background pavement intensity {@code bgMean}.
     */
    static boolean isLineMarking(int grayVal, double bgMean) {
        if (bgMean < 145.0) {
            return grayVal >= BRIGHT_LINE_THRESHOLD;
        }
        return Math.abs(grayVal - bgMean) >= 45.0;
    }
}
