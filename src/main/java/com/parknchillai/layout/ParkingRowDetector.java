package com.parknchillai.layout;

import com.parknchillai.imageprocessing.EdgeDetector;
import com.parknchillai.imageprocessing.FilterProcessor;
import com.parknchillai.imageprocessing.GrayscaleConverter;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Detects horizontal parking rows within a {@link ParkingBoundary} strictly from
 * visible bay-marking line evidence and Sobel edge profiles (ParkNChill v2.3 Requirement 4).
 *
 * <p>Unlike earlier fallback heuristics, unmarked scenes lacking painted row/divider markings
 * return an empty list rather than inventing synthetic rows.
 */
public final class ParkingRowDetector {

    private static final int EDGE_THRESHOLD = 35;

    private ParkingRowDetector() { /* utility class */ }

    public static List<ParkingRow> detectRows(BufferedImage image, ParkingBoundary boundary) {
        if (image == null) throw new IllegalArgumentException("Image must not be null.");
        if (boundary == null) throw new IllegalArgumentException("Boundary must not be null.");

        BufferedImage gray = GrayscaleConverter.ensureGrayscale(image);
        BufferedImage smoothed = FilterProcessor.applyGaussianFilter(gray, 3);
        BufferedImage edges = EdgeDetector.applySobel(smoothed);
        double bgMean = ParkingBoundaryDetector.computeMeanIntensity(gray);

        int bx = boundary.getX();
        int by = boundary.getY();
        int bw = boundary.getWidth();
        int bh = boundary.getHeight();
        int imgH = image.getHeight();

        double[] rowSignal = new double[bh];
        int totalMarkingLines = 0;
        for (int dy = 0; dy < bh; dy++) {
            int y = by + dy;
            int markingCount = 0;
            int edgeCount = 0;
            for (int dx = 0; dx < bw; dx++) {
                int x = bx + dx;
                int g = gray.getRaster().getSample(x, y, 0);
                if (ParkingBoundaryDetector.isLineMarking(g, bgMean)) {
                    markingCount++;
                }
                if (edges.getRaster().getSample(x, y, 0) >= EDGE_THRESHOLD) {
                    edgeCount++;
                }
            }
            if (markingCount >= 2) {
                totalMarkingLines++;
            }
            rowSignal[dy] = markingCount * 3.0 + (markingCount >= 2 ? edgeCount * 0.4 : 0.0);
        }

        if (totalMarkingLines < Math.max(8, (int) Math.round(bh * 0.08))) {
            return List.of();
        }

        double sumSignal = 0.0;
        for (int i = 0; i < bh; i++) {
            sumSignal += rowSignal[i];
        }
        double meanSignal = sumSignal / Math.max(1, bh);

        double activeCutoff = Math.max(4.0, meanSignal * 0.15);
        List<int[]> rawIntervals = new ArrayList<>();
        int start = -1;
        for (int i = 0; i < bh; i++) {
            boolean active = rowSignal[i] >= activeCutoff;
            if (active && start == -1) {
                start = i;
            } else if (!active && start != -1) {
                rawIntervals.add(new int[]{start, i - 1});
                start = -1;
            }
        }
        if (start != -1) {
            rawIntervals.add(new int[]{start, bh - 1});
        }

        List<int[]> merged = new ArrayList<>();
        for (int[] iv : rawIntervals) {
            if (merged.isEmpty()) {
                merged.add(iv);
            } else {
                int[] prev = merged.get(merged.size() - 1);
                if (iv[0] - prev[1] <= 4) {
                    prev[1] = iv[1];
                } else {
                    merged.add(iv);
                }
            }
        }

        int minRowHeight = Math.max(16, (int) Math.round(imgH * 0.12));
        List<ParkingRow> rows = new ArrayList<>();
        for (int[] iv : merged) {
            int ry = by + iv[0];
            int rh = iv[1] - iv[0 + 0] + 1;
            if (rh >= minRowHeight) {
                rows.add(new ParkingRow(rows.size(), ry, rh, bx, bw));
            }
        }

        return rows;
    }
}
