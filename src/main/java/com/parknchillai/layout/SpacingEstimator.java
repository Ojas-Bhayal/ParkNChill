package com.parknchillai.layout;

import com.parknchillai.imageprocessing.GrayscaleConverter;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Detects individual parking-bay horizontal intervals within a {@link ParkingRow}
 * from visible stall markings (closed bay outlines or parallel vertical/angled divider lines).
 *
 * <p>For unmarked scenes lacking verifiable bay divider lines, {@link #detectBayIntervals}
 * returns an empty list rather than fabricating uniform empty bays (ParkNChill v2.3 Requirement 4).
 */
public final class SpacingEstimator {

    private SpacingEstimator() { /* utility class */ }

    public static List<int[]> detectBayIntervals(BufferedImage image, ParkingRow row) {
        if (image == null) throw new IllegalArgumentException("Image must not be null.");
        if (row == null) throw new IllegalArgumentException("ParkingRow must not be null.");

        BufferedImage gray = GrayscaleConverter.ensureGrayscale(image);
        double bgMean = ParkingBoundaryDetector.computeMeanIntensity(gray);
        int rx = row.getXStart();
        int ry = row.getY();
        int rw = row.getWidth();
        int rh = row.getHeight();

        boolean[] hasStallBoxColumn = new boolean[rw];
        int[] colMarkingCounts = new int[rw];
        int topBorderYEnd = Math.min(ry + rh - 1, ry + 4);
        int botBorderYStart = Math.max(ry, ry + rh - 5);

        for (int dx = 0; dx < rw; dx++) {
            int x = rx + dx;
            int markingCount = 0;
            boolean topMarking = false;
            boolean botMarking = false;

            for (int y = ry; y < ry + rh; y++) {
                int g = gray.getRaster().getSample(x, y, 0);
                if (ParkingBoundaryDetector.isLineMarking(g, bgMean)) {
                    markingCount++;
                    if (y <= topBorderYEnd) topMarking = true;
                    if (y >= botBorderYStart) botMarking = true;
                }
            }
            colMarkingCounts[dx] = markingCount;
            hasStallBoxColumn[dx] = (topMarking && botMarking) || (markingCount >= rh * 0.65);
        }

        List<int[]> boxIntervals = new ArrayList<>();
        int start = -1;
        for (int dx = 0; dx < rw; dx++) {
            if (hasStallBoxColumn[dx] && start == -1) {
                start = dx;
            } else if (!hasStallBoxColumn[dx] && start != -1) {
                boxIntervals.add(new int[]{start, dx - 1});
                start = -1;
            }
        }
        if (start != -1) {
            boxIntervals.add(new int[]{start, rw - 1});
        }

        List<int[]> mergedBoxes = new ArrayList<>();
        for (int[] iv : boxIntervals) {
            if (mergedBoxes.isEmpty()) {
                mergedBoxes.add(iv);
            } else {
                int[] prev = mergedBoxes.get(mergedBoxes.size() - 1);
                if (iv[0] - prev[1] <= 3) {
                    prev[1] = iv[1];
                } else {
                    mergedBoxes.add(iv);
                }
            }
        }

        int minBayW = Math.max(16, (int) Math.round(rh * 0.45));
        int maxBayW = Math.max(minBayW + 10, (int) Math.round(rh * 1.45));

        List<int[]> validBays = new ArrayList<>();
        for (int[] iv : mergedBoxes) {
            int w = iv[1] - iv[0] + 1;
            if (w >= minBayW && w <= maxBayW) {
                validBays.add(new int[]{rx + iv[0], w});
            }
        }

        if (validBays.size() >= 2) {
            return validBays;
        }

        // Secondary evidence path: Open-ended parallel vertical/angled stall divider lines
        // (thin line segments 1..10 px wide spanning >= 45% of row height, repeated >= 3 times)
        List<Integer> dividerCenters = detectParallelDividerLineX(colMarkingCounts, rw, rh);
        if (dividerCenters.size() >= 3) {
            List<int[]> dividerBays = new ArrayList<>();
            for (int i = 0; i < dividerCenters.size() - 1; i++) {
                int x0 = rx + dividerCenters.get(i);
                int x1 = rx + dividerCenters.get(i + 1);
                int w = x1 - x0;
                if (w >= minBayW && w <= maxBayW) {
                    dividerBays.add(new int[]{x0, w});
                }
            }
            if (dividerBays.size() >= 2) {
                return dividerBays;
            }
        }

        // No reliable bay-marking geometry found -> do NOT invent uniform empty bays
        return List.of();
    }

    /**
     * Identifies thin vertical/angled stall divider line centers (width 1..10 px,
     * vertical marking coverage {@code >= 45%} of row height) within a row.
     */
    static List<Integer> detectParallelDividerLineX(int[] colMarkingCounts, int rw, int rh) {
        int minDividerPixels = Math.max(10, (int) Math.round(rh * 0.45));
        List<Integer> centers = new ArrayList<>();
        int start = -1;
        for (int dx = 0; dx < rw; dx++) {
            boolean isLineCol = colMarkingCounts[dx] >= minDividerPixels;
            if (isLineCol && start == -1) {
                start = dx;
            } else if (!isLineCol && start != -1) {
                int span = dx - start;
                if (span >= 1 && span <= 10) {
                    centers.add(start + span / 2);
                }
                start = -1;
            }
        }
        if (start != -1) {
            int span = rw - start;
            if (span >= 1 && span <= 10) {
                centers.add(start + span / 2);
            }
        }
        return centers;
    }

    public static int estimateSlotWidth(BufferedImage image, ParkingRow row) {
        if (image == null) throw new IllegalArgumentException("Image must not be null.");
        if (row == null) throw new IllegalArgumentException("ParkingRow must not be null.");

        BufferedImage gray = GrayscaleConverter.ensureGrayscale(image);
        double bgMean = ParkingBoundaryDetector.computeMeanIntensity(gray);
        int rx = row.getXStart();
        int ry = row.getY();
        int rw = row.getWidth();
        int rh = row.getHeight();

        double[] profile = new double[rw];
        double mean = 0.0;
        for (int dx = 0; dx < rw; dx++) {
            int x = rx + dx;
            double colEnergy = 0.0;
            for (int y = ry; y < ry + rh; y++) {
                int g = gray.getRaster().getSample(x, y, 0);
                int prevG = (dx > 0) ? gray.getRaster().getSample(x - 1, y, 0) : g;
                colEnergy += Math.abs(g - prevG);
                if (ParkingBoundaryDetector.isLineMarking(g, bgMean)) {
                    colEnergy += 32.0;
                }
            }
            profile[dx] = colEnergy;
            mean += colEnergy;
        }
        mean /= Math.max(1, rw);

        int minLag = Math.max(18, (int) Math.round(rh * 0.55));
        int maxLag = Math.min(rw / 2, Math.max(minLag + 10, (int) Math.round(rh * 1.25)));

        int bestLag = Math.max(20, (int) Math.round(rh * 0.80));
        double bestCorr = Double.NEGATIVE_INFINITY;

        for (int lag = minLag; lag <= maxLag; lag++) {
            double corr = 0.0;
            int count = rw - lag;
            if (count <= 0) break;
            for (int i = 0; i < count; i++) {
                corr += (profile[i] - mean) * (profile[i + lag] - mean);
            }
            corr /= count;
            if (corr > bestCorr) {
                bestCorr = corr;
                bestLag = lag;
            }
        }

        return Math.max(15, Math.min(rw, bestLag));
    }
}
