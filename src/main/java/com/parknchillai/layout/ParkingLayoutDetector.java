package com.parknchillai.layout;

import com.parknchillai.imageprocessing.EdgeDetector;
import com.parknchillai.imageprocessing.FilterProcessor;
import com.parknchillai.imageprocessing.GrayscaleConverter;
import com.parknchillai.parking.ParkingSlot;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Locale;

/**
 * Automatic parking-space localization pipeline (ParkNChill v2.3 Requirement 4):
 * <ol>
 *   <li>Grayscale &amp; contrast normalization ({@link ParkingBoundaryDetector#computeMeanIntensity})</li>
 *   <li>Gaussian smoothing &amp; Sobel edge/line detection</li>
 *   <li>Hough-style directional line orientation &amp; segment counting ({@link #estimateMarkingGeometry})</li>
 *   <li>Parallel-line &amp; horizontal row grouping ({@link ParkingRowDetector})</li>
 *   <li>Bay interval extraction &amp; spacing estimation ({@link SpacingEstimator})</li>
 *   <li>Overlapping proposal suppression ({@link SlotGenerator#suppressOverlappingProposals})</li>
 *   <li>Explicit uncertainty reporting on unmarked scenes (never inventing empty parking spaces)</li>
 * </ol>
 */
public final class ParkingLayoutDetector {

    private ParkingLayoutDetector() { /* utility class */ }

    public record MarkingGeometryStats(
            double dominantOrientationDeg,
            int lineSegmentCount
    ) {}

    public static LayoutDetectionResult detectLayout(BufferedImage image) {
        if (image == null) {
            throw new IllegalArgumentException("Input image must not be null.");
        }
        ParkingBoundary boundary = ParkingBoundaryDetector.detectBoundary(image);
        MarkingGeometryStats geomStats = estimateMarkingGeometry(image, boundary);
        List<ParkingRow> rows = ParkingRowDetector.detectRows(image, boundary);
        int estWidth = rows.isEmpty()
            ? 0
            : SpacingEstimator.estimateSlotWidth(image, rows.get(0));
        List<ParkingSlot> slots = rows.isEmpty()
            ? List.of()
            : SlotGenerator.generateSlots(image, boundary, rows);

        boolean reliable = !slots.isEmpty() && geomStats.lineSegmentCount() >= 4;
        if (!reliable) {
            return new LayoutDetectionResult(
                boundary,
                List.of(),
                0,
                List.of(),
                false,
                geomStats.dominantOrientationDeg(),
                geomStats.lineSegmentCount(),
                "Parking-slot localization is uncertain: insufficient visible bay markings in image (0 spaces inferred; no empty spaces invented)."
            );
        }

        String summary = String.format(Locale.ROOT,
            "Localized %d parking space(s) across %d row(s) from visible bay markings (est. bay width=%d px, orientation=%.1f deg, line segments=%d).",
            slots.size(), rows.size(), estWidth, geomStats.dominantOrientationDeg(), geomStats.lineSegmentCount());

        return new LayoutDetectionResult(
            boundary,
            rows,
            estWidth,
            slots,
            true,
            geomStats.dominantOrientationDeg(),
            geomStats.lineSegmentCount(),
            summary
        );
    }

    /**
     * Estimates the dominant parking-marking line orientation (in degrees {@code [0, 180)})
     * and counts coherent high-contrast line segments using a quantized Hough-angle histogram
     * over marking-supported Sobel edge pixels.
     */
    public static MarkingGeometryStats estimateMarkingGeometry(BufferedImage image, ParkingBoundary boundary) {
        BufferedImage gray = GrayscaleConverter.ensureGrayscale(image);
        BufferedImage smoothed = FilterProcessor.applyGaussianFilter(gray, 3);
        BufferedImage edges = EdgeDetector.applySobel(smoothed);
        double bgMean = ParkingBoundaryDetector.computeMeanIntensity(gray);

        int bx = Math.max(1, boundary.getX());
        int by = Math.max(1, boundary.getY());
        int xEnd = Math.min(image.getWidth() - 1, boundary.getX() + boundary.getWidth());
        int yEnd = Math.min(image.getHeight() - 1, boundary.getY() + boundary.getHeight());

        double[] angleBins = new double[36]; // 5-degree bins across [0..180)
        int lineRunCount = 0;

        for (int y = by; y < yEnd; y++) {
            int horizRun = 0;
            for (int x = bx; x < xEnd; x++) {
                int g = gray.getRaster().getSample(x, y, 0);
                int e = edges.getRaster().getSample(x, y, 0);
                if (ParkingBoundaryDetector.isLineMarking(g, bgMean)) {
                    horizRun++;
                    if (e >= 28) {
                        int gx = smoothed.getRaster().getSample(Math.min(image.getWidth() - 1, x + 1), y, 0)
                               - smoothed.getRaster().getSample(Math.max(0, x - 1), y, 0);
                        int gy = smoothed.getRaster().getSample(x, Math.min(image.getHeight() - 1, y + 1), 0)
                               - smoothed.getRaster().getSample(x, Math.max(0, y - 1), 0);
                        // Line direction is perpendicular to gradient (gx, gy)
                        double lineAngle = Math.toDegrees(Math.atan2(gx, -gy));
                        if (lineAngle < 0) lineAngle += 180.0;
                        if (lineAngle >= 180.0) lineAngle -= 180.0;
                        int bin = Math.min(35, (int) (lineAngle / 5.0));
                        angleBins[bin] += e;
                    }
                } else {
                    if (horizRun >= 12) lineRunCount++;
                    horizRun = 0;
                }
            }
            if (horizRun >= 12) lineRunCount++;
        }

        // Also count vertical line runs
        for (int x = bx; x < xEnd; x++) {
            int vertRun = 0;
            for (int y = by; y < yEnd; y++) {
                int g = gray.getRaster().getSample(x, y, 0);
                if (ParkingBoundaryDetector.isLineMarking(g, bgMean)) {
                    vertRun++;
                } else {
                    if (vertRun >= 12) lineRunCount++;
                    vertRun = 0;
                }
            }
            if (vertRun >= 12) lineRunCount++;
        }

        int bestBin = 18; // default 90.0 deg (vertical divider lines)
        double bestWeight = -1.0;
        for (int b = 0; b < angleBins.length; b++) {
            if (angleBins[b] > bestWeight) {
                bestWeight = angleBins[b];
                bestBin = b;
            }
        }
        double dominantDeg = bestWeight > 0.0 ? (bestBin * 5.0 + 2.5) : 90.0;
        return new MarkingGeometryStats(dominantDeg, lineRunCount);
    }
}
