package com.parknchillai.parking;

import com.parknchillai.imageprocessing.PerspectiveTransformer;

import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Extracts consistent ROI sub-images from a full parking-area image according to
 * a {@link ParkingSlot}'s geometry (ParkNChill v2.3 Requirement 5):
 * <ul>
 *   <li>For standard axis-aligned rectangular slots, extracts {@code fullImage.getSubimage(x, y, width, height)}.</li>
 *   <li>For 4-point angled/trapezoidal quadrilateral slots ({@code hasCustomPolygon() == true && size == 4}),
 *       orders the 4 vertices into {@code (TL, TR, BR, BL)} and applies perspective rectification via
 *       {@link PerspectiveTransformer#warpPerspective} so features are extracted strictly from the
 *       rectified slot interior without adjacent-bay bleed-through.</li>
 *   <li>For general multi-point polygons ({@code size != 4}), extracts the bounding box and masks out
 *       pixels outside the polygon contour with the mean interior RGB colour.</li>
 * </ul>
 */
public final class RoiExtractor {

    private RoiExtractor() { /* utility class */ }

    /**
     * Crops or rectifies the region of interest for a single parking slot from the full
     * parking-area image.
     */
    public static BufferedImage extract(BufferedImage fullImage, ParkingSlot slot) {
        if (fullImage == null) throw new IllegalArgumentException("fullImage must not be null.");
        if (slot == null)      throw new IllegalArgumentException("slot must not be null.");

        int imgW = fullImage.getWidth();
        int imgH = fullImage.getHeight();

        if (slot.getX() < 0 || slot.getY() < 0 || slot.getWidth() <= 0 || slot.getHeight() <= 0
                || slot.getX() + slot.getWidth() > imgW || slot.getY() + slot.getHeight() > imgH) {
            throw new IllegalArgumentException(
                String.format("Slot %d ROI (%d,%d,%dx%d) extends outside image (%dx%d).",
                    slot.getId(), slot.getX(), slot.getY(),
                    slot.getWidth(), slot.getHeight(), imgW, imgH));
        }

        if (!slot.hasCustomPolygon()) {
            BufferedImage sub = fullImage.getSubimage(slot.getX(), slot.getY(), slot.getWidth(), slot.getHeight());
            // Detach Raster/DataBuffer from fullImage so the large parent image buffer can be garbage collected
            BufferedImage standalone = new BufferedImage(
                slot.getWidth(), slot.getHeight(),
                fullImage.getType() != 0 ? fullImage.getType() : BufferedImage.TYPE_INT_RGB
            );
            java.awt.Graphics2D g = standalone.createGraphics();
            g.drawImage(sub, 0, 0, null);
            g.dispose();
            return standalone;
        }

        List<double[]> poly = slot.getPolygonVertices();
        if (poly.size() == 4) {
            double[][] orderedQuad = orderQuadClockwise(poly);
            int dstW = Math.max(12, slot.getWidth());
            int dstH = Math.max(12, slot.getHeight());
            return PerspectiveTransformer.warpPerspective(fullImage, orderedQuad, dstW, dstH);
        }

        return extractPolygonMaskedSubimage(fullImage, slot, poly);
    }

    /**
     * Orders 4 quadrilateral vertices into clockwise {@code [Top-Left, Top-Right, Bottom-Right, Bottom-Left]}.
     * Uses centroid polar-angle sorting to handle rotated, slanted, or inverted quad polygons without distortion.
     */
    public static double[][] orderQuadClockwise(List<double[]> pts) {
        if (pts == null || pts.size() != 4) {
            throw new IllegalArgumentException("orderQuadClockwise requires exactly 4 vertices.");
        }

        double cx = 0;
        double cy = 0;
        double minX = Double.MAX_VALUE;
        double minY = Double.MAX_VALUE;
        for (double[] p : pts) {
            cx += p[0];
            cy += p[1];
            if (p[0] < minX) minX = p[0];
            if (p[1] < minY) minY = p[1];
        }
        cx /= 4.0;
        cy /= 4.0;

        final double centerX = cx;
        final double centerY = cy;
        List<double[]> clockwise = new ArrayList<>(pts);
        clockwise.sort(Comparator.comparingDouble(p -> Math.atan2(p[1] - centerY, p[0] - centerX)));

        // Locate top-left vertex closest to the top-left bounding box corner (minX, minY)
        int tlIdx = 0;
        double minDistSq = Double.MAX_VALUE;
        for (int i = 0; i < 4; i++) {
            double[] p = clockwise.get(i);
            double dx = p[0] - minX;
            double dy = p[1] - minY;
            double distSq = dx * dx + dy * dy;
            if (distSq < minDistSq) {
                minDistSq = distSq;
                tlIdx = i;
            }
        }

        double[] tl = clockwise.get(tlIdx);
        double[] tr = clockwise.get((tlIdx + 1) % 4);
        double[] br = clockwise.get((tlIdx + 2) % 4);
        double[] bl = clockwise.get((tlIdx + 3) % 4);
        return new double[][]{tl, tr, br, bl};
    }

    private static BufferedImage extractPolygonMaskedSubimage(BufferedImage fullImage,
                                                              ParkingSlot slot,
                                                              List<double[]> poly) {
        int sx = slot.getX();
        int sy = slot.getY();
        int sw = slot.getWidth();
        int sh = slot.getHeight();

        Path2D.Double path = new Path2D.Double();
        for (int i = 0; i < poly.size(); i++) {
            double px = poly.get(i)[0] - sx;
            double py = poly.get(i)[1] - sy;
            if (i == 0) path.moveTo(px, py);
            else path.lineTo(px, py);
        }
        path.closePath();

        long sumR = 0, sumG = 0, sumB = 0;
        int insideCount = 0;
        for (int y = 0; y < sh; y++) {
            for (int x = 0; x < sw; x++) {
                if (path.contains(x + 0.5, y + 0.5)) {
                    int rgb = fullImage.getRGB(sx + x, sy + y);
                    sumR += (rgb >> 16) & 0xFF;
                    sumG += (rgb >> 8) & 0xFF;
                    sumB += rgb & 0xFF;
                    insideCount++;
                }
            }
        }
        int fillRgb = insideCount > 0
            ? ((0xFF << 24)
                | ((int) (sumR / insideCount) << 16)
                | ((int) (sumG / insideCount) << 8)
                | (int) (sumB / insideCount))
            : 0xFF3C3E42;

        BufferedImage out = new BufferedImage(sw, sh, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < sh; y++) {
            for (int x = 0; x < sw; x++) {
                if (path.contains(x + 0.5, y + 0.5)) {
                    out.setRGB(x, y, fullImage.getRGB(sx + x, sy + y));
                } else {
                    out.setRGB(x, y, fillRgb);
                }
            }
        }
        return out;
    }
}
