package com.parknchillai.detection;

import com.parknchillai.parking.ParkingSlot;

import java.util.Locale;

/**
 * Immutable bounding-box detection produced by a raw-image vehicle detector in
 * original image coordinates {@code (x, y, width, height)}, together with
 * inference dimensions and processing metadata (ParkNChill v2.3 Requirement 3 &amp; 5).
 *
 * <p>When evaluating overlap against a {@link ParkingSlot} that carries a custom polygon
 * or angled quadrilateral contour, {@link #intersectionArea(ParkingSlot)},
 * {@link #intersectionOverSlotArea(ParkingSlot)}, {@link #iouWithSlot(ParkingSlot)}, and
 * {@link #isCenterInside(ParkingSlot)} use the slot's exact polygon geometry.
 */
public record VehicleDetection(
        String detectionId,
        VehicleCategory category,
        double x,
        double y,
        double width,
        double height,
        double detectionScore,
        String modelId,
        int inferenceWidth,
        int inferenceHeight,
        String processingMetadata
) {
    public VehicleDetection {
        if (detectionId == null || detectionId.isBlank()) {
            throw new IllegalArgumentException("detectionId must not be null or blank.");
        }
        if (category == null) {
            throw new IllegalArgumentException("category must not be null.");
        }
        if (!Double.isFinite(x) || !Double.isFinite(y) || !Double.isFinite(width) || !Double.isFinite(height)) {
            throw new IllegalArgumentException("Bounding-box coordinates must be finite.");
        }
        if (width <= 0.0 || height <= 0.0) {
            throw new IllegalArgumentException("Bounding-box width and height must be > 0.");
        }
        if (!Double.isFinite(detectionScore) || detectionScore < 0.0 || detectionScore > 1.0) {
            throw new IllegalArgumentException("detectionScore must be in [0.0, 1.0]; got " + detectionScore);
        }
        if (modelId == null || modelId.isBlank()) {
            modelId = "UNKNOWN_MODEL";
        }
        if (inferenceWidth <= 0) {
            inferenceWidth = DetectionPreprocessor.DEFAULT_INPUT_SIZE;
        }
        if (inferenceHeight <= 0) {
            inferenceHeight = DetectionPreprocessor.DEFAULT_INPUT_SIZE;
        }
        if (processingMetadata == null || processingMetadata.isBlank()) {
            processingMetadata = String.format(Locale.ROOT, "input=%dx%d,pass=full-frame", inferenceWidth, inferenceHeight);
        }
    }

    public VehicleDetection(
            String detectionId,
            VehicleCategory category,
            double x,
            double y,
            double width,
            double height,
            double detectionScore,
            String modelId) {
        this(detectionId, category, x, y, width, height, detectionScore, modelId,
            DetectionPreprocessor.DEFAULT_INPUT_SIZE,
            DetectionPreprocessor.DEFAULT_INPUT_SIZE,
            "input=640x640,pass=full-frame");
    }

    public double centerX() { return x + width / 2.0; }
    public double centerY() { return y + height / 2.0; }
    public double area()    { return width * height; }

    public double intersectionArea(double rx, double ry, double rw, double rh) {
        if (rw <= 0.0 || rh <= 0.0) return 0.0;
        double ix0 = Math.max(this.x, rx);
        double iy0 = Math.max(this.y, ry);
        double ix1 = Math.min(this.x + this.width, rx + rw);
        double iy1 = Math.min(this.y + this.height, ry + rh);
        double iw = Math.max(0.0, ix1 - ix0);
        double ih = Math.max(0.0, iy1 - iy0);
        return iw * ih;
    }

    /**
     * Computes exact intersection area between this vehicle detection box and {@code slot},
     * using Sutherland-Hodgman polygon clipping when {@code slot} has an angled/polygon contour.
     */
    public double intersectionArea(ParkingSlot slot) {
        if (slot == null || slot.getWidth() <= 0 || slot.getHeight() <= 0) return 0.0;
        return slot.computeIntersectionAreaWithRect(this.x, this.y, this.width, this.height);
    }

    public double iouWith(VehicleDetection other) {
        if (other == null) return 0.0;
        double inter = intersectionArea(other.x, other.y, other.width, other.height);
        if (inter <= 0.0) return 0.0;
        double union = this.area() + other.area() - inter;
        return union > 0.0 ? inter / union : 0.0;
    }

    public double iouWithSlot(ParkingSlot slot) {
        if (slot == null || slot.getWidth() <= 0 || slot.getHeight() <= 0) return 0.0;
        double inter = intersectionArea(slot);
        if (inter <= 0.0) return 0.0;
        double slotArea = slot.computeExactArea();
        double union = this.area() + slotArea - inter;
        return union > 0.0 ? inter / union : 0.0;
    }

    public double intersectionOverSlotArea(ParkingSlot slot) {
        if (slot == null || slot.getWidth() <= 0 || slot.getHeight() <= 0) return 0.0;
        double slotArea = slot.computeExactArea();
        return slotArea > 0.0 ? intersectionArea(slot) / slotArea : 0.0;
    }

    public boolean isCenterInside(ParkingSlot slot) {
        if (slot == null || slot.getWidth() <= 0 || slot.getHeight() <= 0) return false;
        return slot.containsPoint(centerX(), centerY());
    }

    public String formatCompact() {
        return String.format(Locale.ROOT, "%s(%s, score=%.2f, box=[%.0f,%.0f,%.0fx%.0f], inf=%dx%d)",
            detectionId, category.getDisplayName(), detectionScore, x, y, width, height,
            inferenceWidth, inferenceHeight);
    }
}
