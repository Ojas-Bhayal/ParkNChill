package com.parknchillai.layout;

import com.parknchillai.parking.ParkingSlot;

import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Result of automatic parking-space localization (ParkNChill v2.3 Requirement 4),
 * including detected rows, bay width estimate, non-overlapping slot bounding boxes,
 * perspective-aware quadrilateral polygons, dominant bay-marking orientation,
 * and an explicit reliability flag ({@link #isLocalizationReliable()}) so unmarked
 * scenes never silently invent empty parking spaces.
 */
public final class LayoutDetectionResult {

    private final ParkingBoundary boundary;
    private final List<ParkingRow> rows;
    private final int estimatedSlotWidth;
    private final List<ParkingSlot> slots;
    private final boolean localizationReliable;
    private final double dominantMarkingOrientationDeg;
    private final int detectedLineSegmentCount;
    private final String localizationSummary;

    public LayoutDetectionResult(ParkingBoundary boundary,
                                 List<ParkingRow> rows,
                                 int estimatedSlotWidth,
                                 List<ParkingSlot> slots) {
        this(boundary, rows, estimatedSlotWidth, slots,
            !slots.isEmpty(), 90.0, slots.size() * 2,
            !slots.isEmpty()
                ? String.format(Locale.ROOT, "Localized %d parking space(s) across %d row(s) from visible bay markings.", slots.size(), rows.size())
                : "Parking-slot localization is uncertain: insufficient visible bay markings in image (0 spaces inferred).");
    }

    public LayoutDetectionResult(ParkingBoundary boundary,
                                 List<ParkingRow> rows,
                                 int estimatedSlotWidth,
                                 List<ParkingSlot> slots,
                                 boolean localizationReliable,
                                 double dominantMarkingOrientationDeg,
                                 int detectedLineSegmentCount,
                                 String localizationSummary) {
        if (boundary == null) throw new IllegalArgumentException("Boundary must not be null.");
        if (rows == null) throw new IllegalArgumentException("Rows list must not be null.");
        if (slots == null) throw new IllegalArgumentException("Slots list must not be null.");
        this.boundary = boundary;
        this.rows = Collections.unmodifiableList(rows);
        this.estimatedSlotWidth = estimatedSlotWidth;
        this.slots = Collections.unmodifiableList(slots);
        this.localizationReliable = localizationReliable && !slots.isEmpty();
        this.dominantMarkingOrientationDeg = dominantMarkingOrientationDeg;
        this.detectedLineSegmentCount = Math.max(0, detectedLineSegmentCount);
        this.localizationSummary = (localizationSummary == null || localizationSummary.isBlank())
            ? (this.localizationReliable
                ? String.format(Locale.ROOT, "Localized %d parking space(s) across %d row(s).", slots.size(), rows.size())
                : "Parking-slot localization is uncertain: no reliable bay markings detected.")
            : localizationSummary;
    }

    public ParkingBoundary getBoundary()                { return boundary; }
    public List<ParkingRow> getRows()                   { return rows; }
    public int getEstimatedSlotWidth()                  { return estimatedSlotWidth; }
    public List<ParkingSlot> getSlots()                 { return slots; }
    public int getDetectedSlotCount()                   { return slots.size(); }
    public boolean isLocalizationReliable()             { return localizationReliable; }
    public double getDominantMarkingOrientationDeg()    { return dominantMarkingOrientationDeg; }
    public int getDetectedLineSegmentCount()            { return detectedLineSegmentCount; }
    public String getLocalizationSummary()              { return localizationSummary; }

    @Override
    public String toString() {
        return String.format(Locale.ROOT,
            "LayoutDetectionResult[boundary=%s, rows=%d, estSlotWidth=%d, slots=%d, reliable=%s, orient=%.1f deg, lines=%d]",
            boundary, rows.size(), estimatedSlotWidth, slots.size(),
            localizationReliable, dominantMarkingOrientationDeg, detectedLineSegmentCount);
    }
}
