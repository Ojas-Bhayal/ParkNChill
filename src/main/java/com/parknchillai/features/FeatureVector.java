package com.parknchillai.features;

import com.parknchillai.detection.SlotOverlapAnalyzer;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Immutable container and authoritative specification registry for the parking-space
 * numerical feature vector.
 *
 * <p>Supports three compatible schema levels:
 * <ol>
 *   <li><strong>{@code v2.3-10feat-raw-v1} ({@link #DETECTION_AUGMENTED_FEATURE_COUNT} = 10)</strong>:
 *       The 8 classical IVP features (indices {@code 0..7}) plus 2 vehicle-detection-derived
 *       spatial features:
 *       <ul>
 *         <li>Index {@code 8}: {@code vehicleOverlapRatio} — Intersection-over-Slot-Area ({@code IoS}) in {@code [0, 1]}</li>
 *         <li>Index {@code 9}: {@code vehicleMatchedFlag} — {@code 1.0} if a vehicle detection is matched to the slot, else {@code 0.0}</li>
 *       </ul>
 *   </li>
 *   <li><strong>{@code v2.2-8feat} ({@link #FEATURE_COUNT} = 8)</strong>:
 *       The 8 multi-modal IVP features ({@code meanIntensity}, {@code foregroundRatio},
 *       {@code darkPixelRatio}, {@code edgeDensity}, {@code objectArea}, {@code meanSaturation},
 *       {@code saturationRatio}, {@code largestBlobRatio}).</li>
 *   <li><strong>{@code v2.0-5feat} ({@link #LEGACY_FEATURE_COUNT} = 5)</strong>:
 *       Legacy 5-feature representation for backward compatibility.</li>
 * </ol>
 */
public final class FeatureVector {

    /** Legacy v1.0 feature count (5 grayscale/morphology/edge features). */
    public static final int LEGACY_FEATURE_COUNT = 5;

    /** Standard 8-feature IVP representation. */
    public static final int FEATURE_COUNT = 8;

    /** v2.3 detection-augmented feature count (8 IVP features + 2 vehicle-overlap features). */
    public static final int DETECTION_AUGMENTED_FEATURE_COUNT = 10;

    /** Default 8-feature schema identifier recorded in dataset manifests. */
    public static final String SCHEMA_VERSION = "v2.2-8feat";

    /** Canonical 10-feature schema using enhanced slot pixels and two vehicle features (WS-6 authoritative reference). */
    public static final String ENHANCED_SCHEMA_VERSION = com.parknchillai.imageprocessing.ProcessingPolicy.CANONICAL_SCHEMA_ID;

    /** Legacy 10-feature schema using raw ROI pixels. */
    public static final String RAW_SCHEMA_VERSION = "v2.3-10feat-raw-v1";

    /** Schema identifier when detector was unavailable or failed. */
    public static final String DETECTOR_FAILED_SCHEMA_VERSION = "v2.3-8feat-detector-failed";

    /**
     * Active 10-feature schema version for ParkNChill v2.3.
     */
    public static final String DETECTION_AUGMENTED_SCHEMA_VERSION = ENHANCED_SCHEMA_VERSION;

    public static final int IDX_MEAN_INTENSITY        = 0;
    public static final int IDX_FOREGROUND_RATIO      = 1;
    public static final int IDX_DARK_PIXEL_RATIO      = 2;
    public static final int IDX_EDGE_DENSITY          = 3;
    public static final int IDX_OBJECT_AREA           = 4;
    public static final int IDX_MEAN_SATURATION       = 5;
    public static final int IDX_SATURATION_RATIO      = 6;
    public static final int IDX_LARGEST_BLOB_RATIO    = 7;
    public static final int IDX_VEHICLE_OVERLAP_RATIO = 8;
    public static final int IDX_VEHICLE_MATCHED_FLAG  = 9;

    /**
     * Authoritative metadata for a single feature dimension, shared across
     * {@link FeatureExtractor}, the JavaFX {@code PipelineInspectorView}, and dataset validators.
     */
    public record FeatureSpec(
            int index,
            String key,
            String displayName,
            String formula,
            double minValid,
            double maxValid,
            String sourceStage,
            String classifierRole
    ) {
        public boolean isWithinValidBounds(double value) {
            return Double.isFinite(value) && value >= minValid - 1e-9 && value <= maxValid + 1e-9;
        }
    }

    /** Authoritative specification list for the 8 core IVP features. */
    public static final List<FeatureSpec> SPECS = List.of(
        new FeatureSpec(
            IDX_MEAN_INTENSITY,
            "meanIntensity",
            "Mean Intensity",
            "(1 / (W*H)) * sum_{x,y} SmoothedGray(x,y)",
            0.0, 255.0,
            "Stage 4: 5×5 Gaussian Smoothed Grayscale",
            "Measures overall brightness of the parking space after noise smoothing."
        ),
        new FeatureSpec(
            IDX_FOREGROUND_RATIO,
            "foregroundRatio",
            "Foreground Ratio",
            "(1 / (W*H)) * count(Binary(x,y) == 255)",
            0.0, 1.0,
            "Stage 5: Binary Threshold Mask (Otsu or Two-Sided Adaptive+HSV)",
            "Quantifies raw segmented foreground coverage prior to morphological closing."
        ),
        new FeatureSpec(
            IDX_DARK_PIXEL_RATIO,
            "darkPixelRatio",
            "Dark Pixel Ratio",
            "(1 / (W*H)) * count(SmoothedGray(x,y) < 85)",
            0.0, 1.0,
            "Stage 4: 5×5 Gaussian Smoothed Grayscale",
            "Captures dark asphalt, tires, windshields, and under-vehicle shadow regions."
        ),
        new FeatureSpec(
            IDX_EDGE_DENSITY,
            "edgeDensity",
            "Edge Density",
            "(1 / (W*H)) * count(SobelMag(x,y) > 30)",
            0.0, 1.0,
            "Stage 8: 3×3 Sobel Edge Magnitude",
            "Secondary split feature distinguishing high-gradient vehicle contours from flat or weathered empty pavement."
        ),
        new FeatureSpec(
            IDX_OBJECT_AREA,
            "objectArea",
            "Object Area",
            "(1 / (W*H)) * count(MorphoClosed(x,y) == 255)",
            0.0, 1.0,
            "Stage 6: 3×3 Morphologically Closed Mask",
            "Measures consolidated foreground area after bridging small internal gaps."
        ),
        new FeatureSpec(
            IDX_MEAN_SATURATION,
            "meanSaturation",
            "HSV Mean Saturation",
            "(1 / (W*H)) * sum_{x,y} round(((Cmax - Cmin) / Cmax) * 255)",
            0.0, 255.0,
            "Stage 3: HSV Saturation Map",
            "Shadow-invariant color intensity (near 0 on achromatic gray asphalt and shadows; elevated on painted vehicles)."
        ),
        new FeatureSpec(
            IDX_SATURATION_RATIO,
            "saturationRatio",
            "Saturation Ratio (S > 90)",
            "(1 / (W*H)) * count(HSV_S(x,y) > 90)",
            0.0, 1.0,
            "Stage 3: HSV Saturation Map (threshold S > 90)",
            "Fraction of distinctly chromatic pixels (HSV S > 90) used for shadow-free adaptive fusion and colour discrimination."
        ),
        new FeatureSpec(
            IDX_LARGEST_BLOB_RATIO,
            "largestBlobRatio",
            "Largest Blob Ratio",
            "maxComponentPixels / (W*H) via 8-connected BFS flood-fill",
            0.0, 1.0,
            "Stage 7: Largest Connected Component (BFS)",
            "Primary IVP split feature separating contiguous vehicle bodies from scattered pavement speckles."
        )
    );

    /** Authoritative specification list for all 10 features (8 IVP + 2 detection-derived). */
    public static final List<FeatureSpec> EXTENDED_SPECS = List.of(
        SPECS.get(0),
        SPECS.get(1),
        SPECS.get(2),
        SPECS.get(3),
        SPECS.get(4),
        SPECS.get(5),
        SPECS.get(6),
        SPECS.get(7),
        new FeatureSpec(
            IDX_VEHICLE_OVERLAP_RATIO,
            "vehicleOverlapRatio",
            "Vehicle Overlap Ratio (IoS)",
            "max_{det} (intersectionArea(det, slot) / slotArea)",
            0.0, 1.0,
            "Vehicle Detection & Slot Overlap Analyzer",
            "Measures the fraction of the parking space covered by the best-matching detected vehicle bounding box."
        ),
        new FeatureSpec(
            IDX_VEHICLE_MATCHED_FLAG,
            "vehicleMatchedFlag",
            "Vehicle Matched Indicator",
            "1.0 if a detected vehicle is assigned to this slot, else 0.0",
            0.0, 1.0,
            "Vehicle Detection & Slot Overlap Analyzer",
            "Binary indicator showing whether spatial vehicle-to-slot assignment matched a vehicle to this space."
        )
    );

    private final double[] values;
    private final double vehicleOverlapRatio;
    private final double vehicleMatchedFlag;
    private final boolean hasDetectionFeatures;
    private final boolean detectorAvailable;

    public FeatureVector(double[] values) {
        if (values == null || (values.length != FEATURE_COUNT
                && values.length != LEGACY_FEATURE_COUNT
                && values.length != DETECTION_AUGMENTED_FEATURE_COUNT)) {
            throw new IllegalArgumentException(
                "FeatureVector requires " + FEATURE_COUNT + ", " + DETECTION_AUGMENTED_FEATURE_COUNT
                    + ", or legacy " + LEGACY_FEATURE_COUNT + " values; got "
                    + (values == null ? "null" : values.length));
        }
        for (int i = 0; i < Math.min(FEATURE_COUNT, values.length); i++) {
            if (!Double.isFinite(values[i])) {
                throw new IllegalArgumentException(
                    "Feature index " + i + " must be finite (not NaN or Infinity); got " + values[i]);
            }
        }
        this.values = new double[FEATURE_COUNT];
        System.arraycopy(values, 0, this.values, 0, Math.min(FEATURE_COUNT, values.length));
        if (values.length == DETECTION_AUGMENTED_FEATURE_COUNT) {
            boolean detAvail = Double.isFinite(values[IDX_VEHICLE_OVERLAP_RATIO])
                && Double.isFinite(values[IDX_VEHICLE_MATCHED_FLAG]);
            this.vehicleOverlapRatio = values[IDX_VEHICLE_OVERLAP_RATIO];
            this.vehicleMatchedFlag  = values[IDX_VEHICLE_MATCHED_FLAG];
            this.hasDetectionFeatures = true;
            this.detectorAvailable = detAvail;
        } else {
            this.vehicleOverlapRatio = 0.0;
            this.vehicleMatchedFlag  = 0.0;
            this.hasDetectionFeatures = false;
            this.detectorAvailable = true;
        }
    }

    private FeatureVector(double[] eightIvpValues,
                          double vehicleOverlapRatio,
                          double vehicleMatchedFlag,
                          boolean hasDetectionFeatures,
                          boolean detectorAvailable) {
        this.values = eightIvpValues.clone();
        this.vehicleOverlapRatio = vehicleOverlapRatio;
        this.vehicleMatchedFlag = vehicleMatchedFlag;
        this.hasDetectionFeatures = hasDetectionFeatures;
        this.detectorAvailable = detectorAvailable;
    }

    /**
     * Returns a new {@link FeatureVector} enriched with the two detection-derived features
     * ({@code vehicleOverlapRatio} in {@code [0, 1]} and {@code vehicleMatchedFlag} in {@code {0.0, 1.0}}).
     */
    public FeatureVector withDetectionFeatures(double overlapRatio, boolean vehicleMatched) {
        if (!Double.isFinite(overlapRatio) || overlapRatio < -1e-9 || overlapRatio > 1.0 + 1e-9) {
            throw new IllegalArgumentException("vehicleOverlapRatio must be in [0.0, 1.0]; got " + overlapRatio);
        }
        double clampedIoS = Math.max(0.0, Math.min(1.0, overlapRatio));
        double matchedVal = vehicleMatched ? 1.0 : 0.0;
        return new FeatureVector(this.values, clampedIoS, matchedVal, true, true);
    }

    /**
     * Returns a new {@link FeatureVector} marked with honest detector failure/unavailability,
     * recording NaN for vehicle features and setting schema to {@link #DETECTOR_FAILED_SCHEMA_VERSION}.
     */
    public FeatureVector withDetectorUnavailable(String failureReason) {
        return new FeatureVector(this.values, Double.NaN, Double.NaN, true, false);
    }

    /**
     * Returns a new {@link FeatureVector} enriched with detection overlap measurements from
     * {@link SlotOverlapAnalyzer.SlotOverlapEvidence}.
     */
    public FeatureVector withDetectionEvidence(SlotOverlapAnalyzer.SlotOverlapEvidence evidence) {
        if (evidence == null) {
            return withDetectionFeatures(0.0, false);
        }
        boolean matched = evidence.status() == com.parknchillai.parking.SlotStatus.OCCUPIED
            || (!evidence.contributingDetections().isEmpty() && evidence.bestIoS() >= 0.26);
        return withDetectionFeatures(evidence.bestIoS(), matched);
    }

    public boolean hasDetectionFeatures() {
        return hasDetectionFeatures;
    }

    public boolean isDetectorAvailable() {
        return detectorAvailable;
    }

    public String getSchemaVersion() {
        if (!hasDetectionFeatures) {
            return SCHEMA_VERSION;
        }
        if (!detectorAvailable) {
            return DETECTOR_FAILED_SCHEMA_VERSION;
        }
        return DETECTION_AUGMENTED_SCHEMA_VERSION;
    }

    /** Returns {@code true} if all feature values lie within their theoretical valid bounds. */
    public boolean hasAllValuesWithinValidBounds() {
        for (int i = 0; i < FEATURE_COUNT; i++) {
            if (!SPECS.get(i).isWithinValidBounds(values[i])) {
                return false;
            }
        }
        if (!hasDetectionFeatures || !detectorAvailable) {
            return true;
        }
        return EXTENDED_SPECS.get(IDX_VEHICLE_OVERLAP_RATIO).isWithinValidBounds(vehicleOverlapRatio)
            && EXTENDED_SPECS.get(IDX_VEHICLE_MATCHED_FLAG).isWithinValidBounds(vehicleMatchedFlag);
    }

    /** Returns the value at the given index in {@code [0, 10)}. */
    public double get(int index) {
        if (index >= 0 && index < FEATURE_COUNT) {
            return values[index];
        }
        if (index == IDX_VEHICLE_OVERLAP_RATIO) {
            return vehicleOverlapRatio;
        }
        if (index == IDX_VEHICLE_MATCHED_FLAG) {
            return vehicleMatchedFlag;
        }
        throw new IndexOutOfBoundsException("Feature index out of range: " + index);
    }

    /** Returns a defensive copy of the 8 core IVP feature values. */
    public double[] toArray() { return values.clone(); }

    /** Returns a defensive copy of all 10 features (8 IVP + 2 detection-derived). */
    public double[] toExtendedArray() {
        return toArray(DETECTION_AUGMENTED_FEATURE_COUNT);
    }

    /** Returns a defensive copy matched to {@code count} features (5, 8, or 10). */
    public double[] toArray(int count) {
        if (count <= 0 || count > DETECTION_AUGMENTED_FEATURE_COUNT) {
            throw new IllegalArgumentException(
                "Requested feature count must be in [1, " + DETECTION_AUGMENTED_FEATURE_COUNT + "]; got " + count);
        }
        if (count <= FEATURE_COUNT) {
            return Arrays.copyOf(values, count);
        }
        double[] ext = new double[count];
        System.arraycopy(values, 0, ext, 0, FEATURE_COUNT);
        if (count > IDX_VEHICLE_OVERLAP_RATIO) ext[IDX_VEHICLE_OVERLAP_RATIO] = vehicleOverlapRatio;
        if (count > IDX_VEHICLE_MATCHED_FLAG)  ext[IDX_VEHICLE_MATCHED_FLAG]  = vehicleMatchedFlag;
        return ext;
    }

    public double getMeanIntensity()       { return values[IDX_MEAN_INTENSITY]; }
    public double getForegroundRatio()     { return values[IDX_FOREGROUND_RATIO]; }
    public double getDarkPixelRatio()      { return values[IDX_DARK_PIXEL_RATIO]; }
    public double getEdgeDensity()         { return values[IDX_EDGE_DENSITY]; }
    public double getObjectArea()          { return values[IDX_OBJECT_AREA]; }
    public double getMeanSaturation()      { return values[IDX_MEAN_SATURATION]; }
    public double getSaturationRatio()     { return values[IDX_SATURATION_RATIO]; }
    public double getLargestBlobRatio()    { return values[IDX_LARGEST_BLOB_RATIO]; }
    public double getVehicleOverlapRatio() { return vehicleOverlapRatio; }
    public double getVehicleMatchedFlag()  { return vehicleMatchedFlag; }
    public boolean isVehicleMatched()      { return vehicleMatchedFlag >= 0.5; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof FeatureVector other)) return false;
        return Arrays.equals(this.values, other.values);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(this.values);
    }

    @Override
    public String toString() {
        if (hasDetectionFeatures) {
            return "FeatureVector" + Arrays.toString(toExtendedArray());
        }
        return "FeatureVector" + Arrays.toString(values);
    }

    public String toCsvRow() {
        return toCsvRow(FEATURE_COUNT);
    }

    public String toCsvRow(int featureCount) {
        double[] arr = toArray(Math.min(DETECTION_AUGMENTED_FEATURE_COUNT, Math.max(1, featureCount)));
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(String.format(Locale.ROOT, "%.6f", arr[i]));
        }
        return sb.toString();
    }

    public static String csvHeader() {
        return "meanIntensity,foregroundRatio,darkPixelRatio,edgeDensity,objectArea,meanSaturation,saturationRatio,largestBlobRatio,label";
    }

    public static String extendedCsvHeader() {
        return "meanIntensity,foregroundRatio,darkPixelRatio,edgeDensity,objectArea,meanSaturation,saturationRatio,largestBlobRatio,vehicleOverlapRatio,vehicleMatchedFlag,label";
    }
}
