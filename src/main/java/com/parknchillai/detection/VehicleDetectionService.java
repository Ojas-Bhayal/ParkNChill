package com.parknchillai.detection;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Locale;

/**
 * Responsibility 2 of the Vehicle Detection Pipeline:
 * Replaceable raw-image vehicle detector service contract.
 *
 * <p>Implementations must NEVER fabricate or hard-code vehicle bounding boxes when
 * the underlying model or runtime dependency is missing, incompatible, or fails at inference time.
 */
public interface VehicleDetectionService {

    /**
     * Explicit detector status values required by Phase 4:
     * {@code READY}, {@code MODEL_MISSING}, {@code RUNTIME_MISSING},
     * {@code MODEL_INCOMPATIBLE}, and {@code INFERENCE_FAILED}.
     */
    enum DetectorStatus {
        READY("READY"),
        MODEL_MISSING("MODEL_MISSING"),
        RUNTIME_MISSING("RUNTIME_MISSING"),
        MODEL_INCOMPATIBLE("MODEL_INCOMPATIBLE"),
        INFERENCE_FAILED("INFERENCE_FAILED");

        private final String displayName;
        DetectorStatus(String displayName) { this.displayName = displayName; }
        public String getDisplayName() { return displayName; }
    }

    record DetectionBatchResult(
            DetectorStatus status,
            String modelName,
            String licenseInfo,
            String statusMessage,
            List<VehicleDetection> rawDetections,
            List<VehicleDetection> filteredDetections,
            long inferenceTimeMs,
            DetectionStrategy strategy,
            int onnxPasses,
            boolean fromCache
    ) {
        public DetectionBatchResult(
                DetectorStatus status,
                String modelName,
                String licenseInfo,
                String statusMessage,
                List<VehicleDetection> rawDetections,
                List<VehicleDetection> filteredDetections,
                long inferenceTimeMs
        ) {
            this(status, modelName, licenseInfo, statusMessage, rawDetections, filteredDetections,
                inferenceTimeMs, DetectionStrategy.ACCURATE, 1, false);
        }

        public boolean isAvailable() {
            return status == DetectorStatus.READY;
        }

        public DetectionBatchResult withCacheHit(long hitElapsedMs) {
            return new DetectionBatchResult(
                status, modelName, licenseInfo,
                statusMessage + " [CACHE HIT]",
                rawDetections, filteredDetections, hitElapsedMs,
                strategy, 0, true
            );
        }

        /**
         * Distinguishes between a detector that ran cleanly and found zero vehicles vs.
         * a detector that could not run at all, including WS-2 strategy, passes, and cache diagnostics.
         */
        public String formatExecutionSummary() {
            String cacheTag = fromCache ? "CACHE HIT" : "INFERENCE";
            String stratTag = (strategy != null ? strategy.name() : "ACCURATE") + ", Passes=" + onnxPasses;
            if (status != DetectorStatus.READY) {
                return "Detection Could Not Run [" + status.name() + "]: " + statusMessage;
            }
            if (filteredDetections == null || filteredDetections.isEmpty()) {
                if (statusMessage != null && !statusMessage.isBlank()) return statusMessage;
                return String.format(Locale.ROOT, "Detector READY (%s) [%s, %s]: 0 vehicles detected in image (%d ms).",
                    modelName, stratTag, cacheTag, inferenceTimeMs);
            }
            return String.format(Locale.ROOT, "Detector READY (%s) [%s, %s]: %d vehicle(s) detected after NMS (%d raw candidates, %d ms).",
                modelName, stratTag, cacheTag, filteredDetections.size(),
                rawDetections != null ? rawDetections.size() : filteredDetections.size(),
                inferenceTimeMs);
        }
    }

    /**
     * Runs vehicle detection on {@code fullImage} and returns the raw and NMS-filtered detections
     * mapped back to {@code fullImage}'s coordinate space, or an explicit unavailable/failed
     * {@link DetectionBatchResult} with zero fabricated detections.
     */
    DetectionBatchResult detectVehicles(BufferedImage fullImage);

    /**
     * Runs vehicle detection with an explicit {@link DetectionStrategy} (FAST vs ACCURATE).
     */
    default DetectionBatchResult detectVehicles(BufferedImage fullImage, DetectionStrategy strategy) {
        return detectVehicles(fullImage);
    }

    /**
     * Runs vehicle detection with an explicit SHA-256 (bypassing hash recomputation) and {@link DetectionStrategy}.
     */
    default DetectionBatchResult detectVehicles(BufferedImage fullImage, String explicitImageSha256, DetectionStrategy strategy) {
        return detectVehicles(fullImage, strategy);
    }

    /**
     * Returns the current configuration/availability status of the detector without running inference.
     */
    DetectorStatus getStatus();

    /**
     * Returns a human-readable diagnostic summary of the model, required dependencies, and status.
     */
    String describeConfiguration();

    /**
     * Clears cached detection results from the LRU cache.
     */
    default void clearDetectionCache() {}
}
