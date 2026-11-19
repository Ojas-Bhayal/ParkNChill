package com.parknchillai.app;

import com.parknchillai.detection.OccupancyDecisionEngine;
import com.parknchillai.detection.VehicleDetection;
import com.parknchillai.detection.VehicleDetectionService;

import java.awt.image.BufferedImage;
import java.lang.ref.SoftReference;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Summary outcome of processing an entire parking-lot image (ParkNChill v2.3),
 * containing per-slot results, distinct Occupied / Available / Needs-Review / Unclassified counts,
 * explicit occupancy rate denominators, active {@link OccupancyDecisionEngine.AnalysisMode},
 * raw/NMS vehicle detections, unassigned vehicle detections, slot-localization reliability status,
 * the annotated visual representation, and execution time.
 *
 * <p>Memory Optimization (WS-4): Full-resolution original and annotated bitmaps can be unloaded
 * from strong references via {@link #unloadFullImages()} when inactive, retaining a {@link SoftReference}
 * to prevent out-of-memory errors across multi-image sessions.</p>
 */
public final class AnalysisSummary {

    private BufferedImage originalImage;
    private BufferedImage annotatedImage;
    private SoftReference<BufferedImage> softOriginalImage;
    private SoftReference<BufferedImage> softAnnotatedImage;
    private final List<SlotAnalysisResult> slotResults;
    private final int totalSlots;
    private final int occupiedSlots;
    private final int availableSlots;
    private final int needsReviewSlots;
    private final int unclassifiedSlots;
    private final double occupancyRate;
    private final long executionTimeMs;
    private final String modelSummary;
    private final OccupancyDecisionEngine.AnalysisMode analysisMode;
    private final VehicleDetectionService.DetectionBatchResult detectionBatchResult;
    private final List<VehicleDetection> unassignedVehicleDetections;
    private final boolean slotLocalizationReliable;
    private final String slotLocalizationMessage;

    public AnalysisSummary(BufferedImage originalImage,
                           BufferedImage annotatedImage,
                           List<SlotAnalysisResult> slotResults,
                           int totalSlots,
                           int occupiedSlots,
                           int availableSlots,
                           double occupancyRate,
                           long executionTimeMs,
                           String modelSummary) {
        this(originalImage, annotatedImage, slotResults, totalSlots, occupiedSlots, availableSlots,
            Math.max(0, totalSlots - occupiedSlots - availableSlots), 0, occupancyRate, executionTimeMs,
            modelSummary, OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE, null,
            List.of(), !slotResults.isEmpty(), "");
    }

    public AnalysisSummary(BufferedImage originalImage,
                           BufferedImage annotatedImage,
                           List<SlotAnalysisResult> slotResults,
                           int totalSlots,
                           int occupiedSlots,
                           int availableSlots,
                           int needsReviewSlots,
                           double occupancyRate,
                           long executionTimeMs,
                           String modelSummary) {
        this(originalImage, annotatedImage, slotResults, totalSlots, occupiedSlots, availableSlots,
            needsReviewSlots, 0, occupancyRate, executionTimeMs, modelSummary,
            OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE, null,
            List.of(), !slotResults.isEmpty(), "");
    }

    public AnalysisSummary(BufferedImage originalImage,
                           BufferedImage annotatedImage,
                           List<SlotAnalysisResult> slotResults,
                           int totalSlots,
                           int occupiedSlots,
                           int availableSlots,
                           int needsReviewSlots,
                           double occupancyRate,
                           long executionTimeMs,
                           String modelSummary,
                           OccupancyDecisionEngine.AnalysisMode analysisMode,
                           VehicleDetectionService.DetectionBatchResult detectionBatchResult) {
        this(originalImage, annotatedImage, slotResults, totalSlots, occupiedSlots, availableSlots,
            needsReviewSlots, 0, occupancyRate, executionTimeMs, modelSummary,
            analysisMode, detectionBatchResult, List.of(), !slotResults.isEmpty(), "");
    }

    public AnalysisSummary(BufferedImage originalImage,
                           BufferedImage annotatedImage,
                           List<SlotAnalysisResult> slotResults,
                           int totalSlots,
                           int occupiedSlots,
                           int availableSlots,
                           int needsReviewSlots,
                           int unclassifiedSlots,
                           double occupancyRate,
                           long executionTimeMs,
                           String modelSummary,
                           OccupancyDecisionEngine.AnalysisMode analysisMode,
                           VehicleDetectionService.DetectionBatchResult detectionBatchResult,
                           List<VehicleDetection> unassignedVehicleDetections,
                           boolean slotLocalizationReliable,
                           String slotLocalizationMessage) {
        this.originalImage               = originalImage;
        this.annotatedImage              = annotatedImage;
        this.softOriginalImage           = originalImage != null ? new SoftReference<>(originalImage) : null;
        this.softAnnotatedImage          = annotatedImage != null ? new SoftReference<>(annotatedImage) : null;
        this.slotResults                 = Collections.unmodifiableList(slotResults != null ? slotResults : List.of());
        this.totalSlots                  = totalSlots;
        this.occupiedSlots               = occupiedSlots;
        this.availableSlots              = availableSlots;
        this.needsReviewSlots            = needsReviewSlots;
        this.unclassifiedSlots           = unclassifiedSlots;
        this.occupancyRate               = occupancyRate;
        this.executionTimeMs             = executionTimeMs;
        this.modelSummary                = modelSummary;
        this.analysisMode                = analysisMode != null ? analysisMode : OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE;
        this.detectionBatchResult        = detectionBatchResult;
        this.unassignedVehicleDetections = Collections.unmodifiableList(
            unassignedVehicleDetections != null ? unassignedVehicleDetections : List.of());
        this.slotLocalizationReliable    = slotLocalizationReliable;
        this.slotLocalizationMessage     = slotLocalizationMessage != null ? slotLocalizationMessage : "";
    }

    public synchronized void unloadFullImages() {
        this.originalImage = null;
        this.annotatedImage = null;
        this.softOriginalImage = null;
        this.softAnnotatedImage = null;
    }

    public synchronized BufferedImage getOriginalImage() {
        if (originalImage != null) return originalImage;
        return softOriginalImage != null ? softOriginalImage.get() : null;
    }

    public synchronized BufferedImage getAnnotatedImage() {
        if (annotatedImage != null) return annotatedImage;
        return softAnnotatedImage != null ? softAnnotatedImage.get() : null;
    }

    public List<SlotAnalysisResult> getSlotResults()                               { return slotResults; }
    public int getTotalSlots()                                                     { return totalSlots; }
    public int getOccupiedSlots()                                                  { return occupiedSlots; }
    public int getAvailableSlots()                                                 { return availableSlots; }
    public int getNeedsReviewSlots()                                               { return needsReviewSlots; }
    public int getUnclassifiedSlots()                                              { return unclassifiedSlots; }
    public double getOccupancyRate()                                               { return occupancyRate; }
    public long getExecutionTimeMs()                                               { return executionTimeMs; }
    public String getModelSummary()                                                { return modelSummary; }
    public OccupancyDecisionEngine.AnalysisMode getAnalysisMode()                  { return analysisMode; }
    public VehicleDetectionService.DetectionBatchResult getDetectionBatchResult()  { return detectionBatchResult; }
    public List<VehicleDetection> getUnassignedVehicleDetections()                 { return unassignedVehicleDetections; }
    public boolean isSlotLocalizationReliable()                                    { return slotLocalizationReliable; }
    public String getSlotLocalizationMessage()                                     { return slotLocalizationMessage; }

    public List<VehicleDetection> getVehicleDetections() {
        return detectionBatchResult != null ? detectionBatchResult.filteredDetections() : List.of();
    }

    public String formatSummaryText() {
        return String.format(Locale.ROOT,
            "Total: %d | Occupied: %d (%.1f%%) | Available: %d | Needs Review: %d | Mode: %s | Time: %d ms",
            totalSlots, occupiedSlots, occupancyRate, availableSlots, needsReviewSlots,
            analysisMode.getDisplayName(), executionTimeMs);
    }
}
