package com.parknchillai.app;

import com.parknchillai.ai.DecisionTreeModel;
import com.parknchillai.detection.OccupancyDecisionEngine;
import com.parknchillai.detection.SlotOverlapAnalyzer;
import com.parknchillai.detection.VehicleDetection;
import com.parknchillai.detection.VehicleDetectionService;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;

import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

/**
 * Complete per-slot diagnostic record generated during Phase 1 / Phase 6 pipeline execution.
 *
 * <p>Contains:
 * <ol>
 *   <li>Slot ID and crop coordinates {@code (x, y, width, height, origin)}</li>
 *   <li>All 8 numerical features ({@link FeatureVector})</li>
 *   <li>Decision Tree prediction, step-by-step split path, and terminal leaf sample counts
 *       ({@code occupiedCount, availableCount, totalSamples, leafPurity})</li>
 *   <li>Vehicle detections and spatial vehicle-to-slot overlap evidence ({@code IoS, IoU, centerInSlot})</li>
 *   <li>Final occupancy outcome, active {@link OccupancyDecisionEngine.AnalysisMode}, prediction source,
 *       and review reason (when ambiguous or failed)</li>
 *   <li>Model version and dataset provenance summary</li>
 * </ol>
 */
public record SlotDiagnosticRecord(
        int slotId,
        int cropX,
        int cropY,
        int cropWidth,
        int cropHeight,
        ParkingSlot.GeometryOrigin geometryOrigin,
        FeatureVector features,
        OccupancyDecisionEngine.AnalysisMode analysisMode,
        SlotStatus treePrediction,
        String decisionPath,
        int leafOccupiedCount,
        int leafAvailableCount,
        int leafTotalSamples,
        double leafPurity,
        VehicleDetectionService.DetectorStatus detectorStatus,
        SlotStatus vehicleDetectorPrediction,
        List<VehicleDetection> contributingDetections,
        SlotOverlapAnalyzer.SlotOverlapEvidence overlapEvidence,
        SlotStatus finalStatus,
        String predictionSource,
        String reviewReason,
        String modelVersion,
        String datasetProvenance
) {
    public String formatVehicleDetectionsSummary() {
        if (detectorStatus != VehicleDetectionService.DetectorStatus.READY) {
            return "Detection Could Not Run [" + (detectorStatus != null ? detectorStatus.name() : "MODEL_MISSING") + "]";
        }
        if (contributingDetections == null || contributingDetections.isEmpty()) {
            return "Detector READY: 0 vehicles overlapping this slot";
        }
        return contributingDetections.stream()
            .map(VehicleDetection::formatCompact)
            .collect(Collectors.joining("; "));
    }

    public String formatOverlapEvidenceSummary() {
        if (detectorStatus != VehicleDetectionService.DetectorStatus.READY || overlapEvidence == null) {
            return "Vehicle inference unavailable (" + (detectorStatus != null ? detectorStatus : "NONE") + "); overlap not evaluated.";
        }
        return overlapEvidence.explanation();
    }

    public String formatMultiLineReport() {
        return String.format(Locale.ROOT,
            "Slot #%d | Crop=(%d, %d, %dx%d px, %s) | Final=%s | Mode=%s\n" +
            "  Prediction Source : %s\n" +
            "  8 Features        : meanInt=%.2f, fgRatio=%.4f, darkRatio=%.4f, edgeDens=%.4f, objArea=%.4f, meanSat=%.2f, satRatio=%.4f, blobRatio=%.4f\n" +
            "  CART Tree Trace   : %s -> %s (Leaf Purity=%.1f%%, occ=%d, avail=%d, total=%d)\n" +
            "  Vehicle Detections: %s\n" +
            "  Slot Overlap      : %s\n" +
            "  Review Reason     : %s\n" +
            "  Model & Provenance: %s | %s",
            slotId, cropX, cropY, cropWidth, cropHeight,
            geometryOrigin != null ? geometryOrigin.getDisplayName() : "CSV_CONFIG",
            finalStatus,
            analysisMode != null ? analysisMode.getDisplayName() : "Feature-Based Decision Tree",
            predictionSource != null ? predictionSource : "Feature-Based Decision Tree",
            features.getMeanIntensity(), features.getForegroundRatio(), features.getDarkPixelRatio(),
            features.getEdgeDensity(), features.getObjectArea(), features.getMeanSaturation(),
            features.getSaturationRatio(), features.getLargestBlobRatio(),
            decisionPath, treePrediction, leafPurity * 100.0,
            leafOccupiedCount, leafAvailableCount, leafTotalSamples,
            formatVehicleDetectionsSummary(),
            formatOverlapEvidenceSummary(),
            (reviewReason == null || reviewReason.isBlank()) ? "None (Confirmed Classification)" : reviewReason,
            modelVersion != null ? modelVersion : "CART-v2.2-8feat",
            datasetProvenance != null ? datasetProvenance : "Synthetic/Demo Corpus (Real-World Generalization NOT ESTABLISHED)"
        );
    }

    public static SlotDiagnosticRecord fromComponents(
            ParkingSlot slot,
            FeatureVector features,
            DecisionTreeModel.DecisionTrace trace,
            OccupancyDecisionEngine.CombinedSlotDecision combinedDecision,
            VehicleDetectionService.DetectionBatchResult detectorBatch,
            SlotOverlapAnalyzer.SlotOverlapEvidence overlapEvidence,
            String modelVersion,
            String datasetProvenance) {
        DecisionTreeModel.LeafSummary leaf = trace != null ? trace.leaf()
            : new DecisionTreeModel.LeafSummary(SlotStatus.NEEDS_REVIEW, 0, 0, 0, 0.0);
        VehicleDetectionService.DetectorStatus detSt = detectorBatch != null
            ? detectorBatch.status()
            : VehicleDetectionService.DetectorStatus.MODEL_MISSING;
        List<VehicleDetection> dets = overlapEvidence != null
            ? overlapEvidence.contributingDetections()
            : List.of();
        OccupancyDecisionEngine.AnalysisMode mode = combinedDecision != null
            ? combinedDecision.mode()
            : OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE;
        SlotStatus finalSt = combinedDecision != null
            ? combinedDecision.finalStatus()
            : (trace != null ? trace.finalStatus() : SlotStatus.NEEDS_REVIEW);
        SlotStatus treeSt = combinedDecision != null
            ? combinedDecision.treeStatus()
            : (trace != null ? trace.finalStatus() : SlotStatus.NEEDS_REVIEW);
        SlotStatus vehSt = combinedDecision != null
            ? combinedDecision.vehicleDetectorStatus()
            : SlotStatus.NEEDS_REVIEW;
        String src = combinedDecision != null
            ? combinedDecision.predictionSource()
            : "Feature-Based CART Decision Tree (v2.2-8feat)";
        String rev = combinedDecision != null
            ? combinedDecision.reviewReason()
            : (trace != null && trace.reviewRequired() ? trace.uncertaintyReason() : "");

        return new SlotDiagnosticRecord(
            slot.getId(),
            slot.getX(),
            slot.getY(),
            slot.getWidth(),
            slot.getHeight(),
            slot.getOrigin(),
            features,
            mode,
            treeSt,
            trace != null ? trace.compactPath() : "",
            leaf.occupiedCount(),
            leaf.availableCount(),
            leaf.totalSamples(),
            leaf.leafPurity(),
            detSt,
            vehSt,
            dets,
            overlapEvidence,
            finalSt,
            src,
            rev,
            modelVersion,
            datasetProvenance
        );
    }
}
