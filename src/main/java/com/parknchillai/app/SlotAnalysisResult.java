package com.parknchillai.app;

import com.parknchillai.ai.DecisionTreeModel;
import com.parknchillai.detection.OccupancyDecisionEngine;
import com.parknchillai.detection.SlotOverlapAnalyzer;
import com.parknchillai.detection.VehicleDetection;
import com.parknchillai.detection.VehicleDetectionService;
import com.parknchillai.features.FeatureExtractor;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;

import java.awt.image.BufferedImage;
import java.lang.ref.SoftReference;
import java.util.List;
import java.util.Locale;

/**
 * Encapsulates the complete analysis for an individual parking space,
 * including its assigned status ({@code OCCUPIED}, {@code AVAILABLE}, or {@code NEEDS_REVIEW}),
 * empirical leaf class purity, 8 extracted features, structured {@link DecisionTreeModel.DecisionTrace},
 * vehicle detections &amp; spatial overlap evidence, and {@link SlotDiagnosticRecord}.
 *
 * <p>Memory Optimization (WS-4): Intermediate IVP stage representations (gray, smoothed, binary,
 * closed, edges, saturation, largestBlobMask) are NOT permanently retained per slot across
 * sessions. They are generated on demand via {@link FeatureExtractor#processWithStages} and cached
 * behind a {@link SoftReference} to keep memory bounded across hundreds of parking spaces.</p>
 */
public final class SlotAnalysisResult {

    private final ParkingSlot slot;
    private final SlotStatus status;
    private final double confidence;
    private final FeatureVector features;
    private final String decisionPath;
    private final DecisionTreeModel.DecisionTrace decisionTrace;
    private final BufferedImage roiImage;
    private final BufferedImage originalRoiImage;
    private final SlotDiagnosticRecord diagnosticRecord;
    private SoftReference<FeatureExtractor.ProcessedRoi> stageCache;

    public SlotAnalysisResult(ParkingSlot slot,
                              SlotStatus status,
                              FeatureVector features,
                              String decisionPath,
                              BufferedImage roiImage,
                              BufferedImage grayImage,
                              BufferedImage smoothedImage,
                              BufferedImage binaryImage,
                              BufferedImage closedImage,
                              BufferedImage edgeImage) {
        this(slot, status, 1.0, features, decisionPath,
            roiImage, grayImage, smoothedImage, binaryImage, closedImage, edgeImage, null, null, null, null, null);
    }

    public SlotAnalysisResult(ParkingSlot slot,
                              SlotStatus status,
                              double confidence,
                              FeatureVector features,
                              String decisionPath,
                              BufferedImage roiImage,
                              BufferedImage grayImage,
                              BufferedImage smoothedImage,
                              BufferedImage binaryImage,
                              BufferedImage closedImage,
                              BufferedImage edgeImage,
                              BufferedImage saturationImage,
                              BufferedImage largestBlobMask) {
        this(slot, status, confidence, features, decisionPath,
            roiImage, grayImage, smoothedImage, binaryImage, closedImage, edgeImage,
            saturationImage, largestBlobMask, null, null, null);
    }

    public SlotAnalysisResult(ParkingSlot slot,
                              SlotStatus status,
                              double confidence,
                              FeatureVector features,
                              String decisionPath,
                              BufferedImage roiImage,
                              BufferedImage grayImage,
                              BufferedImage smoothedImage,
                              BufferedImage binaryImage,
                              BufferedImage closedImage,
                              BufferedImage edgeImage,
                              BufferedImage saturationImage,
                              BufferedImage largestBlobMask,
                              DecisionTreeModel.DecisionTrace decisionTrace) {
        this(slot, status, confidence, features, decisionPath,
            roiImage, grayImage, smoothedImage, binaryImage, closedImage, edgeImage,
            saturationImage, largestBlobMask, decisionTrace, null, null);
    }

    public SlotAnalysisResult(ParkingSlot slot,
                              SlotStatus status,
                              double confidence,
                              FeatureVector features,
                              String decisionPath,
                              BufferedImage roiImage,
                              BufferedImage grayImage,
                              BufferedImage smoothedImage,
                              BufferedImage binaryImage,
                              BufferedImage closedImage,
                              BufferedImage edgeImage,
                              BufferedImage saturationImage,
                              BufferedImage largestBlobMask,
                              DecisionTreeModel.DecisionTrace decisionTrace,
                              SlotDiagnosticRecord diagnosticRecord) {
        this(slot, status, confidence, features, decisionPath,
            roiImage, grayImage, smoothedImage, binaryImage, closedImage, edgeImage,
            saturationImage, largestBlobMask, decisionTrace, diagnosticRecord, null);
    }

    public SlotAnalysisResult(ParkingSlot slot,
                              SlotStatus status,
                              double confidence,
                              FeatureVector features,
                              String decisionPath,
                              BufferedImage roiImage,
                              BufferedImage grayImage,
                              BufferedImage smoothedImage,
                              BufferedImage binaryImage,
                              BufferedImage closedImage,
                              BufferedImage edgeImage,
                              BufferedImage saturationImage,
                              BufferedImage largestBlobMask,
                              DecisionTreeModel.DecisionTrace decisionTrace,
                              SlotDiagnosticRecord diagnosticRecord,
                              BufferedImage originalRoiImage) {
        this.slot             = slot;
        this.status           = status;
        this.confidence       = confidence;
        this.features         = features;
        this.decisionPath     = decisionPath;
        this.roiImage         = roiImage;
        this.originalRoiImage = (originalRoiImage == roiImage) ? null : originalRoiImage;
        if (grayImage != null || binaryImage != null || edgeImage != null) {
            this.stageCache = new SoftReference<>(new FeatureExtractor.ProcessedRoi(
                originalRoiImage, roiImage, grayImage, smoothedImage, binaryImage, closedImage,
                edgeImage, saturationImage, largestBlobMask, features
            ));
        } else {
            this.stageCache = null;
        }
        this.decisionTrace    = decisionTrace != null ? decisionTrace : buildFallbackTrace(status, confidence, decisionPath);
        this.diagnosticRecord = diagnosticRecord != null ? diagnosticRecord
            : SlotDiagnosticRecord.fromComponents(
                slot, features, this.decisionTrace, null, null, null,
                "CART-v2.2-8feat", "Synthetic/Demo Corpus (Real-World Generalization NOT ESTABLISHED)");
    }

    private static DecisionTreeModel.DecisionTrace buildFallbackTrace(SlotStatus status, double purity, String path) {
        int occ = status == SlotStatus.OCCUPIED ? 1 : 0;
        int avail = status == SlotStatus.AVAILABLE ? 1 : 0;
        DecisionTreeModel.LeafSummary leaf = new DecisionTreeModel.LeafSummary(status, occ, avail, occ + avail, purity);
        return new DecisionTreeModel.DecisionTrace(
            List.of(), leaf, List.of(), status, status == SlotStatus.NEEDS_REVIEW,
            status == SlotStatus.NEEDS_REVIEW ? "Flagged for review" : "Standard prediction",
            path != null ? path : "", path != null ? path : ""
        );
    }

    private synchronized FeatureExtractor.ProcessedRoi getOrComputeStages() {
        FeatureExtractor.ProcessedRoi p = (stageCache != null) ? stageCache.get() : null;
        if (p == null) {
            BufferedImage enh = getRoiImage();
            BufferedImage raw = getOriginalRoiImage();
            if (enh != null) {
                p = FeatureExtractor.processWithStages(raw != null ? raw : enh, enh, false);
                stageCache = new SoftReference<>(p);
            }
        }
        return p;
    }

    public ParkingSlot getSlot()                                { return slot; }
    public int getId()                                          { return slot.getId(); }
    public SlotStatus getStatus()                               { return status; }
    /** Returns the empirical training-leaf class purity in {@code [0, 1]} (not a calibrated probability). */
    public double getConfidence()                               { return confidence; }
    public double getLeafPurity()                               { return confidence; }
    public FeatureVector getFeatures()                          { return features; }
    public String getDecisionPath()                             { return decisionPath; }
    public DecisionTreeModel.DecisionTrace getDecisionTrace()   { return decisionTrace; }
    public BufferedImage getRoiImage()                          { return roiImage; }
    public BufferedImage getEnhancedRoiImage()                  { return roiImage; }
    public BufferedImage getOriginalRoiImage()                  { return originalRoiImage != null ? originalRoiImage : roiImage; }

    public synchronized BufferedImage getGrayImage() {
        FeatureExtractor.ProcessedRoi p = getOrComputeStages();
        return p != null ? p.getGray() : null;
    }

    public synchronized BufferedImage getSmoothedImage() {
        FeatureExtractor.ProcessedRoi p = getOrComputeStages();
        return p != null ? p.getSmoothed() : null;
    }

    public synchronized BufferedImage getBinaryImage() {
        FeatureExtractor.ProcessedRoi p = getOrComputeStages();
        return p != null ? p.getBinary() : null;
    }

    public synchronized BufferedImage getClosedImage() {
        FeatureExtractor.ProcessedRoi p = getOrComputeStages();
        return p != null ? p.getClosed() : null;
    }

    public synchronized BufferedImage getEdgeImage() {
        FeatureExtractor.ProcessedRoi p = getOrComputeStages();
        return p != null ? p.getEdges() : null;
    }

    public synchronized BufferedImage getSaturationImage() {
        FeatureExtractor.ProcessedRoi p = getOrComputeStages();
        return p != null ? p.getSaturation() : null;
    }

    public synchronized BufferedImage getLargestBlobMask() {
        FeatureExtractor.ProcessedRoi p = getOrComputeStages();
        return p != null ? p.getLargestBlobMask() : null;
    }

    public SlotDiagnosticRecord getDiagnosticRecord()           { return diagnosticRecord; }

    public String getPredictionSource() {
        return diagnosticRecord != null ? diagnosticRecord.predictionSource() : "Feature-Based CART Decision Tree";
    }

    public String getReviewReason() {
        return diagnosticRecord != null ? diagnosticRecord.reviewReason() : "";
    }

    public SlotStatus getTreeStatus() {
        return diagnosticRecord != null ? diagnosticRecord.treePrediction() : status;
    }

    public SlotStatus getVehicleDetectorStatus() {
        return diagnosticRecord != null ? diagnosticRecord.vehicleDetectorPrediction() : SlotStatus.NEEDS_REVIEW;
    }

    public List<VehicleDetection> getContributingDetections() {
        return diagnosticRecord != null ? diagnosticRecord.contributingDetections() : List.of();
    }

    public SlotOverlapAnalyzer.SlotOverlapEvidence getOverlapEvidence() {
        return diagnosticRecord != null ? diagnosticRecord.overlapEvidence() : null;
    }

    public VehicleDetectionService.DetectorStatus getDetectorStatus() {
        return diagnosticRecord != null
            ? diagnosticRecord.detectorStatus()
            : VehicleDetectionService.DetectorStatus.MODEL_MISSING;
    }

    public OccupancyDecisionEngine.AnalysisMode getAnalysisMode() {
        return diagnosticRecord != null
            ? diagnosticRecord.analysisMode()
            : OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE;
    }

    /**
     * Formats this slot analysis result as a 12-column RFC 4180 CSV row using {@link Locale#ROOT}.
     */
    public String toCsvRow() {
        String escapedRule = decisionPath == null ? ""
            : decisionPath.replace("\r", " ").replace("\n", " ").replace("\"", "\"\"");
        return String.format(Locale.ROOT,
            "%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,\"%s\"\n",
            getId(),
            status,
            confidence,
            features.getMeanIntensity(),
            features.getForegroundRatio(),
            features.getDarkPixelRatio(),
            features.getEdgeDensity(),
            features.getObjectArea(),
            features.getMeanSaturation(),
            features.getSaturationRatio(),
            features.getLargestBlobRatio(),
            escapedRule
        );
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "SlotAnalysisResult[id=%d, status=%s, source=%s, leafPurity=%.0f%%, path=%s]",
            slot.getId(), status, getPredictionSource(), confidence * 100.0, decisionPath);
    }
}
