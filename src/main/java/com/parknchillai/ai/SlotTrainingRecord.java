package com.parknchillai.ai;

import com.parknchillai.app.SlotAnalysisResult;
import com.parknchillai.app.SlotDiagnosticRecord;
import com.parknchillai.detection.SlotOverlapAnalyzer;
import com.parknchillai.detection.VehicleDetection;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;

import java.time.Instant;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

/**
 * Complete per-slot training record generated from vehicle detection, slot geometry,
 * and IVP feature extraction (ParkNChill v2.3 Requirements 2, 3 &amp; 5).
 *
 * <p>Distinguishes three explicit provenance &amp; review states:
 * <ol>
 *   <li>{@link #PROVENANCE_PROVISIONAL_DETECTOR} ({@code provisional = true}):
 *       Automatically generated from live vehicle detection; strictly rejected from training
 *       until reviewed by a human operator via {@link #promoteToHumanVerified(String, SlotStatus)}.</li>
 *   <li>{@link #PROVENANCE_HUMAN_REVIEWED} ({@code provisional = false}):
 *       Human-confirmed slot record with an explicit non-blank {@code reviewerId}.</li>
 *   <li>{@link #PROVENANCE_IMPORTED_GROUND_TRUTH} ({@code provisional = false}):
 *       Validated external ground-truth annotation (e.g. PKLot XML/CSV) carrying
 *       {@code sourceDataset}, {@code sourceImageId}, {@code annotationFilePath}, {@code slotId},
 *       {@code sourceImageSha256}, {@code polygonCoordinatesSpec}, and {@code featureSchemaVersion},
 *       with an empty {@code reviewerId = ""} (never marked {@code HUMAN_VERIFIED} and never assigned
 *       a fabricated reviewer ID such as {@code PKLOT_ANNOTATION}). Any subsequent user correction
 *       is recorded via {@link #withAuditedUserCorrection(String, SlotStatus, String)} without
 *       overwriting {@link #originalSourceLabel()}.</li>
 * </ol>
 */
public record SlotTrainingRecord(
        String sampleId,
        String sourceImageId,
        String sourceImageSha256,
        String captureGroup,
        int slotId,
        int x,
        int y,
        int width,
        int height,
        ParkingSlot.GeometryOrigin geometryOrigin,
        String vehicleDetectionEvidence,
        double vehicleOverlapRatio,
        double vehicleIou,
        boolean vehicleCenterInSlot,
        boolean vehicleMatchedToSlot,
        FeatureVector features,
        SlotStatus occupancyLabel,
        boolean provisional,
        String reviewerId,
        String reviewStatus,
        String provenance,
        String featureSchemaVersion,
        String modelVersion,
        String partition,
        String timestamp,
        String sourceDataset,
        String annotationFilePath,
        SlotStatus originalSourceLabel,
        String userCorrectionAuditEvent,
        String polygonCoordinatesSpec
) {
    public static final String PROVENANCE_PROVISIONAL_DETECTOR  = "PROVISIONAL_DETECTOR_OUTPUT";
    public static final String PROVENANCE_HUMAN_REVIEWED        = "HUMAN_REVIEWED_SLOT_DETECTION";
    public static final String PROVENANCE_IMPORTED_GROUND_TRUTH = "IMPORTED_GROUND_TRUTH";

    /**
     * Explicit policy governing which non-provisional records are eligible to train the Decision Tree
     * (Requirement 3).
     */
    public enum TrainingEligibilityPolicy {
        /** Accepts only human-reviewed records ({@link #isHumanVerified()}). */
        HUMAN_VERIFIED_ONLY,
        /** Accepts both human-reviewed records and validated external ground-truth records ({@link #isImportedGroundTruth()}). */
        HUMAN_VERIFIED_AND_IMPORTED_GROUND_TRUTH
    }

    /**
     * Backward-compatible 25-argument constructor for interactive session records.
     */
    public SlotTrainingRecord(
            String sampleId,
            String sourceImageId,
            String sourceImageSha256,
            String captureGroup,
            int slotId,
            int x,
            int y,
            int width,
            int height,
            ParkingSlot.GeometryOrigin geometryOrigin,
            String vehicleDetectionEvidence,
            double vehicleOverlapRatio,
            double vehicleIou,
            boolean vehicleCenterInSlot,
            boolean vehicleMatchedToSlot,
            FeatureVector features,
            SlotStatus occupancyLabel,
            boolean provisional,
            String reviewerId,
            String reviewStatus,
            String provenance,
            String featureSchemaVersion,
            String modelVersion,
            String partition,
            String timestamp) {
        this(
            sampleId, sourceImageId, sourceImageSha256, captureGroup, slotId,
            x, y, width, height, geometryOrigin, vehicleDetectionEvidence,
            vehicleOverlapRatio, vehicleIou, vehicleCenterInSlot, vehicleMatchedToSlot,
            features, occupancyLabel, provisional, reviewerId, reviewStatus,
            provenance, featureSchemaVersion, modelVersion, partition, timestamp,
            PROVENANCE_IMPORTED_GROUND_TRUTH.equals(provenance) ? "PKLot" : "InteractiveSession",
            "",
            occupancyLabel,
            "",
            String.format(Locale.ROOT, "%d.0,%d.0;%d.0,%d.0;%d.0,%d.0;%d.0,%d.0",
                x, y, x + width, y, x + width, y + height, x, y + height)
        );
    }

    public SlotTrainingRecord {
        if (sampleId == null || sampleId.isBlank()) {
            throw new IllegalArgumentException("sampleId must not be null or blank.");
        }
        if (sourceImageId == null || sourceImageId.isBlank()) {
            throw new IllegalArgumentException("sourceImageId must not be null or blank.");
        }
        if (captureGroup == null || captureGroup.isBlank()) {
            throw new IllegalArgumentException("captureGroup must not be null or blank.");
        }
        if (slotId <= 0 || width <= 0 || height <= 0 || x < 0 || y < 0) {
            throw new IllegalArgumentException("Invalid slotId or slot geometry.");
        }
        if (features == null || !features.hasAllValuesWithinValidBounds()) {
            throw new IllegalArgumentException("features must be non-null and within valid mathematical bounds.");
        }
        if (occupancyLabel == null) {
            throw new IllegalArgumentException("occupancyLabel must not be null.");
        }
        if (provenance == null || provenance.isBlank()) {
            provenance = provisional ? PROVENANCE_PROVISIONAL_DETECTOR : PROVENANCE_HUMAN_REVIEWED;
        }
        if (reviewerId != null && "PKLOT_ANNOTATION".equalsIgnoreCase(reviewerId.trim())) {
            throw new IllegalArgumentException(
                "Fabricated reviewerId 'PKLOT_ANNOTATION' is prohibited. Imported PKLot records must use provenance IMPORTED_GROUND_TRUTH with empty reviewerId.");
        }
        if (!provisional) {
            if (occupancyLabel != SlotStatus.OCCUPIED && occupancyLabel != SlotStatus.AVAILABLE) {
                throw new IllegalArgumentException(
                    "Non-provisional training record must have occupancyLabel OCCUPIED or AVAILABLE; got " + occupancyLabel);
            }
            if (PROVENANCE_IMPORTED_GROUND_TRUTH.equals(provenance)) {
                boolean hasUserCorrection = userCorrectionAuditEvent != null && !userCorrectionAuditEvent.isBlank();
                if (!hasUserCorrection && reviewerId != null && !reviewerId.isBlank()) {
                    throw new IllegalArgumentException(
                        "Unedited IMPORTED_GROUND_TRUTH record must not carry a fabricated reviewerId; got '" + reviewerId + "'.");
                }
                if (hasUserCorrection && (reviewerId == null || reviewerId.isBlank() || "UNKNOWN".equalsIgnoreCase(reviewerId.trim()))) {
                    throw new IllegalArgumentException(
                        "Audited user correction on an imported record requires an explicit human reviewerId.");
                }
                if (reviewStatus != null && reviewStatus.startsWith("HUMAN_VERIFIED") && !hasUserCorrection) {
                    throw new IllegalArgumentException(
                        "Unedited IMPORTED_GROUND_TRUTH record must not be marked HUMAN_VERIFIED.");
                }
            } else {
                if (reviewerId == null || reviewerId.isBlank() || "UNKNOWN".equalsIgnoreCase(reviewerId.trim())) {
                    throw new IllegalArgumentException(
                        "Human-verified training record requires an explicit non-blank reviewerId.");
                }
            }
        }
        if (featureSchemaVersion == null || featureSchemaVersion.isBlank()) {
            featureSchemaVersion = features.getSchemaVersion();
        }
        if (modelVersion == null || modelVersion.isBlank()) {
            modelVersion = "ONNX-Detector+CART-v2.3";
        }
        if (partition == null || partition.isBlank()) {
            partition = "TRAIN";
        }
        if (timestamp == null || timestamp.isBlank()) {
            timestamp = Instant.now().toString();
        }
        if (sourceDataset == null || sourceDataset.isBlank()) {
            sourceDataset = PROVENANCE_IMPORTED_GROUND_TRUTH.equals(provenance) ? "PKLot" : "InteractiveSession";
        }
        if (annotationFilePath == null) {
            annotationFilePath = "";
        }
        if (originalSourceLabel == null) {
            originalSourceLabel = occupancyLabel;
        }
        if (userCorrectionAuditEvent == null) {
            userCorrectionAuditEvent = "";
        }
        if (polygonCoordinatesSpec == null || polygonCoordinatesSpec.isBlank()) {
            polygonCoordinatesSpec = String.format(Locale.ROOT, "%d.0,%d.0;%d.0,%d.0;%d.0,%d.0;%d.0,%d.0",
                x, y, x + width, y, x + width, y + height, x, y + height);
        }
    }

    /**
     * Returns {@code true} only if this record was confirmed or corrected by a genuine human reviewer
     * with an explicit {@code reviewerId}. Unedited {@code IMPORTED_GROUND_TRUTH} records return {@code false}.
     */
    public boolean isHumanVerified() {
        if (provisional) return false;
        if (occupancyLabel != SlotStatus.OCCUPIED && occupancyLabel != SlotStatus.AVAILABLE) return false;
        if (PROVENANCE_IMPORTED_GROUND_TRUTH.equals(provenance)) {
            return userCorrectionAuditEvent != null && !userCorrectionAuditEvent.isBlank()
                && reviewerId != null && !reviewerId.isBlank();
        }
        return reviewerId != null && !reviewerId.isBlank();
    }

    /**
     * Returns {@code true} if this record is a validated external ground-truth annotation
     * ({@code provenance = IMPORTED_GROUND_TRUTH}) with complete source metadata and checksum.
     */
    public boolean isImportedGroundTruth() {
        return !provisional
            && PROVENANCE_IMPORTED_GROUND_TRUTH.equals(provenance)
            && (occupancyLabel == SlotStatus.OCCUPIED || occupancyLabel == SlotStatus.AVAILABLE)
            && sourceDataset != null && !sourceDataset.isBlank()
            && annotationFilePath != null && !annotationFilePath.isBlank()
            && sourceImageSha256 != null && !sourceImageSha256.isBlank()
            && !"UNHASHED_MEMORY_IMAGE".equals(sourceImageSha256);
    }

    /**
     * Evaluates whether this record is eligible for Decision Tree training under {@code policy}.
     * Provisional detector predictions ({@code provisional == true}) are ALWAYS rejected.
     */
    public boolean isEligibleForTraining(TrainingEligibilityPolicy policy) {
        if (provisional) return false;
        TrainingEligibilityPolicy activePolicy = policy != null
            ? policy
            : TrainingEligibilityPolicy.HUMAN_VERIFIED_AND_IMPORTED_GROUND_TRUTH;
        return switch (activePolicy) {
            case HUMAN_VERIFIED_ONLY -> isHumanVerified();
            case HUMAN_VERIFIED_AND_IMPORTED_GROUND_TRUTH -> isHumanVerified() || isImportedGroundTruth();
        };
    }

    /**
     * Creates a validated {@code IMPORTED_GROUND_TRUTH} record from an external dataset (such as PKLot)
     * without marking it {@code HUMAN_VERIFIED} or attaching a fabricated reviewer ID (Requirement 3).
     */
    public static SlotTrainingRecord createImportedGroundTruthRecord(
            String sourceDataset,
            String sourceImageId,
            String annotationFilePath,
            String sourceImageSha256,
            String captureGroup,
            ParkingSlot slot,
            String vehicleDetectionEvidence,
            double vehicleOverlapRatio,
            double vehicleIou,
            boolean vehicleCenterInSlot,
            boolean vehicleMatchedToSlot,
            FeatureVector features,
            SlotStatus groundTruthLabel,
            String modelVersion,
            String partition) {
        if (sourceDataset == null || sourceDataset.isBlank()) {
            throw new IllegalArgumentException("sourceDataset must not be null or blank for imported ground truth.");
        }
        if (annotationFilePath == null || annotationFilePath.isBlank()) {
            throw new IllegalArgumentException("annotationFilePath must not be null or blank for imported ground truth.");
        }
        if (sourceImageSha256 == null || sourceImageSha256.isBlank()) {
            throw new IllegalArgumentException("sourceImageSha256 must not be null or blank for imported ground truth.");
        }
        if (slot == null) {
            throw new IllegalArgumentException("slot must not be null.");
        }
        String schemaVer = features.getSchemaVersion();
        String sampleId = DatasetManifest.computeSampleId(
            sourceImageId, sourceImageSha256, captureGroup, slot.getId(),
            slot.getX(), slot.getY(), slot.getWidth(), slot.getHeight(), schemaVer
        );
        String polySpec = slot.formatPolygonCoordinates();
        String revStatus = String.format(Locale.ROOT,
            "IMPORTED_GROUND_TRUTH[dataset=%s;ann=%s;sha256=%s;origLabel=%s;det=%s;ios=%.3f]",
            sourceDataset.trim(), annotationFilePath.trim(), sourceImageSha256.trim(),
            groundTruthLabel.name(), vehicleDetectionEvidence, vehicleOverlapRatio);

        return new SlotTrainingRecord(
            sampleId,
            sourceImageId.trim(),
            sourceImageSha256.trim(),
            captureGroup.trim(),
            slot.getId(),
            slot.getX(),
            slot.getY(),
            slot.getWidth(),
            slot.getHeight(),
            slot.getOrigin(),
            vehicleDetectionEvidence,
            vehicleOverlapRatio,
            vehicleIou,
            vehicleCenterInSlot,
            vehicleMatchedToSlot,
            features,
            groundTruthLabel,
            false,
            "", // Never attach a fabricated reviewerId to unedited imported ground truth!
            revStatus,
            PROVENANCE_IMPORTED_GROUND_TRUTH,
            schemaVer,
            modelVersion != null ? modelVersion : "PKLot-ONNX-v2.3",
            partition != null ? partition.trim().toUpperCase(Locale.ROOT) : "TRAIN",
            Instant.now().toString(),
            sourceDataset.trim(),
            annotationFilePath.trim(),
            groundTruthLabel,
            "",
            polySpec
        );
    }

    /**
     * Records an auditable user correction on an imported or existing record while preserving
     * {@link #originalSourceLabel()} and {@link #annotationFilePath()} untouched (Requirement 3).
     */
    public SlotTrainingRecord withAuditedUserCorrection(String reviewerId,
                                                        SlotStatus correctedLabel,
                                                        String reason) {
        if (reviewerId == null || reviewerId.isBlank()
                || "UNKNOWN".equalsIgnoreCase(reviewerId.trim())
                || "PKLOT_ANNOTATION".equalsIgnoreCase(reviewerId.trim())) {
            throw new IllegalArgumentException("Audited user correction requires an explicit human reviewerId.");
        }
        if (correctedLabel != SlotStatus.OCCUPIED && correctedLabel != SlotStatus.AVAILABLE) {
            throw new IllegalArgumentException("Corrected label must be OCCUPIED or AVAILABLE; got " + correctedLabel);
        }
        String cleanReviewer = reviewerId.trim();
        String cleanReason = (reason == null || reason.isBlank()) ? "Manual reviewer verification/correction" : reason.trim();
        String now = Instant.now().toString();
        String auditEvent = String.format(Locale.ROOT,
            "USER_CORRECTION[by=%s;origLabel=%s;correctedLabel=%s;reason=%s;at=%s]",
            cleanReviewer, this.originalSourceLabel.name(), correctedLabel.name(), cleanReason, now);
        String updatedRevStatus = this.reviewStatus + "|" + auditEvent;

        return new SlotTrainingRecord(
            sampleId,
            sourceImageId,
            sourceImageSha256,
            captureGroup,
            slotId,
            x,
            y,
            width,
            height,
            geometryOrigin,
            vehicleDetectionEvidence,
            vehicleOverlapRatio,
            vehicleIou,
            vehicleCenterInSlot,
            vehicleMatchedToSlot,
            features,
            correctedLabel,
            false,
            cleanReviewer,
            updatedRevStatus,
            this.provenance,
            featureSchemaVersion,
            modelVersion,
            partition,
            now,
            sourceDataset,
            annotationFilePath,
            this.originalSourceLabel, // Preserved without overwriting!
            auditEvent,
            polygonCoordinatesSpec
        );
    }

    public static SlotTrainingRecord createProvisionalFromSlotResult(
            String sourceImageId,
            String sourceImageSha256,
            String captureGroup,
            SlotAnalysisResult slotResult,
            String modelVersion) {
        if (slotResult == null) {
            throw new IllegalArgumentException("slotResult must not be null.");
        }
        ParkingSlot slot = slotResult.getSlot();
        SlotDiagnosticRecord diag = slotResult.getDiagnosticRecord();
        SlotOverlapAnalyzer.SlotOverlapEvidence ev = diag != null ? diag.overlapEvidence() : null;

        double ios = ev != null ? ev.bestIoS() : 0.0;
        double iou = ev != null ? ev.bestIou() : 0.0;
        boolean centerIn = ev != null && ev.centerInsideSlot();
        boolean matched = ev != null && (ev.status() == SlotStatus.OCCUPIED
            || (!ev.contributingDetections().isEmpty() && ios >= 0.26));

        FeatureVector enrichedFeatures = slotResult.getFeatures().withDetectionFeatures(ios, matched);
        String schemaVer = FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION;
        String cleanSource = (sourceImageId == null || sourceImageId.isBlank()) ? "live_image.png" : sourceImageId.trim();
        String cleanGroup = (captureGroup == null || captureGroup.isBlank()) ? "GRP_LIVE_SESSION" : captureGroup.trim();
        String cleanSha = (sourceImageSha256 == null || sourceImageSha256.isBlank()) ? "UNHASHED_MEMORY_IMAGE" : sourceImageSha256.trim();

        String sampleId = DatasetManifest.computeSampleId(
            cleanSource, cleanSha, cleanGroup, slot.getId(),
            slot.getX(), slot.getY(), slot.getWidth(), slot.getHeight(), schemaVer
        );

        String detEv = formatDetectionEvidence(ev);
        String revStatus = String.format(Locale.ROOT,
            "PROVISIONAL_AUTO_DETECTED[sha256=%s;det=%s;ios=%.3f]", cleanSha, detEv, ios);

        return new SlotTrainingRecord(
            sampleId,
            cleanSource,
            cleanSha,
            cleanGroup,
            slot.getId(),
            slot.getX(),
            slot.getY(),
            slot.getWidth(),
            slot.getHeight(),
            slot.getOrigin(),
            detEv,
            ios,
            iou,
            centerIn,
            matched,
            enrichedFeatures,
            slotResult.getStatus(),
            true,
            "",
            revStatus,
            PROVENANCE_PROVISIONAL_DETECTOR,
            schemaVer,
            modelVersion != null ? modelVersion : "ONNX-Detector+CART-v2.3",
            "TRAIN",
            Instant.now().toString(),
            "InteractiveSession",
            "",
            slotResult.getStatus(),
            "",
            slot.formatPolygonCoordinates()
        );
    }

    public SlotTrainingRecord promoteToHumanVerified(String reviewerId, SlotStatus verifiedLabel) {
        return promoteToHumanVerified(reviewerId, verifiedLabel, this.partition);
    }

    public SlotTrainingRecord promoteToHumanVerified(String reviewerId,
                                                     SlotStatus verifiedLabel,
                                                     String targetPartition) {
        if (reviewerId == null || reviewerId.isBlank()
                || "UNKNOWN".equalsIgnoreCase(reviewerId.trim())
                || "PKLOT_ANNOTATION".equalsIgnoreCase(reviewerId.trim())) {
            throw new IllegalArgumentException("Explicit non-blank human reviewerId is required to verify a training record.");
        }
        if (verifiedLabel != SlotStatus.OCCUPIED && verifiedLabel != SlotStatus.AVAILABLE) {
            throw new IllegalArgumentException("Verified training label must be OCCUPIED or AVAILABLE; got " + verifiedLabel);
        }
        String cleanReviewer = reviewerId.trim();
        String revStatus = String.format(Locale.ROOT,
            "HUMAN_VERIFIED[sha256=%s;by=%s;det=%s;ios=%.3f;matched=%d]",
            sourceImageSha256, cleanReviewer, vehicleDetectionEvidence,
            vehicleOverlapRatio, vehicleMatchedToSlot ? 1 : 0);
        return new SlotTrainingRecord(
            sampleId,
            sourceImageId,
            sourceImageSha256,
            captureGroup,
            slotId,
            x,
            y,
            width,
            height,
            geometryOrigin,
            vehicleDetectionEvidence,
            vehicleOverlapRatio,
            vehicleIou,
            vehicleCenterInSlot,
            vehicleMatchedToSlot,
            features,
            verifiedLabel,
            false,
            cleanReviewer,
            revStatus,
            PROVENANCE_HUMAN_REVIEWED,
            featureSchemaVersion,
            modelVersion,
            targetPartition != null ? targetPartition.trim().toUpperCase(Locale.ROOT) : "TRAIN",
            Instant.now().toString(),
            sourceDataset,
            annotationFilePath,
            this.originalSourceLabel,
            "",
            polygonCoordinatesSpec
        );
    }

    private static String formatDetectionEvidence(SlotOverlapAnalyzer.SlotOverlapEvidence ev) {
        if (ev == null || ev.contributingDetections() == null || ev.contributingDetections().isEmpty()) {
            return "NONE";
        }
        List<VehicleDetection> dets = ev.contributingDetections();
        return dets.stream()
            .map(d -> String.format(Locale.ROOT, "%s(%s:%.2f)",
                d.detectionId(), d.category().name(), d.detectionScore()))
            .collect(Collectors.joining("|"));
    }
}
