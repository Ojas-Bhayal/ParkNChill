package com.parknchillai.ai;

import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.SlotStatus;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Validates training and evaluation datasets prior to model fitting, checking
 * schema compatibility, finite values, mathematical feature bounds, duplicate rows,
 * class representation, capture-group disjointness, and evidence-based data provenance
 * (PNC-FIX-03, PNC-REM-01, PNC-REM-04, PNC-REM-06).
 *
 * <p>Keeps five provenance categories distinct:
 * <ul>
 *   <li>{@code demoExtractedSamples} ({@code DEMO_LOT_EXTRACTED})</li>
 *   <li>{@code proceduralSyntheticSamples} ({@code PROCEDURAL_SYNTHETIC})</li>
 *   <li>{@code legacyImportedSamples} ({@code LEGACY_IMPORTED} / {@code INFERRED_MIGRATION})</li>
 *   <li>{@code userDeclaredSamples} ({@code USER_LABELLED} / {@code USER_DECLARED_REAL_CAMERA})</li>
 *   <li>{@code realCameraVerifiedSamples} ({@code VERIFIED_REAL_CAMERA} passing all file, decode,
 *       ROI-bounds, SHA-256 content hash, explicit reviewer, and non-mixed-group checks)</li>
 * </ul>
 */
public final class DatasetValidator {

    public static final int MIN_VERIFIED_REAL_SAMPLES = 10;
    public static final int MIN_VERIFIED_REAL_GROUPS  = 3;

    private DatasetValidator() { /* utility class */ }

    public record ValidationSummary(
            boolean validForTraining,
            int totalSamples,
            int occupiedSamples,
            int availableSamples,
            int featureCount,
            int duplicateSampleCount,
            int distinctCaptureGroups,
            int demoExtractedSamples,
            int proceduralSyntheticSamples,
            int legacyImportedSamples,
            int userDeclaredSamples,
            int realCameraVerifiedSamples,
            int verifiedRealCameraGroups,
            boolean realImageSamplesPresent,
            boolean groupDisjointEvaluationPossible,
            boolean realWorldGeneralizationEstablished,
            List<String> errors,
            List<String> warnings
    ) {
        /**
         * Backward-compatible accessor returning {@link #userDeclaredSamples()}.
         */
        public int userLabelledSamples() {
            return userDeclaredSamples;
        }

        /**
         * Backward-compatible accessor returning {@link #realWorldGeneralizationEstablished()}.
         */
        public boolean hasVerifiedRealCameraData() {
            return realWorldGeneralizationEstablished;
        }

        /**
         * Evaluates whether real-world camera generalization can be established after running
         * the held-out test protocol on genuine verified real-camera groups.
         * Inventory counts or file hashes alone never set {@code realWorldGeneralizationEstablished = true}.
         */
        public ValidationSummary withEvaluatedRealCameraTestResult(
                ModelEvaluator.EvaluationReport heldOutTestReport,
                Set<String> evaluatedTestGroups,
                Set<String> verifiedRealGroupsSet) {
            boolean testIsVerifiedReal = evaluatedTestGroups != null
                && !evaluatedTestGroups.isEmpty()
                && verifiedRealGroupsSet != null
                && verifiedRealGroupsSet.containsAll(evaluatedTestGroups);
            boolean passesAcceptance = heldOutTestReport != null
                && heldOutTestReport.metricsAvailable
                && heldOutTestReport.scope == ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION
                && heldOutTestReport.f1 >= 0.85
                && heldOutTestReport.falseAvailableCount() == 0;
            boolean established = validForTraining
                && realImageSamplesPresent
                && groupDisjointEvaluationPossible
                && testIsVerifiedReal
                && passesAcceptance;
            return new ValidationSummary(
                validForTraining, totalSamples, occupiedSamples, availableSamples,
                featureCount, duplicateSampleCount, distinctCaptureGroups,
                demoExtractedSamples, proceduralSyntheticSamples, legacyImportedSamples,
                userDeclaredSamples, realCameraVerifiedSamples, verifiedRealCameraGroups,
                realImageSamplesPresent, groupDisjointEvaluationPossible, established,
                errors, warnings
            );
        }

        public String formatProvenanceSummary() {
            return String.format(Locale.ROOT,
                "Total: %d samples (%d OCCUPIED, %d AVAILABLE across %d capture groups) | " +
                "Provenance: %d Demo-Lot, %d Procedural Synthetic, %d Legacy-Imported, %d User-Declared, %d Verified Real-Camera (%d Real Groups) | " +
                "Real-Data Inventory: %s | Group-Disjoint Split: %s | Real-World Camera Generalization: %s",
                totalSamples, occupiedSamples, availableSamples, distinctCaptureGroups,
                demoExtractedSamples, proceduralSyntheticSamples, legacyImportedSamples,
                userDeclaredSamples, realCameraVerifiedSamples, verifiedRealCameraGroups,
                realImageSamplesPresent ? "PRESENT" : "ABSENT",
                groupDisjointEvaluationPossible ? "VALID" : "UNAVAILABLE",
                realWorldGeneralizationEstablished ? "ESTABLISHED" : "NOT ESTABLISHED (Synthetic/Demo Corpus)");
        }
    }

    /**
     * Validates a loaded {@link DatasetLoader.Dataset} and its manifest records,
     * throwing an {@link IllegalArgumentException} with a clear diagnostic message if
     * training cannot proceed safely.
     */
    public static ValidationSummary validateOrThrow(DatasetLoader.Dataset dataset,
                                                    List<DatasetManifest.ManifestRecord> manifest) {
        ValidationSummary summary = validate(dataset, manifest);
        if (!summary.validForTraining()) {
            throw new IllegalArgumentException(
                "Dataset validation failed: " + String.join("; ", summary.errors()));
        }
        return summary;
    }

    public static ValidationSummary validate(DatasetLoader.Dataset dataset,
                                             List<DatasetManifest.ManifestRecord> manifest) {
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        if (dataset == null || dataset.size() == 0) {
            errors.add("Dataset is null or contains 0 samples.");
            return new ValidationSummary(
                false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, false, false, errors, warnings);
        }

        int total = dataset.size();
        int fCount = dataset.featureCount();
        if (fCount != FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT
                && fCount != FeatureVector.FEATURE_COUNT
                && fCount != FeatureVector.LEGACY_FEATURE_COUNT) {
            errors.add("Unsupported feature count: " + fCount + " (expected 10, 8 or legacy 5).");
        }

        int occ = 0;
        int avail = 0;
        int dupCount = 0;
        Set<String> seenRows = new HashSet<>();

        for (int r = 0; r < total; r++) {
            double[] row = dataset.features[r];
            SlotStatus lbl = dataset.labels[r];
            if (lbl == SlotStatus.OCCUPIED) occ++;
            else if (lbl == SlotStatus.AVAILABLE) avail++;
            else errors.add("Row " + r + " has invalid training label: " + lbl);

            if (row == null || row.length != fCount) {
                errors.add("Row " + r + " has inconsistent feature length.");
                continue;
            }

            StringBuilder rowKey = new StringBuilder();
            for (int c = 0; c < row.length; c++) {
                double v = row[c];
                if (!Double.isFinite(v)) {
                    errors.add("Row " + r + ", feature " + c + " is not finite: " + v);
                } else if (c < FeatureVector.SPECS.size() && !FeatureVector.SPECS.get(c).isWithinValidBounds(v)) {
                    errors.add(String.format(Locale.ROOT,
                        "Row %d, feature %s (%.4f) is outside valid bounds [%.1f, %.1f].",
                        r, FeatureVector.SPECS.get(c).key(), v,
                        FeatureVector.SPECS.get(c).minValid(), FeatureVector.SPECS.get(c).maxValid()));
                }
                rowKey.append(String.format(Locale.ROOT, "%.6f,", v));
            }
            if (!seenRows.add(rowKey.toString())) {
                dupCount++;
            }
        }

        if (occ < 2 || avail < 2) {
            errors.add(String.format(Locale.ROOT,
                "Both classes must have at least 2 samples for training and evaluation (got OCCUPIED=%d, AVAILABLE=%d).",
                occ, avail));
        }
        if (dupCount > 0) {
            warnings.add(dupCount + " duplicate feature vector(s) detected in dataset.");
        }

        int demoCount = 0;
        int synthCount = 0;
        int legacyCount = 0;
        int userDeclaredCount = 0;
        int verifiedRealCameraCount = 0;
        Set<String> allValidGroups = new LinkedHashSet<>();
        Set<String> verifiedRealGroups = new LinkedHashSet<>();
        Set<String> nonRealGroups = new LinkedHashSet<>();
        Map<String, Integer> realCountPerGroup = new HashMap<>();
        Map<String, String> groupPartitionMap = new HashMap<>();
        boolean partitionLeakageDetected = false;
        boolean hasUnknownLegacyGroups = false;

        if (manifest != null && !manifest.isEmpty()) {
            try {
                DatasetManifest.verifyStrictAlignment(dataset, manifest);
            } catch (Exception alignEx) {
                errors.add(alignEx.getMessage());
            }

            for (int i = 0; i < manifest.size(); i++) {
                DatasetManifest.ManifestRecord m = manifest.get(i);
                String grp = m.captureGroup() != null ? m.captureGroup().trim() : "";
                if (grp.isEmpty() || "UNKNOWN".equalsIgnoreCase(grp) || "DEFAULT_GROUP".equalsIgnoreCase(grp)) {
                    errors.add("Row " + i + " (" + m.sampleId() + ") has missing or placeholder captureGroup.");
                } else if ("LEGACY_UNGROUPED".equalsIgnoreCase(grp) || "GRP_MIGRATED_UNVERIFIED".equalsIgnoreCase(grp)) {
                    hasUnknownLegacyGroups = true;
                    warnings.add("Row " + i + " (" + m.sampleId() + ") has unverified legacy captureGroup '" + grp + "'; group-disjoint evaluation disabled.");
                } else {
                    allValidGroups.add(grp);
                    String part = m.partition() != null ? m.partition().trim().toUpperCase(Locale.ROOT) : "TRAIN";
                    String existingPart = groupPartitionMap.putIfAbsent(grp, part);
                    if (existingPart != null && !existingPart.equals(part)) {
                        partitionLeakageDetected = true;
                        errors.add("Capture group '" + grp + "' is split across partitions " + existingPart + " and " + part + ".");
                    }
                }

                String prov = m.provenance() != null ? m.provenance().trim().toUpperCase(Locale.ROOT) : "";
                String src  = m.sourceImageId() != null ? m.sourceImageId().trim() : "";

                if ("DEMO_LOT_EXTRACTED".equals(prov) || prov.contains("DEMO")) {
                    demoCount++;
                    nonRealGroups.add(grp);
                    if (!src.startsWith("procedural://") && !new File(src).exists()) {
                        warnings.add("Referenced demo source image not found on disk: " + src);
                    }
                } else if ("LEGACY_IMPORTED".equals(prov) || "INFERRED_MIGRATION".equals(prov) || prov.contains("LEGACY") || prov.contains("INFERRED")) {
                    legacyCount++;
                    nonRealGroups.add(grp);
                } else if ("PROCEDURAL_SYNTHETIC".equals(prov) || prov.contains("SYNTHETIC") || prov.contains("PROCEDURAL")) {
                    synthCount++;
                    nonRealGroups.add(grp);
                } else if ("USER_LABELLED".equals(prov) || "USER_DECLARED_REAL_CAMERA".equals(prov)) {
                    userDeclaredCount++;
                    nonRealGroups.add(grp);
                } else if ("VERIFIED_REAL_CAMERA".equals(prov) || prov.contains("REAL_CAMERA")) {
                    String rejectReason = verifyRealCameraEvidence(m);
                    if (rejectReason == null) {
                        verifiedRealCameraCount++;
                        if (!grp.isEmpty()) {
                            verifiedRealGroups.add(grp);
                            realCountPerGroup.merge(grp, 1, Integer::sum);
                        }
                    } else {
                        userDeclaredCount++;
                        nonRealGroups.add(grp);
                        warnings.add("Sample " + m.sampleId() + " rejected from verified real-camera subset: " + rejectReason);
                    }
                } else {
                    synthCount++;
                    nonRealGroups.add(grp);
                    warnings.add("Sample " + m.sampleId() + " has unrecognised provenance '" + prov + "'; treated as unverified.");
                }
            }
        } else {
            synthCount = total;
            warnings.add("No provenance manifest supplied; treating samples as unverified synthetic data.");
        }

        // Remove any capture group from verifiedRealGroups (and subtract its samples from verifiedRealCameraCount)
        // if that group was ambiguously reused for synthetic, demo, legacy, or unverified records (PNC-REM-04)
        for (String mixedGrp : nonRealGroups) {
            if (verifiedRealGroups.remove(mixedGrp)) {
                int removedSamples = realCountPerGroup.getOrDefault(mixedGrp, 0);
                verifiedRealCameraCount = Math.max(0, verifiedRealCameraCount - removedSamples);
                userDeclaredCount += removedSamples;
                warnings.add("Capture group '" + mixedGrp + "' mixes synthetic/demo/unverified and real-camera records; excluded from verified real-camera metrics.");
            }
        }

        boolean realInventoryPresent = verifiedRealCameraCount >= MIN_VERIFIED_REAL_SAMPLES
                                    && verifiedRealGroups.size() >= MIN_VERIFIED_REAL_GROUPS;
        boolean groupDisjointPossible = !hasUnknownLegacyGroups
                                     && allValidGroups.size() >= 3
                                     && !partitionLeakageDetected
                                     && occ >= 2 && avail >= 2;

        // Important (PNC-FIX-03, PNC-REM-04): Inventory counts and SHA-256 hashes alone never establish real-world generalization
        boolean generalizationEstablished = false;
        if (!realInventoryPresent) {
            warnings.add(String.format(Locale.ROOT,
                "Dataset has %d verified real-camera samples across %d verified real capture groups (requires >= %d samples across >= %d real groups); real-world camera generalization is NOT ESTABLISHED.",
                verifiedRealCameraCount, verifiedRealGroups.size(), MIN_VERIFIED_REAL_SAMPLES, MIN_VERIFIED_REAL_GROUPS));
        } else {
            warnings.add("Verified real-camera inventory is present (note: SHA-256 hashes verify file consistency, not physical camera origin or label accuracy), and real-world generalization remains NOT ESTABLISHED until held-out evaluation on verified real-camera test groups passes acceptance criteria.");
        }

        return new ValidationSummary(
            errors.isEmpty(),
            total,
            occ,
            avail,
            fCount,
            dupCount,
            allValidGroups.size(),
            demoCount,
            synthCount,
            legacyCount,
            userDeclaredCount,
            verifiedRealCameraCount,
            verifiedRealGroups.size(),
            realInventoryPresent,
            groupDisjointPossible,
            generalizationEstablished,
            Collections.unmodifiableList(errors),
            Collections.unmodifiableList(warnings)
        );
    }

    /**
     * Verifies traceable evidence for a sample claiming {@code REAL_CAMERA} provenance (PNC-REM-04):
     * <ol>
     *   <li>Non-blank, non-synthetic, non-demo, non-legacy {@code sourceImageId} and {@code captureGroup}.</li>
     *   <li>Explicit human review status ({@code USER_REVIEWED} or {@code HUMAN_VERIFIED}) with an
     *       explicit non-blank {@code by=<reviewerId>} attribute (never inferred from a status prefix).</li>
     *   <li>Source image file exists on disk, decodes via {@link ImageIO#read} to {@code >= 10x10} px,
     *       and the record's ROI {@code (x, y, width, height)} lies strictly inside the decoded image bounds.</li>
     *   <li>Stored 64-character SHA-256 content hash in {@code reviewStatus} matches the on-disk file's SHA-256 digest.</li>
     * </ol>
     *
     * @return {@code null} if all evidence checks pass, or a descriptive rejection reason string.
     */
    public static String verifyRealCameraEvidence(DatasetManifest.ManifestRecord m) {
        if (m == null) return "ManifestRecord is null.";
        String src = m.sourceImageId() != null ? m.sourceImageId().trim() : "";
        if (src.isEmpty() || "UNKNOWN".equalsIgnoreCase(src)
                || src.startsWith("procedural://") || src.startsWith("unverified://")
                || src.startsWith("legacy://") || src.endsWith("sample_parking_lot.png")) {
            return "sourceImageId '" + src + "' is missing, synthetic, legacy, or points to the bundled demo image.";
        }
        String grp = m.captureGroup() != null ? m.captureGroup().trim() : "";
        if (grp.isEmpty() || "UNKNOWN".equalsIgnoreCase(grp)
                || "DEFAULT_GROUP".equalsIgnoreCase(grp) || "LEGACY_UNGROUPED".equalsIgnoreCase(grp)) {
            return "captureGroup is missing, placeholder, or ungrouped legacy.";
        }
        if (!m.isHumanReviewed() || m.reviewerId().isBlank()) {
            return "reviewStatus '" + m.reviewStatus() + "' lacks explicit human reviewer attribution (by=<reviewerId>).";
        }
        String expectedSha = m.sourceSha256();
        if (expectedSha.length() != 64) {
            return "missing or invalid 64-character SHA-256 content hash in reviewStatus.";
        }
        Path imgPath = Path.of(src);
        if (!Files.exists(imgPath) || Files.isDirectory(imgPath)) {
            return "source image file does not exist on disk: " + src;
        }
        String actualSha = DatasetManifest.computeFileSha256OrEmpty(imgPath);
        if (!expectedSha.equalsIgnoreCase(actualSha)) {
            return "source image SHA-256 hash mismatch (expected " + expectedSha.substring(0, 12)
                + "..., got " + (actualSha.length() >= 12 ? actualSha.substring(0, 12) : actualSha) + "...).";
        }
        try {
            BufferedImage decoded = ImageIO.read(imgPath.toFile());
            if (decoded == null || decoded.getWidth() < 10 || decoded.getHeight() < 10) {
                return "source image cannot be decoded or is smaller than 10x10 px: " + src;
            }
            if (m.x() < 0 || m.y() < 0 || m.width() <= 0 || m.height() <= 0
                    || m.x() + m.width() > decoded.getWidth()
                    || m.y() + m.height() > decoded.getHeight()) {
                return String.format(Locale.ROOT,
                    "ROI (%d,%d,%dx%d) extends outside decoded source image dimensions (%dx%d).",
                    m.x(), m.y(), m.width(), m.height(), decoded.getWidth(), decoded.getHeight());
            }
        } catch (Exception e) {
            return "source image decode error: " + e.getMessage();
        }
        return null;
    }
}
