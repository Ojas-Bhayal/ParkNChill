package com.parknchillai.ai;

import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Manages the companion 22-column provenance and partition manifest ({@code data/labels/manifest.csv})
 * in strict row-for-row alignment with the 9-column {@code data/labels/features.csv}.
 *
 * <p>Safety invariants enforced in v2.2 (PNC-FIX-01, PNC-FIX-02, PNC-REM-01..04, PNC-REM-06):
 * <ul>
 *   <li><strong>Fail-Closed Missing-Manifest Policy ({@code PNC-REM-01})</strong>:
 *       {@link #loadOrSync(String, String)} never fabricates a default manifest when a non-empty
 *       {@code features.csv} has no companion manifest on disk. New datasets must be created via
 *       {@link #initializeNewEmptyDataset}, bundled demo data via {@link #initializeBundledDemoDataset},
 *       and legacy unmapped CSVs via {@link #importLegacyDataset}.</li>
 *   <li><strong>Schema-Aware Sample Identity ({@code PNC-REM-03})</strong>:
 *       {@link #computeSampleId} validates {@code schemaVersion} and incorporates both a readable
 *       schema token and a 16-hex SHA-256 canonical tuple digest so changing {@code schemaVersion}
 *       produces a distinct sample ID. Collisions across distinct canonical tuples are rejected.</li>
 *   <li><strong>Explicit Reviewer Attribution & Real-Camera Evidence ({@code PNC-REM-04})</strong>:
 *       Removes implicit {@code OPERATOR} and {@code QA_REVIEWER} fallbacks. Human-reviewed and
 *       real-camera records require a non-blank reviewer ID, existing decodable image ({@code >= 10x10}),
 *       valid in-bounds ROI geometry, and matching 64-hex SHA-256 content hash.</li>
 *   <li><strong>State-Tracked Three-Target Transaction ({@code PNC-REM-02})</strong>:
 *       Commits {@code features.csv}, {@code manifest.csv}, and the optional ROI crop image as a
 *       single state-tracked transaction ({@code PREPARED -> COMMITTING -> COMMITTED}), retaining
 *       backups until all replacements and post-commit verifications succeed.</li>
 *   <li><strong>Protected Validation & Test Capture Groups ({@code PNC-REM-06})</strong>:
 *       Rejects adding or modifying samples belonging to {@code VALIDATION} or {@code TEST}
 *       capture groups during training updates.</li>
 * </ul>
 */
public final class DatasetManifest {

    public static final String DEFAULT_MANIFEST_PATH = "data/labels/manifest.csv";
    public static final String DEFAULT_CROPS_DIR     = "data/labels/crops";
    public static final int    EXPECTED_COLUMN_COUNT = 22;

    public static final String HEADER =
        "sampleId,sourceImageId,captureGroup,slotId,x,y,width,height," +
        "meanIntensity,foregroundRatio,darkPixelRatio,edgeDensity,objectArea,meanSaturation,saturationRatio,largestBlobRatio," +
        "label,schemaVersion,reviewStatus,provenance,partition,timestamp";

    public static final Set<String> SUPPORTED_SCHEMAS = Set.of(
        com.parknchillai.imageprocessing.ProcessingPolicy.CANONICAL_SCHEMA_ID, FeatureVector.RAW_SCHEMA_VERSION, FeatureVector.DETECTOR_FAILED_SCHEMA_VERSION, "v2.3-10feat", "2.3-10F", "v2.2-8feat", "2.2-8F", "v2.0-5feat"
    );
    public static final Set<String> VALID_PARTITIONS = Set.of("TRAIN", "VALIDATION", "TEST");

    public enum TransactionState {
        PREPARED,
        COMMITTING,
        COMMITTED
    }

    public enum FaultInjectionStage {
        NONE,
        BEFORE_REPLACEMENT,
        AFTER_FEATURES_REPLACEMENT,
        AFTER_MANIFEST_REPLACEMENT,
        DURING_CROP_REPLACEMENT,
        BEFORE_COMMIT_FINALIZE
    }

    public record FileMetadataKey(String normalizedPath, long fileSize, long lastModifiedMillis) {}
    private static final Map<FileMetadataKey, String> FILE_HASH_CACHE = new ConcurrentHashMap<>();
    private static final AtomicInteger HASH_COMPUTATION_COUNT = new AtomicInteger(0);

    public static int getHashComputationCount() {
        return HASH_COMPUTATION_COUNT.get();
    }

    public static void resetHashComputationCountForTesting() {
        HASH_COMPUTATION_COUNT.set(0);
        FILE_HASH_CACHE.clear();
    }

    private static volatile FaultInjectionStage faultInjectionStage = FaultInjectionStage.NONE;
    private static volatile boolean simulateRollbackFailureForTesting = false;
    private static volatile String  forcedCollisionSampleIdForTesting = null;

    private DatasetManifest() { /* utility class */ }

    public static void setFaultInjectionStageForTesting(FaultInjectionStage stage) {
        faultInjectionStage = (stage == null) ? FaultInjectionStage.NONE : stage;
    }

    public static void setSimulateWriteFailureForTesting(boolean simulate) {
        faultInjectionStage = simulate ? FaultInjectionStage.AFTER_FEATURES_REPLACEMENT : FaultInjectionStage.NONE;
    }

    public static void setSimulateCropFailureForTesting(boolean simulate) {
        faultInjectionStage = simulate ? FaultInjectionStage.DURING_CROP_REPLACEMENT : FaultInjectionStage.NONE;
    }

    public static void setSimulateRollbackFailureForTesting(boolean simulate) {
        simulateRollbackFailureForTesting = simulate;
    }

    public static void setForcedCollisionSampleIdForTesting(String collisionId) {
        forcedCollisionSampleIdForTesting = collisionId;
    }

    public static void resetTestingHooks() {
        faultInjectionStage = FaultInjectionStage.NONE;
        simulateRollbackFailureForTesting = false;
        forcedCollisionSampleIdForTesting = null;
    }

    public record MigrationReport(
            Path featuresBackupPath,
            Path manifestBackupPath,
            int totalRecords,
            int preservedRecords,
            int inferredRecords,
            List<String> unrecoveredSampleIds,
            List<String> diagnostics
    ) {}

    public record TestCorrectionResult(
            ManifestRecord updatedRecord,
            String newSplitId,
            String warningMessage
    ) {}

    public record ManifestRecord(
            String sampleId,
            String sourceImageId,
            String captureGroup,
            int slotId,
            int x,
            int y,
            int width,
            int height,
            FeatureVector features,
            SlotStatus label,
            String schemaVersion,
            String reviewStatus,
            String provenance,
            String partition,
            String timestamp
    ) {
        public String toCsvLine() {
            return String.format(Locale.ROOT,
                "%s,%s,%s,%d,%d,%d,%d,%d,%s,%s,%s,%s,%s,%s,%s",
                sanitize(sampleId),
                sanitize(sourceImageId),
                sanitize(captureGroup),
                slotId, x, y, width, height,
                features.toCsvRow(FeatureVector.FEATURE_COUNT),
                label.name(),
                sanitize(schemaVersion),
                sanitize(reviewStatus),
                sanitize(provenance),
                sanitize(partition),
                sanitize(timestamp)
            );
        }

        /**
         * Extracts the embedded 64-character SHA-256 content hash of the source image when present
         * in {@code reviewStatus} (e.g. {@code HUMAN_VERIFIED[sha256=<hex>;by=<reviewer>]}).
         */
        public String sourceSha256() {
            if (reviewStatus == null) return "";
            int idx = reviewStatus.indexOf("sha256=");
            if (idx < 0) return "";
            int start = idx + 7;
            int end = reviewStatus.length();
            for (int i = start; i < reviewStatus.length(); i++) {
                char c = reviewStatus.charAt(i);
                if (c == ';' || c == ']' || c == ')' || c == ' ') {
                    end = i;
                    break;
                }
            }
            return reviewStatus.substring(start, end).trim().toLowerCase(Locale.ROOT);
        }

        /**
         * Extracts the explicit reviewer identifier when present in {@code reviewStatus}
         * via {@code by=<reviewerId>}. Never substitutes a default fallback reviewer (PNC-REM-04).
         */
        public String reviewerId() {
            if (reviewStatus == null) return "";
            int idx = reviewStatus.indexOf("by=");
            if (idx < 0) return "";
            int start = idx + 3;
            int end = reviewStatus.length();
            for (int i = start; i < reviewStatus.length(); i++) {
                char c = reviewStatus.charAt(i);
                if (c == ';' || c == ']' || c == ')' || c == ' ') {
                    end = i;
                    break;
                }
            }
            String rev = reviewStatus.substring(start, end).trim();
            if (rev.isEmpty() || "UNKNOWN".equalsIgnoreCase(rev)) {
                return "";
            }
            return rev;
        }

        /**
         * Returns true only when this record carries an explicit human review status prefix
         * AND a non-blank explicit reviewer identifier (PNC-REM-04).
         */
        public boolean isHumanReviewed() {
            if (reviewStatus == null) return false;
            String upper = reviewStatus.toUpperCase(Locale.ROOT);
            return (upper.startsWith("USER_REVIEWED") || upper.startsWith("HUMAN_VERIFIED"))
                && !reviewerId().isBlank();
        }

        /**
         * Registers a user-declared real-camera record with status {@code USER_DECLARED_REAL_CAMERA}
         * prior to full verification promotion (PNC-REM-04).
         */
        public static ManifestRecord createUserDeclaredRealCameraRecord(
                String sampleId,
                String sourceImagePath,
                String captureGroup,
                int slotId,
                int x, int y, int width, int height,
                FeatureVector features,
                SlotStatus label,
                String reviewerId,
                String partition,
                String timestamp) {
            validateCommonRealCameraInputs(
                sampleId, sourceImagePath, captureGroup, slotId, x, y, width, height,
                features, label, FeatureVector.SCHEMA_VERSION, reviewerId, partition, timestamp
            );
            String sha = computeFileSha256OrEmpty(Path.of(sourceImagePath.trim()));
            String revStatus = String.format(Locale.ROOT,
                "USER_REVIEWED[sha256=%s;by=%s]", sha, reviewerId.trim());
            return new ManifestRecord(
                sampleId.trim(), sourceImagePath.trim(), captureGroup.trim(),
                slotId, x, y, width, height,
                features, label, FeatureVector.SCHEMA_VERSION, revStatus,
                "USER_DECLARED_REAL_CAMERA", partition.trim().toUpperCase(Locale.ROOT), timestamp.trim()
            );
        }

        /**
         * Creates a {@code VERIFIED_REAL_CAMERA} manifest record only after validating that:
         * <ul>
         *   <li>{@code reviewerId} is explicitly supplied and non-blank (never substituted).</li>
         *   <li>{@code sourceImagePath} exists on disk, decodes via {@link ImageIO#read} ({@code >= 10x10}),
         *       and produces a valid 64-character SHA-256 content digest.</li>
         *   <li>ROI coordinates {@code (x, y, width, height)} lie strictly within the decoded image dimensions.</li>
         *   <li>{@code captureGroup}, {@code features}, {@code label}, and {@code schemaVersion} are valid.</li>
         * </ul>
         */
        public static ManifestRecord createVerifiedRealCameraRecord(
                String sampleId,
                String sourceImagePath,
                String captureGroup,
                int slotId,
                int x, int y, int width, int height,
                FeatureVector features,
                SlotStatus label,
                String reviewerId,
                String partition,
                String timestamp) {
            ManifestRecord declared = createUserDeclaredRealCameraRecord(
                sampleId, sourceImagePath, captureGroup, slotId, x, y, width, height,
                features, label, reviewerId, partition, timestamp
            );
            return promoteToVerifiedRealCamera(declared);
        }

        private static void validateCommonRealCameraInputs(
                String sampleId,
                String sourceImagePath,
                String captureGroup,
                int slotId,
                int x, int y, int width, int height,
                FeatureVector features,
                SlotStatus label,
                String schemaVersion,
                String reviewerId,
                String partition,
                String timestamp) {
            if (sampleId == null || sampleId.isBlank() || "UNKNOWN".equalsIgnoreCase(sampleId.trim())) {
                throw new IllegalArgumentException("sampleId must not be null, blank, or UNKNOWN.");
            }
            if (reviewerId == null || reviewerId.isBlank() || "UNKNOWN".equalsIgnoreCase(reviewerId.trim())) {
                throw new IllegalArgumentException("Explicit non-blank reviewerId is required (no implicit fallback allowed).");
            }
            if (sourceImagePath == null || sourceImagePath.isBlank()
                    || sourceImagePath.trim().startsWith("procedural://")
                    || sourceImagePath.trim().startsWith("legacy://")
                    || sourceImagePath.trim().startsWith("unverified://")
                    || sourceImagePath.trim().endsWith("sample_parking_lot.png")) {
                throw new IllegalArgumentException("sourceImagePath must point to a real image file on disk: " + sourceImagePath);
            }
            if (captureGroup == null || captureGroup.isBlank()
                    || "UNKNOWN".equalsIgnoreCase(captureGroup.trim())
                    || "DEFAULT_GROUP".equalsIgnoreCase(captureGroup.trim())
                    || "LEGACY_UNGROUPED".equalsIgnoreCase(captureGroup.trim())) {
                throw new IllegalArgumentException("captureGroup must be a valid non-placeholder identifier.");
            }
            if (slotId <= 0 || x < 0 || y < 0 || width <= 0 || height <= 0) {
                throw new IllegalArgumentException("Invalid slotId or ROI geometry.");
            }
            if (features == null || !features.hasAllValuesWithinValidBounds()) {
                throw new IllegalArgumentException("features must be non-null and within valid mathematical bounds.");
            }
            if (label == null || label == SlotStatus.NEEDS_REVIEW) {
                throw new IllegalArgumentException("label must be OCCUPIED or AVAILABLE.");
            }
            if (schemaVersion == null || !SUPPORTED_SCHEMAS.contains(schemaVersion.trim())) {
                throw new IllegalArgumentException("Unsupported schemaVersion: " + schemaVersion);
            }
            if (partition == null || !VALID_PARTITIONS.contains(partition.trim().toUpperCase(Locale.ROOT))) {
                throw new IllegalArgumentException("Invalid partition: " + partition);
            }
            if (timestamp == null || timestamp.isBlank()) {
                throw new IllegalArgumentException("timestamp must not be null or blank.");
            }
        }

        private static String sanitize(String s) {
            if (s == null || s.isBlank()) return "UNKNOWN";
            return s.replace(",", "_").replace("\n", "").replace("\r", "").trim();
        }
    }

    /**
     * Promotes a {@code USER_DECLARED_REAL_CAMERA} record to {@code VERIFIED_REAL_CAMERA} only after
     * all file existence, image decoding, ROI bounds, SHA-256 content hash, and human reviewer
     * evidence checks succeed (PNC-REM-04).
     */
    public static ManifestRecord promoteToVerifiedRealCamera(ManifestRecord candidate) {
        if (candidate == null) {
            throw new IllegalArgumentException("Candidate ManifestRecord must not be null.");
        }
        String rejectReason = DatasetValidator.verifyRealCameraEvidence(candidate);
        if (rejectReason != null) {
            throw new IllegalArgumentException(
                "Cannot promote sample '" + candidate.sampleId() + "' to VERIFIED_REAL_CAMERA: " + rejectReason);
        }
        String revStatus = String.format(Locale.ROOT,
            "HUMAN_VERIFIED[sha256=%s;by=%s]", candidate.sourceSha256(), candidate.reviewerId());
        return new ManifestRecord(
            candidate.sampleId(), candidate.sourceImageId(), candidate.captureGroup(),
            candidate.slotId(), candidate.x(), candidate.y(), candidate.width(), candidate.height(),
            candidate.features(), candidate.label(), candidate.schemaVersion(),
            revStatus, "VERIFIED_REAL_CAMERA", candidate.partition(), candidate.timestamp()
        );
    }

    /**
     * Explicitly initializes a brand-new empty dataset pair (header-only 9-column {@code features.csv}
     * and header-only 22-column {@code manifest.csv}) when neither file exists (PNC-REM-01).
     */
    public static void initializeNewEmptyDataset(String featuresCsvPath, String manifestPath) throws IOException {
        Path fPath = Path.of(featuresCsvPath);
        Path mPath = Path.of(manifestPath);
        if (Files.exists(fPath) && Files.size(fPath) > 0) {
            throw new IllegalStateException("Features file already exists and is non-empty: " + featuresCsvPath);
        }
        if (Files.exists(mPath) && Files.size(mPath) > 0) {
            throw new IllegalStateException("Manifest file already exists and is non-empty: " + manifestPath);
        }
        if (fPath.getParent() != null) Files.createDirectories(fPath.getParent());
        if (mPath.getParent() != null) Files.createDirectories(mPath.getParent());

        try (BufferedWriter fw = Files.newBufferedWriter(fPath, StandardCharsets.UTF_8)) {
            fw.write("# ParkNChill v2.2 — Initialized Empty 9-Column Feature Dataset\n");
            fw.write(FeatureVector.csvHeader());
            fw.newLine();
        }
        try (BufferedWriter mw = Files.newBufferedWriter(mPath, StandardCharsets.UTF_8)) {
            mw.write("# ParkNChill v2.2 — Initialized Empty 22-Column Provenance Manifest\n");
            mw.write(HEADER);
            mw.newLine();
        }
    }

    /**
     * Explicit named initializer for the bundled 48-row demonstration/synthetic dataset (PNC-REM-01).
     * Never invoked implicitly by {@link #loadOrSync(String, String)}, and never marks any row as
     * {@code VERIFIED_REAL_CAMERA}.
     */
    public static List<ManifestRecord> initializeBundledDemoDataset(String featuresCsvPath, String manifestPath) throws IOException {
        DatasetLoader.Dataset ds = DatasetLoader.load(featuresCsvPath);
        List<ManifestRecord> generated = buildBundledDemoManifestForDataset(ds);
        verifyStrictAlignment(ds, generated);
        saveManifest(manifestPath, generated);
        return Collections.unmodifiableList(generated);
    }

    /**
     * Explicit legacy dataset import operation (PNC-REM-01).
     * Creates a timestamped backup of {@code featuresCsvPath}, imports legacy rows with
     * {@code reviewStatus = "UNVERIFIED_LEGACY"}, {@code provenance = "LEGACY_IMPORTED"},
     * and {@code captureGroup = "LEGACY_UNGROUPED"} (so group-disjoint evaluation is disabled
     * until trustworthy capture-group metadata is supplied), and returns a {@link MigrationReport}.
     */
    public static MigrationReport importLegacyDataset(String featuresCsvPath, String manifestPath) throws IOException {
        Path fPath = Path.of(featuresCsvPath);
        Path mPath = Path.of(manifestPath);
        if (!Files.exists(fPath)) {
            throw new IOException("Legacy features CSV does not exist: " + featuresCsvPath);
        }
        String tsSuffix = String.valueOf(System.currentTimeMillis());
        Path fBackup = Path.of(featuresCsvPath + ".bak." + tsSuffix);
        Files.copy(fPath, fBackup, StandardCopyOption.REPLACE_EXISTING);

        Path mBackup = null;
        if (Files.exists(mPath)) {
            mBackup = Path.of(manifestPath + ".bak." + tsSuffix);
            Files.copy(mPath, mBackup, StandardCopyOption.REPLACE_EXISTING);
        }

        DatasetLoader.Dataset ds = DatasetLoader.load(featuresCsvPath);
        List<ManifestRecord> imported = new ArrayList<>(ds.size());
        List<String> unrecoveredIds = new ArrayList<>(ds.size());
        List<String> diagnostics = new ArrayList<>();

        String schemaVer = (ds.featureCount() == FeatureVector.LEGACY_FEATURE_COUNT)
            ? "v2.0-5feat"
            : FeatureVector.SCHEMA_VERSION;
        String nowIso = Instant.now().toString();

        for (int i = 0; i < ds.size(); i++) {
            String id = String.format(Locale.ROOT, "LEG-%s-%03d",
                schemaVer.replaceAll("[^a-zA-Z0-9]", ""), i + 1);
            unrecoveredIds.add(id);
            imported.add(new ManifestRecord(
                id,
                "legacy://unverified_row_" + (i + 1),
                "LEGACY_UNGROUPED",
                (i % 12) + 1, 0, 0, 80, 100,
                new FeatureVector(ds.features[i]),
                ds.labels[i],
                schemaVer,
                "UNVERIFIED_LEGACY",
                "LEGACY_IMPORTED",
                "TRAIN",
                nowIso
            ));
        }

        verifyStrictAlignment(ds, imported);
        saveManifest(manifestPath, imported);
        diagnostics.add(String.format(Locale.ROOT,
            "Imported %d legacy rows as UNVERIFIED_LEGACY / LEGACY_IMPORTED with captureGroup=LEGACY_UNGROUPED. " +
            "Group-disjoint evaluation is disabled until verified capture groups are supplied.",
            ds.size()));

        return new MigrationReport(
            fBackup, mBackup, ds.size(), 0, ds.size(),
            Collections.unmodifiableList(unrecoveredIds),
            Collections.unmodifiableList(diagnostics)
        );
    }

    /**
     * Strictly loads and validates an existing dataset and its companion manifest (PNC-REM-01).
     *
     * <p>This method is strictly read/validate-only: if {@code manifestPath} does not exist for
     * a non-empty {@code featuresCsvPath}, it fails closed with an {@link IllegalStateException}
     * without fabricating a manifest or modifying {@code featuresCsvPath}.
     */
    public static List<ManifestRecord> loadOrSync(String featuresCsvPath, String manifestPath) {
        if (featuresCsvPath == null || featuresCsvPath.isBlank()) {
            throw new IllegalArgumentException("featuresCsvPath must not be null or blank.");
        }
        if (manifestPath == null || manifestPath.isBlank()) {
            throw new IllegalArgumentException("manifestPath must not be null or blank.");
        }

        recoverInterruptedTransactionIfNeeded(featuresCsvPath, manifestPath);

        DatasetLoader.Dataset ds = DatasetLoader.load(featuresCsvPath);
        Path mPath = Path.of(manifestPath);
        if (!Files.exists(mPath)) {
            throw new IllegalStateException(
                "Missing provenance manifest '" + manifestPath + "' for non-empty dataset '" + featuresCsvPath + "' ("
                + ds.size() + " rows). Refusing to invent provenance or partitions; call "
                + "DatasetManifest.importLegacyDataset() or DatasetManifest.initializeBundledDemoDataset() explicitly.");
        }

        List<ManifestRecord> loaded;
        try {
            loaded = loadManifest(manifestPath);
        } catch (IOException e) {
            throw new IllegalStateException(
                "Failed to read existing provenance manifest '" + manifestPath + "': " + e.getMessage(), e);
        }
        verifyStrictAlignment(ds, loaded);
        return loaded;
    }

    /**
     * Strictly validates that {@code ds} and {@code manifest} have identical row counts,
     * matching 8-element feature vectors, matching labels, unique sample IDs, supported
     * schema versions, and explicit partition assignments.
     */
    public static void verifyStrictAlignment(DatasetLoader.Dataset ds, List<ManifestRecord> manifest) {
        if (ds == null) {
            throw new IllegalArgumentException("Dataset must not be null.");
        }
        if (manifest == null) {
            throw new IllegalArgumentException("Manifest list must not be null.");
        }
        if (ds.size() != manifest.size()) {
            throw new IllegalStateException(String.format(Locale.ROOT,
                "Manifest/feature alignment failure: features CSV has %d rows, but manifest has %d records. " +
                "Refusing to overwrite manifest; run DatasetManifest.repairOrMigrate() to back up and reconcile.",
                ds.size(), manifest.size()));
        }

        Set<String> seenIds = new HashSet<>();
        for (int i = 0; i < ds.size(); i++) {
            ManifestRecord rec = manifest.get(i);
            if (rec == null) {
                throw new IllegalStateException("Manifest record at index " + i + " is null.");
            }
            if (rec.sampleId() == null || rec.sampleId().isBlank() || "UNKNOWN".equalsIgnoreCase(rec.sampleId())) {
                throw new IllegalStateException("Manifest record at index " + i + " has blank or UNKNOWN sampleId.");
            }
            if (!seenIds.add(rec.sampleId())) {
                throw new IllegalStateException("Duplicate sampleId '" + rec.sampleId() + "' at manifest index " + i + ".");
            }
            if (rec.schemaVersion() == null || !SUPPORTED_SCHEMAS.contains(rec.schemaVersion())) {
                throw new IllegalStateException(
                    "Unsupported manifest schemaVersion '" + rec.schemaVersion() + "' at row " + i + " (" + rec.sampleId() + ").");
            }
            if (rec.partition() == null || !VALID_PARTITIONS.contains(rec.partition().toUpperCase(Locale.ROOT))) {
                throw new IllegalStateException(
                    "Invalid partition '" + rec.partition() + "' at row " + i + " (" + rec.sampleId() + "); expected TRAIN, VALIDATION, or TEST.");
            }
            if (rec.label() == null || rec.label() == SlotStatus.NEEDS_REVIEW) {
                throw new IllegalStateException("Invalid manifest ground-truth label at row " + i + ": " + rec.label());
            }
            if (ds.labels[i] != rec.label()) {
                throw new IllegalStateException(String.format(Locale.ROOT,
                    "Label mismatch at row %d (%s): features CSV has %s, but manifest has %s.",
                    i, rec.sampleId(), ds.labels[i], rec.label()));
            }
            if (rec.features() == null || !rec.features().hasAllValuesWithinValidBounds()) {
                throw new IllegalStateException(
                    "Manifest record at row " + i + " (" + rec.sampleId() + ") contains out-of-bounds feature values.");
            }
            int cmpCount = Math.min(ds.featureCount(), FeatureVector.FEATURE_COUNT);
            for (int f = 0; f < cmpCount; f++) {
                double fCsv = ds.features[i][f];
                double fMan = rec.features().get(f);
                if (!Double.isFinite(fCsv) || !Double.isFinite(fMan) || Math.abs(fCsv - fMan) > 1e-4) {
                    throw new IllegalStateException(String.format(Locale.ROOT,
                        "Feature mismatch at row %d (%s), feature index %d (%s): features CSV has %.6f, but manifest has %.6f.",
                        i, rec.sampleId(), f, FeatureVector.SPECS.get(f).key(), fCsv, fMan));
                }
            }
        }
    }

    /**
     * Parses a 22-column manifest CSV file, failing closed on malformed rows, invalid column counts,
     * duplicate sample IDs, unsupported schema versions, or invalid partitions.
     */
    public static List<ManifestRecord> loadManifest(String manifestPath) throws IOException {
        Path path = Path.of(manifestPath);
        if (!Files.exists(path)) {
            throw new IOException("Manifest file does not exist: " + manifestPath);
        }

        List<ManifestRecord> records = new ArrayList<>();
        Set<String> seenIds = new HashSet<>();
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            int lineNum = 0;
            boolean headerChecked = false;
            while ((line = reader.readLine()) != null) {
                lineNum++;
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                if (!headerChecked) {
                    headerChecked = true;
                    String[] hCols = line.split(",", -1);
                    if (!line.toLowerCase(Locale.ROOT).startsWith("sampleid,") || hCols.length != EXPECTED_COLUMN_COUNT) {
                        throw new IllegalArgumentException(
                            "Invalid manifest header at line " + lineNum + ": expected " + EXPECTED_COLUMN_COUNT
                            + " columns matching DatasetManifest.HEADER, got " + hCols.length + " columns.");
                    }
                    continue;
                }
                String[] p = line.split(",", -1);
                if (p.length != EXPECTED_COLUMN_COUNT) {
                    throw new IllegalArgumentException(
                        "Malformed manifest row at line " + lineNum + ": expected " + EXPECTED_COLUMN_COUNT
                        + " columns, got " + p.length + ".");
                }
                String sampleId      = p[0].trim();
                String sourceImageId = p[1].trim();
                String captureGroup  = p[2].trim();
                if (sampleId.isEmpty() || "UNKNOWN".equalsIgnoreCase(sampleId)) {
                    throw new IllegalArgumentException("Manifest line " + lineNum + ": sampleId must not be blank or UNKNOWN.");
                }
                if (!seenIds.add(sampleId)) {
                    throw new IllegalArgumentException("Manifest line " + lineNum + ": duplicate sampleId '" + sampleId + "'.");
                }
                int slotId, x, y, width, height;
                double[] fv = new double[8];
                try {
                    slotId = Integer.parseInt(p[3].trim());
                    x      = Integer.parseInt(p[4].trim());
                    y      = Integer.parseInt(p[5].trim());
                    width  = Integer.parseInt(p[6].trim());
                    height = Integer.parseInt(p[7].trim());
                    for (int i = 0; i < 8; i++) {
                        fv[i] = Double.parseDouble(p[8 + i].trim());
                    }
                } catch (NumberFormatException nfe) {
                    throw new IllegalArgumentException("Manifest line " + lineNum + ": invalid numeric value in '" + line + "'.", nfe);
                }
                SlotStatus label     = SlotStatus.fromString(p[16].trim());
                if (label == SlotStatus.NEEDS_REVIEW) {
                    throw new IllegalArgumentException("Manifest line " + lineNum + ": NEEDS_REVIEW is not a valid training label.");
                }
                String schemaVersion = p[17].trim();
                if (!SUPPORTED_SCHEMAS.contains(schemaVersion)) {
                    throw new IllegalArgumentException("Manifest line " + lineNum + ": unsupported schemaVersion '" + schemaVersion + "'.");
                }
                String reviewStatus  = p[18].trim();
                String provenance    = p[19].trim();
                String partition     = p[20].trim().toUpperCase(Locale.ROOT);
                if (!VALID_PARTITIONS.contains(partition)) {
                    throw new IllegalArgumentException("Manifest line " + lineNum + ": invalid partition '" + partition + "'.");
                }
                String timestamp     = p[21].trim();
                FeatureVector vec = new FeatureVector(fv);
                if (!vec.hasAllValuesWithinValidBounds()) {
                    throw new IllegalArgumentException("Manifest line " + lineNum + ": feature values out of valid mathematical bounds.");
                }
                records.add(new ManifestRecord(
                    sampleId, sourceImageId, captureGroup, slotId, x, y, width, height,
                    vec, label, schemaVersion, reviewStatus, provenance, partition, timestamp
                ));
            }
        }
        if (records.isEmpty()) {
            throw new IllegalArgumentException("Manifest file contains no data records: " + manifestPath);
        }
        return Collections.unmodifiableList(records);
    }

    public static void saveManifest(String manifestPath, List<ManifestRecord> records) throws IOException {
        saveManifest(manifestPath, records, null);
    }

    public static void saveManifest(String manifestPath, List<ManifestRecord> records, String splitMetadataComment) throws IOException {
        Path path = Path.of(manifestPath);
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("# ParkNChill v2.2 — Dataset Provenance, ROI Geometry & Partition Manifest (22 Columns)\n");
            if (splitMetadataComment != null && !splitMetadataComment.isBlank()) {
                writer.write("# " + splitMetadataComment.trim() + "\n");
            }
            writer.write(HEADER);
            writer.newLine();
            for (ManifestRecord r : records) {
                writer.write(r.toCsvLine());
                writer.newLine();
            }
        }
    }

    /**
     * Explicit, non-silent repair and migration operation (PNC-FIX-01, PNC-REM-01).
     * Creates timestamped backups of both files, preserves any verified manifest records whose
     * features and labels match, marks reconstructed records explicitly as {@code UNVERIFIED_INFERRED}
     * / {@code INFERRED_MIGRATION} with {@code LEGACY_UNGROUPED}, and returns a {@link MigrationReport}.
     */
    public static MigrationReport repairOrMigrate(String featuresCsvPath, String manifestPath) throws IOException {
        Path fPath = Path.of(featuresCsvPath);
        Path mPath = Path.of(manifestPath);
        if (!Files.exists(fPath)) {
            throw new IOException("Features CSV not found for migration: " + featuresCsvPath);
        }

        String tsSuffix = String.valueOf(System.currentTimeMillis());
        Path fBackup = Path.of(featuresCsvPath + ".bak." + tsSuffix);
        Files.copy(fPath, fBackup, StandardCopyOption.REPLACE_EXISTING);

        Path mBackup = null;
        List<ManifestRecord> salvagable = new ArrayList<>();
        List<String> diagnostics = new ArrayList<>();
        if (Files.exists(mPath)) {
            mBackup = Path.of(manifestPath + ".bak." + tsSuffix);
            Files.copy(mPath, mBackup, StandardCopyOption.REPLACE_EXISTING);
            try {
                salvagable.addAll(loadManifest(manifestPath));
            } catch (Exception ex) {
                diagnostics.add("Existing manifest had parse error (" + ex.getMessage() + "); salvaged valid lines where possible.");
                salvagable.addAll(salvageReadableLines(mPath, diagnostics));
            }
        }

        DatasetLoader.Dataset ds = DatasetLoader.load(featuresCsvPath);
        List<ManifestRecord> rebuilt = new ArrayList<>(ds.size());
        Set<Integer> usedSalvageIdx = new HashSet<>();
        Set<String> usedSampleIds = new HashSet<>();
        List<String> unrecoveredIds = new ArrayList<>();
        int preservedCount = 0;
        int inferredCount = 0;

        for (int i = 0; i < ds.size(); i++) {
            FeatureVector fv = new FeatureVector(ds.features[i]);
            SlotStatus lbl = ds.labels[i];
            ManifestRecord matched = null;
            for (int j = 0; j < salvagable.size(); j++) {
                if (usedSalvageIdx.contains(j)) continue;
                ManifestRecord cand = salvagable.get(j);
                if (cand.label() == lbl && featuresMatch(fv, cand.features()) && !usedSampleIds.contains(cand.sampleId())) {
                    matched = cand;
                    usedSalvageIdx.add(j);
                    usedSampleIds.add(cand.sampleId());
                    break;
                }
            }
            if (matched != null) {
                rebuilt.add(matched);
                preservedCount++;
            } else {
                String inferredId = String.format(Locale.ROOT, "MIG-%03d", i + 1);
                while (usedSampleIds.contains(inferredId)) {
                    inferredId = inferredId + "_X";
                }
                usedSampleIds.add(inferredId);
                unrecoveredIds.add(inferredId);
                inferredCount++;
                rebuilt.add(new ManifestRecord(
                    inferredId,
                    "unverified://migrated_row_" + (i + 1),
                    "LEGACY_UNGROUPED",
                    (i % 12) + 1, 0, 0, 80, 100,
                    fv, lbl, FeatureVector.SCHEMA_VERSION,
                    "UNVERIFIED_INFERRED",
                    "INFERRED_MIGRATION",
                    "TRAIN",
                    Instant.now().toString()
                ));
            }
        }

        verifyStrictAlignment(ds, rebuilt);
        saveManifest(manifestPath, rebuilt);
        diagnostics.add(String.format(Locale.ROOT,
            "Reconciled %d rows (%d preserved, %d inferred with UNVERIFIED_INFERRED provenance).",
            ds.size(), preservedCount, inferredCount));

        return new MigrationReport(
            fBackup, mBackup, ds.size(), preservedCount, inferredCount,
            Collections.unmodifiableList(unrecoveredIds),
            Collections.unmodifiableList(diagnostics)
        );
    }

    /**
     * Computes a deterministic, schema-aware sample ID (PNC-REM-03) from the canonical tuple:
     * {@code (sourceImageDigestOrPath, captureGroup, slotId, x, y, width, height, schemaVersion)}.
     *
     * <p>Throws {@link IllegalArgumentException} if {@code schemaVersion} is null, blank, or
     * not in {@link #SUPPORTED_SCHEMAS}.
     */
    public static String computeSampleId(String sourceImageId,
                                         String captureGroup,
                                         int slotId,
                                         int x, int y, int width, int height,
                                         String schemaVersion) {
        return computeSampleId(sourceImageId, null, captureGroup, slotId, x, y, width, height, schemaVersion);
    }

    /**
     * Computes a deterministic, schema-aware sample ID (PNC-REM-03, WS-5) from the canonical tuple:
     * {@code (sourceImageDigestOrPath, captureGroup, slotId, x, y, width, height, schemaVersion)}.
     *
     * <p>When {@code explicitSourceSha256} is provided, it avoids redundant file I/O operations.</p>
     */
    public static String computeSampleId(String sourceImageId,
                                         String explicitSourceSha256,
                                         String captureGroup,
                                         int slotId,
                                         int x, int y, int width, int height,
                                         String schemaVersion) {
        if (schemaVersion == null || schemaVersion.isBlank() || !SUPPORTED_SCHEMAS.contains(schemaVersion.trim())) {
            throw new IllegalArgumentException(
                "Unsupported or blank schemaVersion '" + schemaVersion + "'; supported: " + SUPPORTED_SCHEMAS);
        }
        if (sourceImageId == null || sourceImageId.isBlank()) {
            throw new IllegalArgumentException("sourceImageId must not be null or blank.");
        }
        if (captureGroup == null || captureGroup.isBlank()) {
            throw new IllegalArgumentException("captureGroup must not be null or blank.");
        }
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("ROI width and height must be positive.");
        }
        if (forcedCollisionSampleIdForTesting != null && !forcedCollisionSampleIdForTesting.isBlank()) {
            return forcedCollisionSampleIdForTesting;
        }

        String normSource = sourceImageId.trim().replace('\\', '/');
        String normGroup  = captureGroup.trim();
        String normSchema = schemaVersion.trim();
        String fileSha;
        if (explicitSourceSha256 != null && !explicitSourceSha256.isBlank() && !explicitSourceSha256.startsWith("UNHASHED")) {
            fileSha = explicitSourceSha256.trim();
        } else {
            fileSha = computeFileSha256OrEmpty(Path.of(normSource));
        }
        String sourceIdentity = (!fileSha.isEmpty() && !fileSha.startsWith("UNHASHED"))
            ? ("SHA256:" + fileSha)
            : ("PATH:" + normSource);

        String canonicalTuple = String.format(Locale.ROOT,
            "%s|GRP:%s|SLOT:%d|ROI:%d,%d,%dx%d|SCHEMA:%s",
            sourceIdentity, normGroup, slotId, x, y, width, height, normSchema
        );
        String tupleHash16 = shortSha256Hex(canonicalTuple).substring(0, 16);
        String schemaToken = normSchema.replaceAll("[^a-zA-Z0-9]", "");

        return String.format(Locale.ROOT, "USR-%s-S%02d-%d_%d_%dx%d-%s",
            schemaToken, slotId, x, y, width, height, tupleHash16);
    }

    /**
     * Convenience overload delegating to {@link #recordUserLabeledSample(String, String, String, String, ParkingSlot, FeatureVector, SlotStatus, String, BufferedImage, boolean)}
     * with explicit reviewer identifier {@code "LOCAL_USER"}.
     */
    public static ManifestRecord recordUserLabeledSample(
            String featuresCsvPath,
            String manifestPath,
            String sourceImageId,
            String captureGroup,
            ParkingSlot slot,
            FeatureVector features,
            SlotStatus groundTruth,
            BufferedImage roiCrop,
            boolean saveCropImage) {
        return recordUserLabeledSample(
            featuresCsvPath, manifestPath, sourceImageId, captureGroup,
            slot, features, groundTruth, "LOCAL_USER", roiCrop, saveCropImage
        );
    }

    /**
     * Records a user-labelled or corrected parking-space sample into {@code featuresCsvPath},
     * {@code manifestPath}, and optional {@code roiCrop} as a single state-tracked recoverable
     * transaction (PNC-REM-02, PNC-REM-03, PNC-REM-04, PNC-REM-06).
     */
    public static ManifestRecord recordUserLabeledSample(
            String featuresCsvPath,
            String manifestPath,
            String sourceImageId,
            String captureGroup,
            ParkingSlot slot,
            FeatureVector features,
            SlotStatus groundTruth,
            String reviewerId,
            BufferedImage roiCrop,
            boolean saveCropImage) {
        if (groundTruth == null || groundTruth == SlotStatus.NEEDS_REVIEW) {
            throw new IllegalArgumentException("Ground-truth label must be OCCUPIED or AVAILABLE.");
        }
        if (features == null || !features.hasAllValuesWithinValidBounds()) {
            throw new IllegalArgumentException("FeatureVector must be non-null and within valid mathematical bounds.");
        }
        if (reviewerId == null || reviewerId.isBlank() || "UNKNOWN".equalsIgnoreCase(reviewerId.trim())) {
            throw new IllegalArgumentException("Explicit non-blank reviewerId is required when recording a user-labelled sample.");
        }

        // 1. Strictly load and verify alignment of existing CSV pair before any mutation
        DatasetLoader.Dataset currentDs = DatasetLoader.load(featuresCsvPath);
        List<ManifestRecord> existing = new ArrayList<>(loadOrSync(featuresCsvPath, manifestPath));
        verifyStrictAlignment(currentDs, existing);

        String normSource = (sourceImageId == null || sourceImageId.isBlank())
            ? "interactive_session"
            : sourceImageId.trim().replace('\\', '/');
        String cleanGroup = (captureGroup == null || captureGroup.isBlank())
            ? "GRP_USER_" + new File(normSource).getName().replaceAll("[^a-zA-Z0-9]", "_")
            : captureGroup.trim();
        int slotId = slot != null ? slot.getId() : 1;
        int sx     = slot != null ? slot.getX() : 0;
        int sy     = slot != null ? slot.getY() : 0;
        int sw     = slot != null ? slot.getWidth() : 80;
        int sh     = slot != null ? slot.getHeight() : 100;

        // 2. Collect all protected TEST and VALIDATION capture groups (PNC-REM-06)
        Set<String> protectedTestGroups = new LinkedHashSet<>();
        Set<String> protectedValGroups  = new LinkedHashSet<>();
        for (ManifestRecord r : existing) {
            if ("TEST".equalsIgnoreCase(r.partition())) {
                protectedTestGroups.add(r.captureGroup());
            } else if ("VALIDATION".equalsIgnoreCase(r.partition())) {
                protectedValGroups.add(r.captureGroup());
            }
        }
        if (protectedTestGroups.contains(cleanGroup)) {
            throw new IllegalStateException(
                "Capture group '" + cleanGroup + "' is reserved for the protected TEST partition and cannot be added to training.");
        }
        if (protectedValGroups.contains(cleanGroup)) {
            throw new IllegalStateException(
                "Capture group '" + cleanGroup + "' is reserved for the VALIDATION partition and cannot be added to training; " +
                "use DatasetManifest.correctValidationSampleLabel() to correct an existing validation sample in place.");
        }

        String sampleId = computeSampleId(
            normSource, cleanGroup, slotId, sx, sy, sw, sh, FeatureVector.SCHEMA_VERSION);
        String fileSha = computeFileSha256OrEmpty(Path.of(normSource));
        String cleanReviewer = reviewerId.trim();
        String reviewStatus = !fileSha.isEmpty()
            ? String.format(Locale.ROOT, "USER_REVIEWED[sha256=%s;by=%s]", fileSha, cleanReviewer)
            : String.format(Locale.ROOT, "USER_REVIEWED[by=%s]", cleanReviewer);
        String nowIso = Instant.now().toString();

        ManifestRecord newRecord = new ManifestRecord(
            sampleId,
            normSource,
            cleanGroup,
            slotId, sx, sy, sw, sh,
            features,
            groundTruth,
            FeatureVector.SCHEMA_VERSION,
            reviewStatus,
            "USER_LABELLED",
            "TRAIN",
            nowIso
        );

        // 3. Match strictly by canonical identity and detect ID collisions (PNC-REM-03)
        int matchedIndex = -1;
        for (int i = 0; i < existing.size(); i++) {
            ManifestRecord r = existing.get(i);
            boolean sameLogicalTuple = r.sourceImageId().equals(normSource)
                && r.captureGroup().equals(cleanGroup)
                && r.slotId() == slotId
                && r.x() == sx && r.y() == sy
                && r.width() == sw && r.height() == sh
                && r.schemaVersion().equals(FeatureVector.SCHEMA_VERSION);

            if (r.sampleId().equals(sampleId)) {
                if (!sameLogicalTuple) {
                    throw new IllegalStateException(
                        "Sample ID collision detected for '" + sampleId + "' against distinct existing record ("
                        + r.sourceImageId() + ", group=" + r.captureGroup() + ", slot=" + r.slotId()
                        + ", schema=" + r.schemaVersion() + "); aborting without modifying files.");
                }
                if (!"TRAIN".equalsIgnoreCase(r.partition())) {
                    throw new IllegalStateException(
                        "Sample " + r.sampleId() + " belongs to protected partition " + r.partition() + " and cannot be moved to TRAIN.");
                }
                matchedIndex = i;
                break;
            }
            if (sameLogicalTuple) {
                if (!"TRAIN".equalsIgnoreCase(r.partition())) {
                    throw new IllegalStateException(
                        "Sample " + r.sampleId() + " belongs to protected partition " + r.partition() + " and cannot be moved to TRAIN.");
                }
                // Preserve existing sampleId if updating an older pre-existing record for the same tuple
                newRecord = new ManifestRecord(
                    r.sampleId(), normSource, cleanGroup, slotId, sx, sy, sw, sh,
                    features, groundTruth, FeatureVector.SCHEMA_VERSION,
                    reviewStatus, "USER_LABELLED", "TRAIN", nowIso
                );
                matchedIndex = i;
                break;
            }
        }

        List<double[]> nextFeatures = new ArrayList<>(currentDs.size() + 1);
        List<SlotStatus> nextLabels = new ArrayList<>(currentDs.size() + 1);
        for (int i = 0; i < currentDs.size(); i++) {
            nextFeatures.add(currentDs.features[i].clone());
            nextLabels.add(currentDs.labels[i]);
        }

        if (matchedIndex >= 0) {
            existing.set(matchedIndex, newRecord);
            nextFeatures.set(matchedIndex, features.toArray());
            nextLabels.set(matchedIndex, groundTruth);
        } else {
            existing.add(newRecord);
            nextFeatures.add(features.toArray());
            nextLabels.add(groundTruth);
        }

        DatasetLoader.Dataset nextDataset = new DatasetLoader.Dataset(
            nextFeatures.toArray(new double[0][]),
            nextLabels.toArray(new SlotStatus[0])
        );

        commitDatasetAndManifestTransaction(
            featuresCsvPath, manifestPath, nextDataset, existing,
            newRecord.sampleId(), roiCrop, saveCropImage, null
        );

        return newRecord;
    }

    /**
     * Explicit validation-sample correction flow (PNC-REM-06): updates the label of an existing
     * {@code VALIDATION} sample in place while keeping its partition strictly {@code VALIDATION}
     * (never moving it into {@code TRAIN}).
     */
    public static ManifestRecord correctValidationSampleLabel(
            String featuresCsvPath,
            String manifestPath,
            String sampleId,
            SlotStatus correctedLabel,
            String reviewerId) {
        if (correctedLabel == null || correctedLabel == SlotStatus.NEEDS_REVIEW) {
            throw new IllegalArgumentException("correctedLabel must be OCCUPIED or AVAILABLE.");
        }
        if (reviewerId == null || reviewerId.isBlank()) {
            throw new IllegalArgumentException("Explicit reviewerId is required for validation label correction.");
        }
        DatasetLoader.Dataset currentDs = DatasetLoader.load(featuresCsvPath);
        List<ManifestRecord> existing = new ArrayList<>(loadOrSync(featuresCsvPath, manifestPath));

        int idx = -1;
        for (int i = 0; i < existing.size(); i++) {
            if (existing.get(i).sampleId().equals(sampleId)) {
                idx = i;
                break;
            }
        }
        if (idx < 0) {
            throw new IllegalArgumentException("Validation sampleId not found: " + sampleId);
        }
        ManifestRecord old = existing.get(idx);
        if (!"VALIDATION".equalsIgnoreCase(old.partition())) {
            throw new IllegalStateException("Sample " + sampleId + " is in partition " + old.partition() + ", not VALIDATION.");
        }

        ManifestRecord updated = new ManifestRecord(
            old.sampleId(), old.sourceImageId(), old.captureGroup(),
            old.slotId(), old.x(), old.y(), old.width(), old.height(),
            old.features(), correctedLabel, old.schemaVersion(),
            "USER_REVIEWED[by=" + reviewerId.trim() + "]",
            old.provenance(), "VALIDATION", Instant.now().toString()
        );
        existing.set(idx, updated);

        double[][] nextFeats = new double[currentDs.size()][];
        SlotStatus[] nextLbls = new SlotStatus[currentDs.size()];
        for (int i = 0; i < currentDs.size(); i++) {
            nextFeats[i] = currentDs.features[i].clone();
            nextLbls[i]  = (i == idx) ? correctedLabel : currentDs.labels[i];
        }
        DatasetLoader.Dataset nextDs = new DatasetLoader.Dataset(nextFeats, nextLbls);
        commitDatasetAndManifestTransaction(
            featuresCsvPath, manifestPath, nextDs, existing, sampleId, null, false, null
        );
        return updated;
    }

    /**
     * Explicit versioned test-set correction procedure (PNC-REM-06): updates an existing
     * {@code TEST} sample's ground-truth label in place, increments the persisted split/evaluation
     * version identifier in {@code manifest.csv}, and returns a warning invalidating prior test scores.
     */
    public static TestCorrectionResult correctTestSampleLabelWithVersionBump(
            String featuresCsvPath,
            String manifestPath,
            String sampleId,
            SlotStatus correctedLabel,
            String reviewerId) {
        if (correctedLabel == null || correctedLabel == SlotStatus.NEEDS_REVIEW) {
            throw new IllegalArgumentException("correctedLabel must be OCCUPIED or AVAILABLE.");
        }
        if (reviewerId == null || reviewerId.isBlank()) {
            throw new IllegalArgumentException("Explicit reviewerId is required for test-set label correction.");
        }
        DatasetLoader.Dataset currentDs = DatasetLoader.load(featuresCsvPath);
        List<ManifestRecord> existing = new ArrayList<>(loadOrSync(featuresCsvPath, manifestPath));

        int idx = -1;
        for (int i = 0; i < existing.size(); i++) {
            if (existing.get(i).sampleId().equals(sampleId)) {
                idx = i;
                break;
            }
        }
        if (idx < 0) {
            throw new IllegalArgumentException("Test sampleId not found: " + sampleId);
        }
        ManifestRecord old = existing.get(idx);
        if (!"TEST".equalsIgnoreCase(old.partition())) {
            throw new IllegalStateException("Sample " + sampleId + " is in partition " + old.partition() + ", not TEST.");
        }

        ManifestRecord updated = new ManifestRecord(
            old.sampleId(), old.sourceImageId(), old.captureGroup(),
            old.slotId(), old.x(), old.y(), old.width(), old.height(),
            old.features(), correctedLabel, old.schemaVersion(),
            "USER_REVIEWED[by=" + reviewerId.trim() + "]",
            old.provenance(), "TEST", Instant.now().toString()
        );
        existing.set(idx, updated);

        double[][] nextFeats = new double[currentDs.size()][];
        SlotStatus[] nextLbls = new SlotStatus[currentDs.size()];
        for (int i = 0; i < currentDs.size(); i++) {
            nextFeats[i] = currentDs.features[i].clone();
            nextLbls[i]  = (i == idx) ? correctedLabel : currentDs.labels[i];
        }
        DatasetLoader.Dataset nextDs = new DatasetLoader.Dataset(nextFeats, nextLbls);
        String newSplitId = "SPLIT-CORRECTED-" + shortSha256Hex(sampleId + ":" + correctedLabel + ":" + Instant.now()).substring(0, 10);
        String splitComment = "SplitMetadata: splitId=" + newSplitId + ",protocol=TEST_LABEL_CORRECTED";

        commitDatasetAndManifestTransaction(
            featuresCsvPath, manifestPath, nextDs, existing, sampleId, null, false, splitComment
        );
        String warning = "WARNING: Held-out TEST sample '" + sampleId + "' was relabelled; new splitId='"
            + newSplitId + "' supersedes and invalidates all previously reported test scores.";
        return new TestCorrectionResult(updated, newSplitId, warning);
    }

    /**
     * Executes a state-tracked, recoverable transaction across {@code features.csv},
     * {@code manifest.csv}, and the optional ROI crop image (PNC-REM-02).
     *
     * <p>Backups and the recovery journal are retained until all targets (including the crop)
     * have been installed and verified, and the journal transitions to {@link TransactionState#COMMITTED}.
     */
    static void commitDatasetAndManifestTransaction(
            String featuresCsvPath,
            String manifestPath,
            DatasetLoader.Dataset nextDataset,
            List<ManifestRecord> nextManifest,
            String sampleId,
            BufferedImage roiCrop,
            boolean saveCropImage,
            String optionalSplitComment) {
        verifyStrictAlignment(nextDataset, nextManifest);

        Path fPath   = Path.of(featuresCsvPath).toAbsolutePath();
        Path mPath   = Path.of(manifestPath).toAbsolutePath();
        Path fTmp    = Path.of(featuresCsvPath + ".tmp").toAbsolutePath();
        Path mTmp    = Path.of(manifestPath + ".tmp").toAbsolutePath();
        Path fBak    = Path.of(featuresCsvPath + ".bak").toAbsolutePath();
        Path mBak    = Path.of(manifestPath + ".bak").toAbsolutePath();
        Path journal = Path.of(manifestPath + ".tx.journal").toAbsolutePath();

        Path cropDst = null;
        Path cropTmp = null;
        Path cropBak = null;
        if (saveCropImage) {
            Path cropDir = (mPath.getParent() != null)
                ? mPath.getParent().resolve("crops")
                : Path.of(DEFAULT_CROPS_DIR).toAbsolutePath();
            cropDst = cropDir.resolve(sampleId + ".png").toAbsolutePath();
            cropTmp = cropDir.resolve(sampleId + ".png.tmp").toAbsolutePath();
            cropBak = cropDir.resolve(sampleId + ".png.bak").toAbsolutePath();
        }

        boolean fExisted    = Files.exists(fPath);
        boolean mExisted    = Files.exists(mPath);
        boolean cropExisted = cropDst != null && Files.exists(cropDst);

        try {
            if (fPath.getParent() != null) Files.createDirectories(fPath.getParent());
            if (mPath.getParent() != null) Files.createDirectories(mPath.getParent());

            // 1. Prepare and validate all temporary files BEFORE backing up or modifying any target
            if (saveCropImage) {
                if (roiCrop == null || roiCrop.getWidth() <= 0 || roiCrop.getHeight() <= 0) {
                    throw new IOException("ROI crop image is null or has invalid dimensions.");
                }
                Files.createDirectories(cropDst.getParent());
                boolean ok = ImageIO.write(roiCrop, "png", cropTmp.toFile());
                if (!ok || !Files.exists(cropTmp) || Files.size(cropTmp) == 0) {
                    Files.deleteIfExists(cropTmp);
                    throw new IOException("ImageIO failed to write temporary PNG crop.");
                }
                BufferedImage decodedCrop = ImageIO.read(cropTmp.toFile());
                if (decodedCrop == null || decodedCrop.getWidth() != roiCrop.getWidth()
                        || decodedCrop.getHeight() != roiCrop.getHeight()) {
                    Files.deleteIfExists(cropTmp);
                    throw new IOException("Temporary PNG crop verification failed on decode.");
                }
            }

            List<String> leadingComments = new ArrayList<>();
            if (fExisted) {
                for (String line : Files.readAllLines(fPath, StandardCharsets.UTF_8)) {
                    if (line.trim().startsWith("#")) {
                        leadingComments.add(line);
                    } else if (!line.trim().isEmpty()) {
                        break;
                    }
                }
            }

            String splitCommentToWrite = optionalSplitComment;
            if (splitCommentToWrite == null && mExisted) {
                for (String line : Files.readAllLines(mPath, StandardCharsets.UTF_8)) {
                    if (line.trim().startsWith("# SplitMetadata:")) {
                        splitCommentToWrite = line.trim().substring(1).trim();
                        break;
                    }
                }
            }

            try (BufferedWriter fw = Files.newBufferedWriter(fTmp, StandardCharsets.UTF_8)) {
                for (String c : leadingComments) {
                    fw.write(c);
                    fw.newLine();
                }
                fw.write(FeatureVector.csvHeader());
                fw.newLine();
                for (int i = 0; i < nextDataset.size(); i++) {
                    FeatureVector fv = new FeatureVector(nextDataset.features[i]);
                    fw.write(fv.toCsvRow(FeatureVector.FEATURE_COUNT) + "," + nextDataset.labels[i].name());
                    fw.newLine();
                }
            }
            saveManifest(mTmp.toString(), nextManifest, splitCommentToWrite);

            DatasetLoader.Dataset verifyDs = DatasetLoader.load(fTmp.toString());
            List<ManifestRecord> verifyMan = loadManifest(mTmp.toString());
            verifyStrictAlignment(verifyDs, verifyMan);

            // 2. Back up all existing targets (including existing destination crop if present)
            if (fExisted) Files.copy(fPath, fBak, StandardCopyOption.REPLACE_EXISTING);
            if (mExisted) Files.copy(mPath, mBak, StandardCopyOption.REPLACE_EXISTING);
            if (cropExisted) Files.copy(cropDst, cropBak, StandardCopyOption.REPLACE_EXISTING);

            // 3. Write PREPARED journal state
            writeJournalState(journal, TransactionState.PREPARED,
                fPath, fTmp, fBak, fExisted,
                mPath, mTmp, mBak, mExisted,
                cropDst, cropTmp, cropBak, cropExisted);

            if (faultInjectionStage == FaultInjectionStage.BEFORE_REPLACEMENT) {
                throw new IOException("Injected fault in PREPARED state before any target replacement.");
            }

            // 4. Transition journal to COMMITTING and replace targets
            writeJournalState(journal, TransactionState.COMMITTING,
                fPath, fTmp, fBak, fExisted,
                mPath, mTmp, mBak, mExisted,
                cropDst, cropTmp, cropBak, cropExisted);

            moveWithAtomicFallback(fTmp, fPath);

            if (faultInjectionStage == FaultInjectionStage.AFTER_FEATURES_REPLACEMENT) {
                throw new IOException("Injected fault after features.csv replacement.");
            }

            moveWithAtomicFallback(mTmp, mPath);

            if (faultInjectionStage == FaultInjectionStage.AFTER_MANIFEST_REPLACEMENT) {
                throw new IOException("Injected fault after manifest.csv replacement.");
            }

            if (saveCropImage && cropDst != null && cropTmp != null) {
                if (faultInjectionStage == FaultInjectionStage.DURING_CROP_REPLACEMENT) {
                    throw new IOException("Injected fault during final crop replacement.");
                }
                moveWithAtomicFallback(cropTmp, cropDst);
            }

            // 5. Re-read and verify all installed final targets before marking COMMITTED
            DatasetLoader.Dataset finalDs = DatasetLoader.load(fPath.toString());
            List<ManifestRecord> finalMan = loadManifest(mPath.toString());
            verifyStrictAlignment(finalDs, finalMan);
            if (saveCropImage && cropDst != null) {
                if (!Files.exists(cropDst) || Files.size(cropDst) == 0 || ImageIO.read(cropDst.toFile()) == null) {
                    throw new IOException("Installed final crop failed post-move verification.");
                }
            }

            if (faultInjectionStage == FaultInjectionStage.BEFORE_COMMIT_FINALIZE) {
                throw new IOException("Injected fault before transitioning journal to COMMITTED.");
            }

            // 6. Atomically transition journal to COMMITTED, then clean up backups/temps and remove journal last
            writeJournalState(journal, TransactionState.COMMITTED,
                fPath, fTmp, fBak, fExisted,
                mPath, mTmp, mBak, mExisted,
                cropDst, cropTmp, cropBak, cropExisted);

            cleanupCommittedArtifacts(journal, fTmp, fBak, mTmp, mBak, cropTmp, cropBak);
        } catch (Exception txEx) {
            if (!Files.exists(journal)) {
                // Failed before journal creation: simply remove any temporary files
                try {
                    Files.deleteIfExists(fTmp);
                    Files.deleteIfExists(mTmp);
                    if (cropTmp != null) Files.deleteIfExists(cropTmp);
                } catch (IOException ignored) {}
                throw new IllegalStateException(
                    "Transaction preparation failed before target modification: " + txEx.getMessage(), txEx);
            }
            // Roll back all targets using the journal
            performRollbackFromJournal(journal, txEx);
        }
    }

    private static void writeJournalState(
            Path journal,
            TransactionState state,
            Path fPath, Path fTmp, Path fBak, boolean fExisted,
            Path mPath, Path mTmp, Path mBak, boolean mExisted,
            Path cropDst, Path cropTmp, Path cropBak, boolean cropExisted) throws IOException {
        Path journalTmp = Path.of(journal.toString() + ".tmp");
        StringBuilder sb = new StringBuilder();
        sb.append("state=").append(state.name()).append('\n');
        sb.append("fPath=").append(fPath).append('\n');
        sb.append("fTmp=").append(fTmp).append('\n');
        sb.append("fBak=").append(fBak).append('\n');
        sb.append("fExisted=").append(fExisted).append('\n');
        sb.append("mPath=").append(mPath).append('\n');
        sb.append("mTmp=").append(mTmp).append('\n');
        sb.append("mBak=").append(mBak).append('\n');
        sb.append("mExisted=").append(mExisted).append('\n');
        sb.append("cropDst=").append(cropDst != null ? cropDst : "").append('\n');
        sb.append("cropTmp=").append(cropTmp != null ? cropTmp : "").append('\n');
        sb.append("cropBak=").append(cropBak != null ? cropBak : "").append('\n');
        sb.append("cropExisted=").append(cropExisted).append('\n');
        Files.writeString(journalTmp, sb.toString(), StandardCharsets.UTF_8);
        moveWithAtomicFallback(journalTmp, journal);
    }

    private static void moveWithAtomicFallback(Path src, Path dst) throws IOException {
        try {
            Files.move(src, dst, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException ex) {
            Files.move(src, dst, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void cleanupCommittedArtifacts(
            Path journal, Path fTmp, Path fBak, Path mTmp, Path mBak, Path cropTmp, Path cropBak) throws IOException {
        Files.deleteIfExists(fTmp);
        Files.deleteIfExists(mTmp);
        if (cropTmp != null) Files.deleteIfExists(cropTmp);
        Files.deleteIfExists(fBak);
        Files.deleteIfExists(mBak);
        if (cropBak != null) Files.deleteIfExists(cropBak);
        // Remove journal last
        Files.deleteIfExists(journal);
    }

    private static void performRollbackFromJournal(Path journal, Exception cause) {
        if (simulateRollbackFailureForTesting) {
            throw new IllegalStateException(
                "RECOVERY_REQUIRED: Rollback failed while restoring transaction backups; journal and backups retained at "
                + journal + " (original cause: " + (cause != null ? cause.getMessage() : "unknown") + ")", cause);
        }
        try {
            Map<String, String> kv = parseJournalFile(journal);
            Path fPath = Path.of(kv.get("fPath"));
            Path fTmp  = Path.of(kv.get("fTmp"));
            Path fBak  = Path.of(kv.get("fBak"));
            boolean fExisted = Boolean.parseBoolean(kv.getOrDefault("fExisted", "true"));

            Path mPath = Path.of(kv.get("mPath"));
            Path mTmp  = Path.of(kv.get("mTmp"));
            Path mBak  = Path.of(kv.get("mBak"));
            boolean mExisted = Boolean.parseBoolean(kv.getOrDefault("mExisted", "true"));

            String cropDstStr = kv.getOrDefault("cropDst", "");
            String cropTmpStr = kv.getOrDefault("cropTmp", "");
            String cropBakStr = kv.getOrDefault("cropBak", "");
            boolean cropExisted = Boolean.parseBoolean(kv.getOrDefault("cropExisted", "false"));

            if (fExisted) {
                if (!Files.exists(fBak)) {
                    throw new IOException("Missing features backup during rollback: " + fBak);
                }
                Files.copy(fBak, fPath, StandardCopyOption.REPLACE_EXISTING);
            } else {
                Files.deleteIfExists(fPath);
            }

            if (mExisted) {
                if (!Files.exists(mBak)) {
                    throw new IOException("Missing manifest backup during rollback: " + mBak);
                }
                Files.copy(mBak, mPath, StandardCopyOption.REPLACE_EXISTING);
            } else {
                Files.deleteIfExists(mPath);
            }

            if (!cropDstStr.isBlank()) {
                Path cropDst = Path.of(cropDstStr);
                Path cropBak = Path.of(cropBakStr);
                if (cropExisted) {
                    if (!Files.exists(cropBak)) {
                        throw new IOException("Missing crop backup during rollback: " + cropBak);
                    }
                    Files.copy(cropBak, cropDst, StandardCopyOption.REPLACE_EXISTING);
                } else {
                    Files.deleteIfExists(cropDst);
                }
                if (!cropTmpStr.isBlank()) Files.deleteIfExists(Path.of(cropTmpStr));
                if (!cropBakStr.isBlank()) Files.deleteIfExists(cropBak);
            }

            Files.deleteIfExists(fTmp);
            Files.deleteIfExists(mTmp);
            Files.deleteIfExists(fBak);
            Files.deleteIfExists(mBak);
            Files.deleteIfExists(journal);
        } catch (Exception rbEx) {
            throw new IllegalStateException(
                "RECOVERY_REQUIRED: Rollback could not complete (" + rbEx.getMessage()
                + "); journal and recovery files retained at " + journal, rbEx);
        }

        if (cause != null) {
            throw new IllegalStateException(
                "Dataset/manifest/crop transaction failed and rolled back cleanly: " + cause.getMessage(), cause);
        }
    }

    /**
     * State-aware crash recovery (PNC-REM-02):
     * <ul>
     *   <li>For {@code PREPARED} or {@code COMMITTING}: restores pre-transaction {@code features.csv},
     *       {@code manifest.csv}, and previous {@code crop.png} (or deletes a newly created crop if none
     *       existed before), then removes temp/backup files and the journal.</li>
     *   <li>For {@code COMMITTED}: validates final installed targets and completes cleanup of
     *       temp/backup files and the journal without rolling back committed data.</li>
     * </ul>
     */
    public static void recoverInterruptedTransactionIfNeeded(String featuresCsvPath, String manifestPath) {
        Path journal = Path.of(manifestPath + ".tx.journal").toAbsolutePath();
        if (!Files.exists(journal)) return;

        if (simulateRollbackFailureForTesting) {
            throw new IllegalStateException(
                "RECOVERY_REQUIRED: Simulated recovery failure; journal and backups retained at " + journal);
        }

        try {
            Map<String, String> kv = parseJournalFile(journal);
            String stateStr = kv.getOrDefault("state", "COMMITTING");
            TransactionState state = TransactionState.valueOf(stateStr);

            Path fPath = Path.of(kv.get("fPath"));
            Path fTmp  = Path.of(kv.get("fTmp"));
            Path fBak  = Path.of(kv.get("fBak"));
            Path mPath = Path.of(kv.get("mPath"));
            Path mTmp  = Path.of(kv.get("mTmp"));
            Path mBak  = Path.of(kv.get("mBak"));
            String cropDstStr = kv.getOrDefault("cropDst", "");
            String cropTmpStr = kv.getOrDefault("cropTmp", "");
            String cropBakStr = kv.getOrDefault("cropBak", "");
            Path cropTmp = cropTmpStr.isBlank() ? null : Path.of(cropTmpStr);
            Path cropBak = cropBakStr.isBlank() ? null : Path.of(cropBakStr);

            if (state == TransactionState.COMMITTED) {
                DatasetLoader.Dataset ds = DatasetLoader.load(fPath.toString());
                List<ManifestRecord> man = loadManifest(mPath.toString());
                verifyStrictAlignment(ds, man);
                if (!cropDstStr.isBlank()) {
                    Path cropDst = Path.of(cropDstStr);
                    if (!Files.exists(cropDst) || Files.size(cropDst) == 0) {
                        throw new IOException("COMMITTED journal target crop is missing or empty: " + cropDst);
                    }
                }
                cleanupCommittedArtifacts(journal, fTmp, fBak, mTmp, mBak, cropTmp, cropBak);
            } else {
                performRollbackFromJournal(journal, null);
            }
        } catch (IllegalStateException ise) {
            throw ise;
        } catch (Exception e) {
            throw new IllegalStateException(
                "RECOVERY_REQUIRED: Failed to recover interrupted transaction at " + journal + ": " + e.getMessage(), e);
        }
    }

    private static Map<String, String> parseJournalFile(Path journal) throws IOException {
        Map<String, String> map = new HashMap<>();
        for (String line : Files.readAllLines(journal, StandardCharsets.UTF_8)) {
            int eq = line.indexOf('=');
            if (eq > 0) {
                map.put(line.substring(0, eq).trim(), line.substring(eq + 1).trim());
            }
        }
        return map;
    }

    public static String readPersistedSplitMetadata(String manifestPath) {
        Path p = Path.of(manifestPath);
        if (!Files.exists(p)) return "";
        try {
            for (String line : Files.readAllLines(p, StandardCharsets.UTF_8)) {
                String t = line.trim();
                if (t.startsWith("# SplitMetadata:")) {
                    return t.substring("# SplitMetadata:".length()).trim();
                }
            }
        } catch (IOException ignored) {}
        return "";
    }

    public static String computeFileSha256OrEmpty(Path filePath) {
        if (filePath == null) {
            return "";
        }
        try {
            if (!Files.isRegularFile(filePath)) {
                return "";
            }
            long size = Files.size(filePath);
            long lastModified = Files.getLastModifiedTime(filePath).toMillis();
            String normPath = filePath.toAbsolutePath().normalize().toString();
            FileMetadataKey key = new FileMetadataKey(normPath, size, lastModified);

            String cached = FILE_HASH_CACHE.get(key);
            if (cached != null) {
                return cached;
            }

            HASH_COMPUTATION_COUNT.incrementAndGet();

            MessageDigest md = MessageDigest.getInstance("SHA-256");
            try (InputStream is = new BufferedInputStream(Files.newInputStream(filePath), 65536)) {
                byte[] buf = new byte[65536];
                int n;
                while ((n = is.read(buf)) != -1) {
                    md.update(buf, 0, n);
                }
            }
            byte[] digest = md.digest();
            StringBuilder sb = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                sb.append(String.format(Locale.ROOT, "%02x", b & 0xff));
            }
            String hex = sb.toString();
            FILE_HASH_CACHE.put(key, hex);
            return hex;
        } catch (Exception e) {
            return "";
        }
    }

    public static String shortSha256Hex(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(input.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                sb.append(String.format(Locale.ROOT, "%02x", b & 0xff));
            }
            return sb.toString();
        } catch (Exception e) {
            return Integer.toHexString(input.hashCode());
        }
    }

    private static boolean featuresMatch(FeatureVector a, FeatureVector b) {
        if (a == null || b == null) return false;
        for (int i = 0; i < FeatureVector.FEATURE_COUNT; i++) {
            if (Math.abs(a.get(i) - b.get(i)) > 1e-4) return false;
        }
        return true;
    }

    private static List<ManifestRecord> salvageReadableLines(Path mPath, List<String> diagnostics) {
        List<ManifestRecord> list = new ArrayList<>();
        try {
            List<String> lines = Files.readAllLines(mPath, StandardCharsets.UTF_8);
            for (int idx = 0; idx < lines.size(); idx++) {
                String line = lines.get(idx).trim();
                if (line.isEmpty() || line.startsWith("#") || line.toLowerCase(Locale.ROOT).startsWith("sampleid,")) {
                    continue;
                }
                String[] p = line.split(",", -1);
                if (p.length != EXPECTED_COLUMN_COUNT) continue;
                try {
                    double[] fv = new double[8];
                    for (int i = 0; i < 8; i++) fv[i] = Double.parseDouble(p[8 + i].trim());
                    SlotStatus lbl = SlotStatus.fromString(p[16].trim());
                    String part = p[20].trim().toUpperCase(Locale.ROOT);
                    if (!VALID_PARTITIONS.contains(part)) part = "TRAIN";
                    list.add(new ManifestRecord(
                        p[0].trim(), p[1].trim(), p[2].trim(),
                        Integer.parseInt(p[3].trim()), Integer.parseInt(p[4].trim()),
                        Integer.parseInt(p[5].trim()), Integer.parseInt(p[6].trim()),
                        Integer.parseInt(p[7].trim()),
                        new FeatureVector(fv), lbl, FeatureVector.SCHEMA_VERSION,
                        p[18].trim(), p[19].trim(), part, p[21].trim()
                    ));
                } catch (Exception ignored) {
                    diagnostics.add("Skipped corrupt manifest line " + (idx + 1));
                }
            }
        } catch (IOException ignored) {}
        return list;
    }

    private static List<ManifestRecord> buildBundledDemoManifestForDataset(DatasetLoader.Dataset ds) {
        List<ManifestRecord> list = new ArrayList<>(ds.size());
        int[][] demoSlotCoords = {
            {1, 10, 10, 80, 100},
            {6, 100, 120, 80, 100},
            {7, 190, 120, 80, 100},
            {12, 280, 230, 80, 100}
        };

        for (int i = 0; i < ds.size(); i++) {
            String sampleId = String.format(Locale.ROOT, "SMP-%03d", i + 1);
            double[] raw = ds.features[i];
            FeatureVector fv = new FeatureVector(raw);
            SlotStatus label = ds.labels[i];

            String sourceImageId;
            String captureGroup;
            int slotId;
            int x = 10, y = 10, w = 80, h = 100;
            String reviewStatus;
            String provenance;
            String partition;

            if (ds.size() == 48 && i < 4) {
                sourceImageId = "data/sample/sample_parking_lot.png";
                captureGroup  = "LOT_A_DEMO";
                slotId        = demoSlotCoords[i][0];
                x             = demoSlotCoords[i][1];
                y             = demoSlotCoords[i][2];
                w             = demoSlotCoords[i][3];
                h             = demoSlotCoords[i][4];
                reviewStatus  = "VERIFIED_DEMO";
                provenance    = "DEMO_LOT_EXTRACTED";
                partition     = "TRAIN";
            } else if (ds.size() == 48 && i == 24) {
                sourceImageId = "data/sample/sample_parking_lot.png";
                captureGroup  = "LOT_A_DEMO";
                slotId        = 2;
                x             = 100;
                y             = 10;
                w             = 80;
                h             = 100;
                reviewStatus  = "VERIFIED_DEMO";
                provenance    = "DEMO_LOT_EXTRACTED";
                partition     = "TRAIN";
            } else {
                sourceImageId = "procedural://multi_lot_calibration/scene_" + (i + 1);
                slotId        = (i % 12) + 1;
                reviewStatus  = "UNVERIFIED_SYNTHETIC";
                provenance    = "PROCEDURAL_SYNTHETIC";

                if (i == 5 || i == 6 || i == 16 || i == 22 || i == 26 || i == 34 || i == 40 || i == 41) {
                    captureGroup = (i < 24) ? "LOT_TEST_OCC_HOLD" : "LOT_TEST_AVAIL_HOLD";
                    partition    = "TEST";
                } else if (i == 7 || i == 8 || i == 18 || i == 23 || i == 27 || i == 35 || i == 42 || i == 43) {
                    captureGroup = (i < 24) ? "LOT_VAL_OCC_CALIB" : "LOT_VAL_AVAIL_CALIB";
                    partition    = "VALIDATION";
                } else {
                    int grpIdx = (i % 4) + 1;
                    captureGroup = "LOT_TRAIN_SYNTH_" + grpIdx;
                    partition    = "TRAIN";
                }
            }

            list.add(new ManifestRecord(
                sampleId,
                sourceImageId,
                captureGroup,
                slotId, x, y, w, h,
                fv,
                label,
                FeatureVector.SCHEMA_VERSION,
                reviewStatus,
                provenance,
                partition,
                "2026-10-21T00:00:00Z"
            ));
        }
        return list;
    }
}
