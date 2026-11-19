package com.parknchillai.app;

import com.parknchillai.ai.ClassifierBenchmark;
import com.parknchillai.ai.DatasetLoader;
import com.parknchillai.ai.DatasetManifest;
import com.parknchillai.ai.DatasetValidator;
import com.parknchillai.ai.DecisionTreeModel;
import com.parknchillai.ai.DecisionTreePredictor;
import com.parknchillai.ai.DecisionTreeTrainer;
import com.parknchillai.ai.ModelEvaluator;
import com.parknchillai.ai.SlotTrainingRecord;
import com.parknchillai.ai.UncertaintyPolicy;
import com.parknchillai.detection.DetectionStrategy;
import com.parknchillai.detection.OccupancyDecisionEngine;
import com.parknchillai.detection.OnnxVehicleDetectionService;
import com.parknchillai.detection.SlotOverlapAnalyzer;
import com.parknchillai.detection.VehicleDetection;
import com.parknchillai.detection.VehicleDetectionService;
import com.parknchillai.features.FeatureExtractor;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.imageprocessing.EnhancedImage;
import com.parknchillai.imageprocessing.GrayscaleConverter;
import com.parknchillai.imageprocessing.ImageEnhancementService;
import com.parknchillai.imageprocessing.OriginalImage;
import com.parknchillai.imageprocessing.ProcessingPolicy;
import com.parknchillai.layout.LayoutDetectionResult;
import com.parknchillai.layout.ParkingLayoutDetector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.RoiExtractor;
import com.parknchillai.parking.SlotStatus;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Core application service orchestrating end-to-end image processing, vehicle detection,
 * polygon-aware slot overlap, and partition-aware CART Decision Tree training &amp; evaluation
 * (ParkNChill v2.3 Requirements 1–5).
 */
public class ParkingAnalysisService {

    public static final String DEFAULT_DATASET_PATH   = "data/labels/features.csv";
    public static final String DEFAULT_MANIFEST_PATH  = DatasetManifest.DEFAULT_MANIFEST_PATH;
    public static final String DEFAULT_PKLOT_TEST_META_PATH = "data/labels/pklot_frozen_test.meta";
    public static final double DEFAULT_TRAIN_FRACTION = 0.75;
    public static final long   DEFAULT_SPLIT_SEED     = 42L;
    public static final double DEFAULT_OVERLAY_FILL_OPACITY = 0.12;

    /**
     * Immutable cryptographic snapshot of the protected held-out test partition,
     * freezing test capture groups, sample count, feature/label SHA-256 digest, and evaluation version.
     */
    public record TestSnapshot(
            int evaluationVersion,
            int testRowCount,
            Set<String> testGroups,
            String snapshotSha256
    ) {}

    /**
     * Detailed audit summary of a partition-aware PKLot Decision Tree training &amp; hyperparameter
     * selection run (Requirements 2 &amp; 3).
     */
    public record PkLotTrainingSummary(
            int totalRecords,
            int trainCount,
            int validationCount,
            int testCount,
            Set<String> trainGroups,
            Set<String> validationGroups,
            Set<String> testGroups,
            int importedGroundTruthCount,
            int humanVerifiedOrCorrectedCount,
            int selectedMaxDepth,
            int selectedMinSamplesPerLeaf,
            ModelEvaluator.EvaluationReport validationReport,
            ModelEvaluator.EvaluationReport heldOutTestReport,
            TestSnapshot frozenTestSnapshot,
            String selectionRationale
    ) {}

    private DecisionTreePredictor predictor;
    private VehicleDetectionService vehicleDetectionService = new OnnxVehicleDetectionService();
    private DetectionStrategy activeDetectionStrategy = DetectionStrategy.ACCURATE;
    private SlotOverlapAnalyzer.OverlapPolicy overlapPolicy = SlotOverlapAnalyzer.OverlapPolicy.defaultValidationTunedPolicy();
    private OccupancyDecisionEngine.AnalysisMode analysisMode = OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION;
    private double overlayFillOpacity = DEFAULT_OVERLAY_FILL_OPACITY;
    private boolean localTrainingDatasetLoaded = false;

    private DatasetLoader.Dataset lastTrainDataset;
    private DatasetLoader.Dataset lastValidationDataset;
    private DatasetLoader.Dataset lastTestDataset;
    private DatasetLoader.GroupedPartitions lastGroupedPartitions;
    private DatasetValidator.ValidationSummary lastValidationSummary;
    private List<DatasetManifest.ManifestRecord> manifestRecords = List.of();
    private TestSnapshot frozenTestSnapshot;
    private PkLotTrainingSummary lastPkLotTrainingSummary;
    private int evaluationVersion = 1;
    private boolean useAdaptiveThreshold = false;
    private String activeDatasetPath = DEFAULT_DATASET_PATH;
    private String activeManifestPath = DEFAULT_MANIFEST_PATH;
    private int currentMaxDepth = 4;
    private int currentMinSamples = 2;

    public ParkingAnalysisService(DecisionTreePredictor predictor) {
        if (predictor == null) {
            throw new IllegalArgumentException("Predictor must not be null.");
        }
        this.predictor = predictor;
    }

    public static ParkingAnalysisService createWithDefaultModel() {
        File defaultCsv = new File(DEFAULT_DATASET_PATH);
        if (!defaultCsv.exists()) {
            return createWithoutTrainingDataset(new OnnxVehicleDetectionService());
        }
        return createWithDataset(DEFAULT_DATASET_PATH, 4, 2);
    }

    public static ParkingAnalysisService createWithoutTrainingDataset(VehicleDetectionService detector) {
        DecisionTreeModel fallbackTree = buildFallbackStructuralTree();
        ParkingAnalysisService svc = new ParkingAnalysisService(new DecisionTreePredictor(fallbackTree));
        svc.localTrainingDatasetLoaded = false;
        svc.analysisMode = OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION;
        if (detector != null) {
            svc.vehicleDetectionService = detector;
        }
        return svc;
    }

    private static DecisionTreeModel buildFallbackStructuralTree() {
        double[][] fallbackFeatures = new double[][]{
            {63.7, 0.067, 0.885, 0.195, 0.022, 17.5, 0.000, 0.022},
            {64.0, 0.068, 0.880, 0.200, 0.023, 18.0, 0.000, 0.023},
            {62.0, 0.050, 0.890, 0.180, 0.020, 15.0, 0.000, 0.020},
            {79.4, 0.484, 0.549, 0.473, 0.439, 117.3, 0.608, 0.374},
            {82.1, 0.448, 0.548, 0.502, 0.405, 119.3, 0.610, 0.349},
            {112.6, 0.458, 0.465, 0.520, 0.420, 131.7, 0.608, 0.398}
        };
        SlotStatus[] fallbackLabels = new SlotStatus[]{
            SlotStatus.AVAILABLE, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE,
            SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED
        };
        return new DecisionTreeTrainer(4, 2).train(fallbackFeatures, fallbackLabels);
    }

    public static ParkingAnalysisService createWithDataset(String datasetCsvPath, int maxDepth, int minSamples) {
        File file = new File(datasetCsvPath);
        if (!file.exists()) {
            throw new IllegalArgumentException("Dataset file not found: " + datasetCsvPath);
        }
        String manifestPath = resolveManifestPathFor(datasetCsvPath);
        DatasetLoader.Dataset fullDataset = DatasetLoader.load(datasetCsvPath);
        List<DatasetManifest.ManifestRecord> manifest = DatasetManifest.loadOrSync(datasetCsvPath, manifestPath);
        DatasetManifest.verifyStrictAlignment(fullDataset, manifest);
        DatasetValidator.ValidationSummary valSummary = DatasetValidator.validateOrThrow(fullDataset, manifest);

        DatasetLoader.GroupedPartitions partitions = buildPartitions(fullDataset, manifest);
        DecisionTreeTrainer trainer = new DecisionTreeTrainer(maxDepth, minSamples);
        DecisionTreeModel model = trainer.train(partitions.train().features, partitions.train().labels);

        ParkingAnalysisService svc = new ParkingAnalysisService(new DecisionTreePredictor(model));
        svc.analysisMode          = OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION;
        svc.localTrainingDatasetLoaded = true;
        svc.lastTrainDataset      = partitions.train();
        svc.lastValidationDataset = partitions.validation();
        svc.lastTestDataset       = partitions.test();
        svc.lastGroupedPartitions = partitions;
        svc.lastValidationSummary = valSummary;
        svc.manifestRecords       = manifest;
        svc.evaluationVersion     = 1;
        svc.frozenTestSnapshot    = computeTestSnapshot(1, partitions.test(), partitions.testGroups());
        svc.activeDatasetPath     = datasetCsvPath;
        svc.activeManifestPath    = manifestPath;
        svc.currentMaxDepth       = maxDepth;
        svc.currentMinSamples     = minSamples;
        return svc;
    }

    private static String resolveManifestPathFor(String datasetCsvPath) {
        if (DEFAULT_DATASET_PATH.equals(datasetCsvPath)) {
            return DEFAULT_MANIFEST_PATH;
        }
        return datasetCsvPath + ".manifest.csv";
    }

    private static DatasetLoader.GroupedPartitions buildPartitions(
            DatasetLoader.Dataset fullDataset,
            List<DatasetManifest.ManifestRecord> manifest) {
        if (fullDataset.size() < 8) {
            return new DatasetLoader.GroupedPartitions(
                fullDataset, fullDataset, fullDataset,
                Set.of("SINGLE_GROUP"), Set.of("SINGLE_GROUP"), Set.of("SINGLE_GROUP"), false);
        }
        return DatasetLoader.buildValidatedPartitions(fullDataset, manifest, DEFAULT_SPLIT_SEED);
    }

    public static TestSnapshot computeTestSnapshot(int version,
                                                   DatasetLoader.Dataset testDs,
                                                   Set<String> testGroups) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            List<String> sortedGroups = new ArrayList<>(testGroups != null ? testGroups : Set.of());
            Collections.sort(sortedGroups);
            for (String g : sortedGroups) {
                md.update((g + "|").getBytes(StandardCharsets.UTF_8));
            }
            if (testDs != null) {
                int fCount = testDs.featureCount() > 0 ? testDs.featureCount() : FeatureVector.FEATURE_COUNT;
                for (int i = 0; i < testDs.size(); i++) {
                    FeatureVector fv = new FeatureVector(testDs.features[i]);
                    String row = fv.toCsvRow(fCount) + ":" + testDs.labels[i].name() + "\n";
                    md.update(row.getBytes(StandardCharsets.UTF_8));
                }
            }
            byte[] dig = md.digest();
            StringBuilder sb = new StringBuilder(dig.length * 2);
            for (byte b : dig) {
                sb.append(String.format(Locale.ROOT, "%02x", b & 0xff));
            }
            return new TestSnapshot(
                version,
                testDs != null ? testDs.size() : 0,
                testGroups != null ? Collections.unmodifiableSet(new LinkedHashSet<>(testGroups)) : Set.of(),
                sb.toString()
            );
        } catch (Exception e) {
            throw new IllegalStateException("Failed to compute test partition SHA-256 snapshot.", e);
        }
    }

    public LayoutDetectionResult autoDetectLayout(BufferedImage fullImage) {
        return ParkingLayoutDetector.detectLayout(fullImage);
    }

    public List<SlotTrainingRecord> generateProvisionalSlotRecords(String sourceImageId,
                                                                   String sourceImageSha256,
                                                                   String captureGroup,
                                                                   AnalysisSummary summary) {
        if (summary == null || summary.getSlotResults().isEmpty()) {
            return List.of();
        }
        String modelVer = String.format(Locale.ROOT, "ONNX-Detector+CART-v2.3(depth=%d,minLeaf=%d,evalV=%d)",
            currentMaxDepth, currentMinSamples, evaluationVersion);
        List<SlotTrainingRecord> out = new ArrayList<>(summary.getSlotResults().size());
        for (SlotAnalysisResult res : summary.getSlotResults()) {
            out.add(SlotTrainingRecord.createProvisionalFromSlotResult(
                sourceImageId, sourceImageSha256, captureGroup, res, modelVer
            ));
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * Partition-aware PKLot training workflow (ParkNChill v2.3 Requirements 2 &amp; 3):
     * <ul>
     *   <li>Consumes the exact saved {@code TRAIN}, {@code VALIDATION}, and {@code TEST} partition
     *       assignments on each {@link SlotTrainingRecord}—never re-splitting via {@code groupAwareSplit}.</li>
     *   <li>Verifies pairwise group disjointness ({@code TRAIN ∩ VAL = ∅}, {@code TRAIN ∩ TEST = ∅},
     *       {@code VAL ∩ TEST = ∅}), minimum 3 independent groups, and presence of both
     *       {@code OCCUPIED} and {@code AVAILABLE} classes in {@code TRAIN}.</li>
     *   <li>Freezes and persists the {@code TEST} group assignments and SHA-256 snapshot to
     *       {@code frozenTestMetaPath} (if non-null) and verifies that any existing frozen test
     *       snapshot has not been altered.</li>
     *   <li>Trains candidate models strictly on {@code TRAIN}, selects optimal {@code (maxDepth, minSamplesPerLeaf)}
     *       strictly on {@code VALIDATION}, promotes the winning 10-feature model only after validation
     *       succeeds, and evaluates once on the frozen {@code TEST} partition.</li>
     * </ul>
     */
    public PkLotTrainingSummary trainFromPartitionedPkLotRecords(
            List<SlotTrainingRecord> partitionedRecords,
            int requestedMaxDepth,
            int requestedMinSamplesPerLeaf,
            Path frozenTestMetaPath) {
        if (partitionedRecords == null || partitionedRecords.isEmpty()) {
            throw new IllegalArgumentException("PKLot training records must not be null or empty.");
        }

        List<double[]> trainFeats = new ArrayList<>();
        List<SlotStatus> trainLabels = new ArrayList<>();
        Set<String> trainGroups = new LinkedHashSet<>();

        List<double[]> valFeats = new ArrayList<>();
        List<SlotStatus> valLabels = new ArrayList<>();
        Set<String> valGroups = new LinkedHashSet<>();

        List<double[]> testFeats = new ArrayList<>();
        List<SlotStatus> testLabels = new ArrayList<>();
        Set<String> testGroups = new LinkedHashSet<>();

        int importedGroundTruthCount = 0;
        int humanVerifiedCount = 0;

        for (int i = 0; i < partitionedRecords.size(); i++) {
            SlotTrainingRecord rec = partitionedRecords.get(i);
            if (rec == null) {
                throw new IllegalArgumentException("PKLot record at index " + i + " must not be null.");
            }
            if (!rec.isEligibleForTraining(SlotTrainingRecord.TrainingEligibilityPolicy.HUMAN_VERIFIED_AND_IMPORTED_GROUND_TRUTH)) {
                throw new IllegalArgumentException(
                    "Record '" + rec.sampleId() + "' is not eligible for training (provisional="
                    + rec.provisional() + ", provenance=" + rec.provenance() + ").");
            }
            if (!FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION.equals(rec.featureSchemaVersion())) {
                throw new IllegalArgumentException(
                    "Model expects schema " + FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION
                    + "; dataset provides " + rec.featureSchemaVersion());
            }

            if (rec.isHumanVerified()) {
                humanVerifiedCount++;
            } else if (rec.isImportedGroundTruth()) {
                importedGroundTruthCount++;
            }

            double[] fv10 = rec.features().toArray(FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT);
            String part = rec.partition() != null ? rec.partition().trim().toUpperCase(Locale.ROOT) : "TRAIN";
            switch (part) {
                case "TRAIN" -> {
                    trainFeats.add(fv10);
                    trainLabels.add(rec.occupancyLabel());
                    trainGroups.add(rec.captureGroup());
                }
                case "VALIDATION", "VAL" -> {
                    valFeats.add(fv10);
                    valLabels.add(rec.occupancyLabel());
                    valGroups.add(rec.captureGroup());
                }
                case "TEST" -> {
                    testFeats.add(fv10);
                    testLabels.add(rec.occupancyLabel());
                    testGroups.add(rec.captureGroup());
                }
                default -> throw new IllegalArgumentException("Unsupported partition '" + part + "' on record " + rec.sampleId());
            }
        }

        if (trainGroups.isEmpty() || valGroups.isEmpty() || testGroups.isEmpty()) {
            throw new IllegalArgumentException(String.format(Locale.ROOT,
                "Refusing PKLot training: dataset must have independent capture groups in TRAIN (%d), VALIDATION (%d), and TEST (%d).",
                trainGroups.size(), valGroups.size(), testGroups.size()));
        }
        if (!Collections.disjoint(trainGroups, valGroups)
                || !Collections.disjoint(trainGroups, testGroups)
                || !Collections.disjoint(valGroups, testGroups)) {
            throw new IllegalArgumentException(
                "Refusing PKLot training: capture groups are not disjoint across TRAIN, VALIDATION, and TEST partitions.");
        }

        boolean trainHasOcc = trainLabels.contains(SlotStatus.OCCUPIED);
        boolean trainHasAvail = trainLabels.contains(SlotStatus.AVAILABLE);
        if (!trainHasOcc || !trainHasAvail) {
            throw new IllegalArgumentException(
                "Refusing PKLot training: TRAIN partition must contain both OCCUPIED and AVAILABLE samples.");
        }

        DatasetLoader.Dataset trainDs = new DatasetLoader.Dataset(
            trainFeats.toArray(new double[0][]), trainLabels.toArray(new SlotStatus[0]));
        DatasetLoader.Dataset valDs = new DatasetLoader.Dataset(
            valFeats.toArray(new double[0][]), valLabels.toArray(new SlotStatus[0]));
        DatasetLoader.Dataset testDs = new DatasetLoader.Dataset(
            testFeats.toArray(new double[0][]), testLabels.toArray(new SlotStatus[0]));

        TestSnapshot newTestSnapshot = computeTestSnapshot(this.evaluationVersion, testDs, testGroups);
        verifyAndPersistFrozenTestSnapshot(newTestSnapshot, frozenTestMetaPath);

        // Perform hyperparameter selection strictly using VALIDATION (never touching TEST!)
        int baseDepth = Math.max(1, Math.min(10, requestedMaxDepth));
        int baseMinLeaf = Math.max(1, Math.min(10, requestedMinSamplesPerLeaf));
        int[] candidateDepths = new int[]{
            baseDepth,
            Math.max(2, baseDepth - 1),
            Math.min(8, baseDepth + 1)
        };
        int[] candidateMinLeafs = new int[]{
            baseMinLeaf,
            1,
            Math.min(6, baseMinLeaf + 1)
        };

        DecisionTreeModel bestModel = null;
        ModelEvaluator.EvaluationReport bestValReport = null;
        int bestDepth = baseDepth;
        int bestMinLeaf = baseMinLeaf;
        double bestScore = -1.0;

        UncertaintyPolicy evalPolicy = this.predictor != null
            ? this.predictor.getUncertaintyPolicy()
            : UncertaintyPolicy.defaultProvisionalPolicy();

        for (int d : candidateDepths) {
            for (int m : candidateMinLeafs) {
                DecisionTreeModel candModel = new DecisionTreeTrainer(d, m).train(trainDs.features, trainDs.labels);
                DecisionTreePredictor candPred = new DecisionTreePredictor(candModel, evalPolicy);
                ModelEvaluator.EvaluationReport valRep = ModelEvaluator.evaluate(
                    candPred, valDs.features, valDs.labels,
                    ModelEvaluator.EvaluationScope.VALIDATION_PARTITION,
                    "Validation hyperparameter selection"
                );
                // Primary criterion: Validation F1 (or Validation Accuracy if single-class validation fold)
                double primaryMetric = valRep.metricsAvailable ? valRep.f1 : valRep.accuracy;
                double tieBreak = valRep.accuracy * 0.01 - Math.abs(d - baseDepth) * 0.0001;
                double composite = primaryMetric + tieBreak;
                if (bestModel == null || composite > bestScore + 1e-9) {
                    bestScore = composite;
                    bestModel = candModel;
                    bestValReport = valRep;
                    bestDepth = d;
                    bestMinLeaf = m;
                }
            }
        }

        if (bestModel == null || bestValReport == null) {
            throw new IllegalStateException("Hyperparameter validation failed to produce a valid 10-feature Decision Tree.");
        }

        // Promote the trained 10-feature model ONLY after training on TRAIN and validation on VALIDATION succeed!
        this.predictor = new DecisionTreePredictor(bestModel, evalPolicy);
        this.lastTrainDataset = trainDs;
        this.lastValidationDataset = valDs;
        this.lastTestDataset = testDs;
        this.lastGroupedPartitions = new DatasetLoader.GroupedPartitions(
            trainDs, valDs, testDs, trainGroups, valGroups, testGroups, true
        );
        this.currentMaxDepth = bestDepth;
        this.currentMinSamples = bestMinLeaf;
        this.frozenTestSnapshot = newTestSnapshot;

        String provenanceSummary = String.format(Locale.ROOT,
            "PKLot Partition-Aware Evaluation (Schema=%s | Policy=%s | Provenance: %d IMPORTED_GROUND_TRUTH, %d HUMAN_VERIFIED/CORRECTED | "
            + "Train=%d [%d groups], Val=%d [%d groups], Frozen Test=%d [%d groups] | Selected depth=%d, minLeaf=%d on Validation)",
            ProcessingPolicy.CANONICAL_SCHEMA_ID, ProcessingPolicy.CANONICAL_POLICY_ID,
            importedGroundTruthCount, humanVerifiedCount,
            trainDs.size(), trainGroups.size(),
            valDs.size(), valGroups.size(),
            testDs.size(), testGroups.size(),
            bestDepth, bestMinLeaf
        );

        // Evaluate ONCE on the frozen TEST partition
        ModelEvaluator.EvaluationReport heldOutTestReport = ModelEvaluator.evaluate(
            this.predictor,
            testDs.features,
            testDs.labels,
            ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION,
            provenanceSummary
        );

        String rationale = String.format(Locale.ROOT,
            "Trained on TRAIN (N=%d, groups=%s); selected (maxDepth=%d, minLeaf=%d) on VALIDATION (N=%d, groups=%s, Val Acc=%.1f%%, Val F1=%s); "
            + "evaluated once on frozen TEST (N=%d, groups=%s, sha256=%s) | Provenance: %d Imported Ground Truth, %d User-Verified.",
            trainDs.size(), trainGroups,
            bestDepth, bestMinLeaf,
            valDs.size(), valGroups,
            bestValReport.accuracy * 100.0, bestValReport.formatMetric(bestValReport.f1),
            testDs.size(), testGroups, newTestSnapshot.snapshotSha256().substring(0, 12),
            importedGroundTruthCount, humanVerifiedCount
        );

        this.lastPkLotTrainingSummary = new PkLotTrainingSummary(
            partitionedRecords.size(),
            trainDs.size(),
            valDs.size(),
            testDs.size(),
            Collections.unmodifiableSet(trainGroups),
            Collections.unmodifiableSet(valGroups),
            Collections.unmodifiableSet(testGroups),
            importedGroundTruthCount,
            humanVerifiedCount,
            bestDepth,
            bestMinLeaf,
            bestValReport,
            heldOutTestReport,
            newTestSnapshot,
            rationale
        );

        return this.lastPkLotTrainingSummary;
    }

    private void verifyAndPersistFrozenTestSnapshot(TestSnapshot candidate, Path metaPath) {
        if (metaPath == null) return;
        try {
            if (Files.exists(metaPath)) {
                List<String> lines = Files.readAllLines(metaPath, StandardCharsets.UTF_8);
                String savedGroups = null;
                String savedHash = null;
                int savedVersion = -1;
                for (String line : lines) {
                    if (line.startsWith("evaluationVersion=")) {
                        savedVersion = Integer.parseInt(line.substring("evaluationVersion=".length()).trim());
                    } else if (line.startsWith("testGroups=")) {
                        savedGroups = line.substring("testGroups=".length()).trim();
                    } else if (line.startsWith("snapshotSha256=")) {
                        savedHash = line.substring("snapshotSha256=".length()).trim();
                    }
                }
                String candGroups = String.join(",", candidate.testGroups());
                if (savedVersion == candidate.evaluationVersion()
                        && savedGroups != null && savedGroups.equals(candGroups)
                        && savedHash != null && !savedHash.equals(candidate.snapshotSha256())) {
                    throw new IllegalStateException(
                        "Frozen PKLot TEST partition integrity violation: test records for groups [" + candGroups
                        + "] changed hash from " + savedHash + " to " + candidate.snapshotSha256());
                }
            }
            if (metaPath.getParent() != null) {
                Files.createDirectories(metaPath.getParent());
            }
            String content = String.format(Locale.ROOT,
                "evaluationVersion=%d%ntestRowCount=%d%ntestGroups=%s%nsnapshotSha256=%s%n",
                candidate.evaluationVersion(),
                candidate.testRowCount(),
                String.join(",", candidate.testGroups()),
                candidate.snapshotSha256()
            );
            Files.writeString(metaPath, content, StandardCharsets.UTF_8);
        } catch (IOException ioe) {
            throw new IllegalStateException("Failed to persist frozen PKLot test partition metadata: " + ioe.getMessage(), ioe);
        }
    }

    /**
     * Trains and evaluates the CART Decision Tree from a collection of <strong>human-reviewed</strong>
     * {@link SlotTrainingRecord} examples collected through the interactive review workflow.
     * Strictly rejects unreviewed provisional records.
     */
    public ModelEvaluator.EvaluationReport trainFromReviewedSlotRecords(
            List<SlotTrainingRecord> reviewedRecords,
            int maxDepth,
            int minSamplesPerLeaf) {
        if (reviewedRecords == null || reviewedRecords.isEmpty()) {
            throw new IllegalArgumentException("Reviewed slot records must not be null or empty.");
        }
        List<double[]> featList = new ArrayList<>(reviewedRecords.size());
        List<SlotStatus> lblList = new ArrayList<>(reviewedRecords.size());
        List<String> grpList = new ArrayList<>(reviewedRecords.size());

        int humanCount = 0;
        int importedCount = 0;

        for (int i = 0; i < reviewedRecords.size(); i++) {
            SlotTrainingRecord rec = reviewedRecords.get(i);
            if (rec == null) {
                throw new IllegalArgumentException("SlotTrainingRecord at index " + i + " must not be null.");
            }
            if (rec.provisional() || !rec.isHumanVerified()) {
                throw new IllegalArgumentException(
                    "Refusing to train Decision Tree on unreviewed provisional record '" + rec.sampleId()
                    + "' (reviewStatus=" + rec.reviewStatus() + "). Human verification is required.");
            }
            if (!FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION.equals(rec.featureSchemaVersion())) {
                throw new IllegalArgumentException(
                    "Model expects schema " + FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION
                    + "; dataset provides " + rec.featureSchemaVersion());
            }
            if (rec.isImportedGroundTruth()) importedCount++;
            else humanCount++;

            featList.add(rec.features().toArray(FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT));
            lblList.add(rec.occupancyLabel());
            grpList.add(rec.captureGroup());
        }

        DatasetLoader.Dataset fullDs = new DatasetLoader.Dataset(
            featList.toArray(new double[0][]),
            lblList.toArray(new SlotStatus[0])
        );
        DatasetLoader.GroupedPartitions parts = DatasetLoader.groupAwareSplit(
            fullDs, grpList, DEFAULT_SPLIT_SEED);

        DatasetLoader.Dataset trainDs = parts.train();
        DatasetLoader.Dataset valDs   = parts.validation();
        DatasetLoader.Dataset testDs  = parts.test();

        DecisionTreeTrainer trainer = new DecisionTreeTrainer(maxDepth, minSamplesPerLeaf);
        DecisionTreeModel trainedModel = trainer.train(trainDs.features, trainDs.labels);

        UncertaintyPolicy prevPol = this.predictor != null
            ? this.predictor.getUncertaintyPolicy()
            : UncertaintyPolicy.defaultProvisionalPolicy();
        this.predictor             = new DecisionTreePredictor(trainedModel, prevPol);
        this.lastTrainDataset      = trainDs;
        this.lastValidationDataset = valDs;
        this.lastTestDataset       = testDs;
        this.lastGroupedPartitions = parts;
        this.currentMaxDepth       = maxDepth;
        this.currentMinSamples     = minSamplesPerLeaf;
        this.frozenTestSnapshot    = computeTestSnapshot(this.evaluationVersion, testDs, parts.testGroups());

        if (!parts.hasSufficientIndependentGroups() || testDs == null || testDs.size() == 0) {
            String provDisclaimer = String.format(Locale.ROOT,
                "Evaluation refused (%s): dataset has %d capture group(s); requires at least 3 independent groups for held-out evaluation. Provenance: %d HUMAN_VERIFIED, %d IMPORTED_GROUND_TRUTH",
                DatasetLoader.GroupedPartitions.INSUFFICIENT_GROUPS_LABEL,
                parts.trainGroups().size() + parts.validationGroups().size() + parts.testGroups().size(),
                humanCount, importedCount);
            return ModelEvaluator.insufficientGroupsResult(provDisclaimer);
        }

        ModelEvaluator.EvaluationScope scope = ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION;
        String provDisclaimer = String.format(Locale.ROOT,
            "Evaluated on held-out test partition (Schema=%s, Policy=%s, GroupDisjoint=true | Provenance: %d HUMAN_VERIFIED, %d IMPORTED_GROUND_TRUTH)",
            ProcessingPolicy.CANONICAL_SCHEMA_ID, ProcessingPolicy.CANONICAL_POLICY_ID,
            humanCount, importedCount);
        return ModelEvaluator.evaluate(this.predictor, testDs.features, testDs.labels, scope, provDisclaimer);
    }

    public AnalysisSummary analyzeAutomatically(BufferedImage fullImage) {
        return analyzeAutomatically(fullImage, this.useAdaptiveThreshold, this.analysisMode);
    }

    public AnalysisSummary analyzeAutomatically(BufferedImage fullImage,
                                                boolean useAdaptiveThreshold,
                                                OccupancyDecisionEngine.AnalysisMode mode) {
        if (fullImage == null) throw new IllegalArgumentException("Image must not be null.");
        LayoutDetectionResult layout = ParkingLayoutDetector.detectLayout(fullImage);
        return analyze(
            fullImage,
            layout.getSlots(),
            useAdaptiveThreshold,
            mode,
            layout.isLocalizationReliable(),
            layout.getLocalizationSummary()
        );
    }

    public AnalysisSummary analyze(BufferedImage fullImage, List<ParkingSlot> slots) {
        return analyze(fullImage, slots, this.useAdaptiveThreshold, this.analysisMode);
    }

    public AnalysisSummary analyze(BufferedImage fullImage, List<ParkingSlot> slots, boolean useAdaptiveThreshold) {
        return analyze(fullImage, slots, useAdaptiveThreshold, this.analysisMode);
    }

    public AnalysisSummary analyze(BufferedImage fullImage,
                                   List<ParkingSlot> slots,
                                   boolean useAdaptiveThreshold,
                                   OccupancyDecisionEngine.AnalysisMode mode) {
        boolean reliable = slots != null && !slots.isEmpty();
        String locMsg = reliable
            ? ("Evaluated " + slots.size() + " parking space(s).")
            : "Parking-slot localization is uncertain: 0 parking spaces localized; displaying vehicle detections only.";
        return analyze(fullImage, slots, useAdaptiveThreshold, mode, reliable, locMsg);
    }

    public AnalysisSummary analyze(BufferedImage fullImage,
                                   List<ParkingSlot> slots,
                                   boolean useAdaptiveThreshold,
                                   OccupancyDecisionEngine.AnalysisMode mode,
                                   boolean slotLocalizationReliable,
                                   String slotLocalizationMessage) {
        return analyze(fullImage, fullImage, slots, useAdaptiveThreshold, mode, slotLocalizationReliable, slotLocalizationMessage);
    }

    public AnalysisSummary analyze(BufferedImage originalImage,
                                   BufferedImage workingImage,
                                   List<ParkingSlot> slots,
                                   boolean useAdaptiveThreshold,
                                   OccupancyDecisionEngine.AnalysisMode mode,
                                   boolean slotLocalizationReliable,
                                   String slotLocalizationMessage) {
        return analyze(originalImage, workingImage, null, slots, useAdaptiveThreshold, mode, slotLocalizationReliable, slotLocalizationMessage);
    }

    public AnalysisSummary analyze(BufferedImage originalImage,
                                   BufferedImage workingImage,
                                   String sourceImageSha256,
                                   List<ParkingSlot> slots,
                                   boolean useAdaptiveThreshold,
                                   OccupancyDecisionEngine.AnalysisMode mode,
                                   boolean slotLocalizationReliable,
                                   String slotLocalizationMessage) {
        if (originalImage == null) throw new IllegalArgumentException("Original image must not be null.");
        BufferedImage working = workingImage != null ? workingImage : originalImage;
        if (slots == null) throw new IllegalArgumentException("Slots list must not be null.");
        final List<ParkingSlot> safeSlots = new ArrayList<>(slots);

        int imgW = originalImage.getWidth();
        int imgH = originalImage.getHeight();
        for (ParkingSlot slot : safeSlots) {
            if (slot == null) throw new IllegalArgumentException("Slot entry must not be null.");
            if (slot.getX() < 0 || slot.getY() < 0 || slot.getWidth() <= 0 || slot.getHeight() <= 0
                    || slot.getX() + slot.getWidth() > imgW || slot.getY() + slot.getHeight() > imgH) {
                throw new IllegalArgumentException(
                    String.format("Slot %d ROI (%d,%d,%dx%d) extends outside image (%dx%d).",
                        slot.getId(), slot.getX(), slot.getY(),
                        slot.getWidth(), slot.getHeight(), imgW, imgH));
            }
        }

        long startTime = System.currentTimeMillis();
        OccupancyDecisionEngine.AnalysisMode activeMode = mode != null
            ? mode
            : OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION;

        OriginalImage originalFullImage = new OriginalImage(originalImage);
        EnhancedImage enhancedFullImage = (working != originalImage)
            ? new EnhancedImage(working, "working-image", "active-working-image")
            : ImageEnhancementService.getDefault().enhance(originalFullImage);

        VehicleDetectionService.DetectionBatchResult detBatch = vehicleDetectionService != null
            ? vehicleDetectionService.detectVehicles(originalFullImage.getInternalBuffer(), sourceImageSha256, activeDetectionStrategy)
            : new VehicleDetectionService.DetectionBatchResult(
                VehicleDetectionService.DetectorStatus.MODEL_MISSING,
                "unconfigured",
                "",
                "VehicleDetectionService is null",
                List.of(),
                List.of(),
                0L,
                activeDetectionStrategy,
                0,
                false
            );

        Map<Integer, SlotOverlapAnalyzer.SlotOverlapEvidence> overlapMap = (detBatch.isAvailable() && !safeSlots.isEmpty())
            ? SlotOverlapAnalyzer.evaluateSlots(safeSlots, detBatch.filteredDetections(), overlapPolicy)
            : Map.of();

        List<VehicleDetection> unassignedDetections = detBatch.isAvailable()
            ? SlotOverlapAnalyzer.findUnassignedDetections(safeSlots, detBatch.filteredDetections(), overlapPolicy)
            : List.of();

        int treeFeatCount = (predictor != null && predictor.getModel() != null)
            ? predictor.getModel().getFeatureCount()
            : FeatureVector.FEATURE_COUNT;
        String modelVersion = String.format(Locale.ROOT, "CART-v2.3-%dfeat(depth=%d,minLeaf=%d,evalV=%d)",
            treeFeatCount, currentMaxDepth, currentMinSamples, evaluationVersion);
        String provNote = lastValidationSummary != null
            ? lastValidationSummary.formatProvenanceSummary()
            : (localTrainingDatasetLoaded
                ? "Synthetic/Demo Corpus (Real-World Generalization NOT ESTABLISHED)"
                : "Live Vehicle Detection Mode (Local features.csv not required)");

        List<SlotAnalysisResult> results = new ArrayList<>(safeSlots.size());

        for (ParkingSlot slot : safeSlots) {
            BufferedImage rawCrop = RoiExtractor.extract(originalFullImage.getInternalBuffer(), slot);
            BufferedImage enhancedCrop = RoiExtractor.extract(enhancedFullImage.getInternalBuffer(), slot);
            FeatureExtractor.ProcessedRoi processed = FeatureExtractor.processWithStages(rawCrop, enhancedCrop, useAdaptiveThreshold);
            SlotOverlapAnalyzer.SlotOverlapEvidence overlapEv = detBatch.isAvailable()
                ? overlapMap.get(slot.getId())
                : null;

            FeatureVector fv = detBatch.isAvailable()
                ? processed.getFeatures().withDetectionEvidence(overlapEv)
                : processed.getFeatures().withDetectorUnavailable(detBatch.status().name());
            DecisionTreeModel.DecisionTrace trace = predictor != null ? predictor.trace(fv) : null;

            double rawMean = FeatureExtractor.computeMeanIntensity(GrayscaleConverter.toGrayscale(rawCrop));
            boolean unobservableSpace = ((rawMean >= 0.5 && rawMean <= 4.0)
                    || (rawMean >= 251.0 && fv.getMeanIntensity() >= 251.0))
                && fv.getEdgeDensity() < 0.005;
            if (unobservableSpace && overlapEv != null && overlapEv.status() == SlotStatus.AVAILABLE) {
                overlapEv = new SlotOverlapAnalyzer.SlotOverlapEvidence(
                    slot.getId(),
                    SlotStatus.NEEDS_REVIEW,
                    overlapEv.contributingDetections(),
                    overlapEv.bestIntersectionArea(),
                    overlapEv.bestIoS(),
                    overlapEv.totalIoS(),
                    overlapEv.bestIou(),
                    overlapEv.centerInsideSlot(),
                    overlapEv.maxDetectionScore(),
                    overlapEv.multiSlotConflict(),
                    overlapEv.conflictingNeighborSlotId(),
                    overlapEv.multipleVehiclesInSlot(),
                    "Space #" + slot.getId() + " is unobservable or heavily obscured (meanIntensity="
                        + String.format(Locale.ROOT, "%.1f", fv.getMeanIntensity()) + "); cannot confirm Available.",
                    "Unobservable or obscured parking space"
                );
            }

            OccupancyDecisionEngine.CombinedSlotDecision combined =
                OccupancyDecisionEngine.decide(activeMode, trace, detBatch, overlapEv);

            SlotStatus finalStatus = combined.finalStatus();
            double leafPurity = trace != null ? trace.leaf().leafPurity() : 0.0;
            slot.setStatus(finalStatus);

            String displayDecisionPath = switch (activeMode) {
                case FEATURE_DECISION_TREE -> trace != null ? trace.compactPath() : combined.comparisonSummary();
                case VEHICLE_DETECTION -> combined.comparisonSummary();
                case COMPARISON -> combined.comparisonSummary() + (trace != null ? (" | Tree: " + trace.compactPath()) : "");
            };

            SlotDiagnosticRecord diag = SlotDiagnosticRecord.fromComponents(
                slot, fv, trace, combined, detBatch, overlapEv, modelVersion, provNote
            );

            results.add(new SlotAnalysisResult(
                slot,
                finalStatus,
                leafPurity,
                fv,
                displayDecisionPath,
                enhancedCrop,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                trace,
                diag,
                rawCrop
            ));
        }

        long duration = System.currentTimeMillis() - startTime;

        int totalCount        = safeSlots.size();
        int occupiedCount     = (int) results.stream().filter(r -> r.getStatus() == SlotStatus.OCCUPIED).count();
        int availableCount    = (int) results.stream().filter(r -> r.getStatus() == SlotStatus.AVAILABLE).count();
        int unclassifiedCount = (int) results.stream().filter(r -> r.getStatus() == SlotStatus.UNCLASSIFIED).count();
        int needsReviewCount  = (int) results.stream().filter(
            r -> r.getStatus() == SlotStatus.NEEDS_REVIEW || r.getStatus() == SlotStatus.UNCLASSIFIED).count();
        double rate = totalCount > 0 ? (occupiedCount * 100.0) / totalCount : 0.0;

        BufferedImage annotated = generateAnnotatedImage(
            working, results, detBatch.filteredDetections(), overlayFillOpacity);

        return new AnalysisSummary(
            originalImage,
            annotated,
            results,
            totalCount,
            occupiedCount,
            availableCount,
            needsReviewCount,
            unclassifiedCount,
            rate,
            duration,
            predictor != null ? predictor.getModel().toTreeString() : "LIVE_VEHICLE_DETECTION_ONLY",
            activeMode,
            detBatch,
            unassignedDetections,
            slotLocalizationReliable && !safeSlots.isEmpty(),
            slotLocalizationMessage
        );
    }

    public BufferedImage generateAnnotatedImage(BufferedImage original, List<SlotAnalysisResult> results) {
        return generateAnnotatedImage(original, results, List.of(), this.overlayFillOpacity);
    }

    public BufferedImage generateAnnotatedImage(BufferedImage original,
                                                List<SlotAnalysisResult> results,
                                                List<VehicleDetection> vehicleDetections,
                                                double fillOpacityFraction) {
        int w = original.getWidth();
        int h = original.getHeight();

        BufferedImage canvas = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = canvas.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        g.drawImage(original, 0, 0, null);

        int fillAlpha = Math.max(0, Math.min(100, (int) Math.round(fillOpacityFraction * 255.0)));

        Font badgeFont = new Font("SansSerif", Font.BOLD, 11);
        g.setFont(badgeFont);
        FontMetrics fm = g.getFontMetrics();

        for (SlotAnalysisResult res : results) {
            ParkingSlot slot = res.getSlot();
            int sx = slot.getX();
            int sy = slot.getY();
            int sw = slot.getWidth();
            int sh = slot.getHeight();

            SlotStatus st = res.getStatus();
            Color strokeColor;
            Color fillColor;
            String shortTag;
            BasicStroke borderStroke;

            if (st == SlotStatus.OCCUPIED) {
                strokeColor  = new Color(239, 68, 68, 240);
                fillColor    = new Color(239, 68, 68, fillAlpha);
                shortTag     = "OCC";
                borderStroke = new BasicStroke(2.2f);
            } else if (st == SlotStatus.AVAILABLE) {
                strokeColor  = new Color(16, 185, 129, 240);
                fillColor    = new Color(16, 185, 129, fillAlpha);
                shortTag     = "AVAIL";
                borderStroke = new BasicStroke(2.0f);
            } else if (st == SlotStatus.UNCLASSIFIED) {
                strokeColor  = new Color(148, 163, 184, 235);
                fillColor    = new Color(148, 163, 184, Math.min(90, fillAlpha));
                shortTag     = "UNCLASS";
                borderStroke = new BasicStroke(
                    2.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10.0f, new float[]{3.0f, 3.0f}, 0.0f);
            } else {
                strokeColor  = new Color(245, 158, 11, 245);
                fillColor    = new Color(245, 158, 11, Math.min(110, fillAlpha + 10));
                shortTag     = "REVIEW";
                borderStroke = new BasicStroke(
                    2.2f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10.0f, new float[]{6.0f, 4.0f}, 0.0f);
            }

            if (slot.hasCustomPolygon()) {
                List<double[]> poly = slot.getPolygonVertices();
                int[] px = new int[poly.size()];
                int[] py = new int[poly.size()];
                for (int i = 0; i < poly.size(); i++) {
                    px[i] = (int) Math.round(poly.get(i)[0]);
                    py[i] = (int) Math.round(poly.get(i)[1]);
                }
                g.setColor(fillColor);
                g.fillPolygon(px, py, poly.size());
                g.setColor(strokeColor);
                g.setStroke(borderStroke);
                g.drawPolygon(px, py, poly.size());
            } else {
                g.setColor(fillColor);
                g.fillRect(sx, sy, sw, sh);
                g.setColor(strokeColor);
                g.setStroke(borderStroke);
                g.drawRect(sx, sy, sw, sh);
            }

            String badgeText = String.format(Locale.ROOT, "#%d %s", slot.getId(), shortTag);
            int textWidth = fm.stringWidth(badgeText);
            int badgeW = textWidth + 10;
            int badgeH = 17;
            int badgeX = sx + 3;
            int badgeY = sy + 3;

            g.setColor(new Color(15, 23, 42, 230));
            g.fillRoundRect(badgeX, badgeY, badgeW, badgeH, 6, 6);
            g.setColor(strokeColor);
            g.setStroke(new BasicStroke(1.2f));
            g.drawRoundRect(badgeX, badgeY, badgeW, badgeH, 6, 6);

            g.setColor(new Color(248, 250, 252));
            g.drawString(badgeText, badgeX + 5, badgeY + badgeH - 4);
        }

        if (vehicleDetections != null && !vehicleDetections.isEmpty()) {
            BasicStroke detStroke = new BasicStroke(
                1.9f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND, 10.0f, new float[]{5.0f, 3.0f}, 0.0f);
            Color detColor = new Color(56, 189, 248, 240);
            for (VehicleDetection det : vehicleDetections) {
                int dx = (int) Math.round(det.x());
                int dy = (int) Math.round(det.y());
                int dw = (int) Math.round(det.width());
                int dh = (int) Math.round(det.height());
                g.setColor(detColor);
                g.setStroke(detStroke);
                g.drawRect(dx, dy, dw, dh);

                String lbl = String.format(Locale.ROOT, "%s:%s %.0f%%",
                    det.detectionId(), det.category().getDisplayName(), det.detectionScore() * 100.0);
                int tw = fm.stringWidth(lbl);
                int ly = Math.max(0, dy - 16);
                g.setColor(new Color(2, 132, 199, 230));
                g.fillRoundRect(dx, ly, tw + 8, 15, 4, 4);
                g.setColor(Color.WHITE);
                g.drawString(lbl, dx + 4, ly + 12);
            }
        }

        g.dispose();
        return canvas;
    }

    public void appendLabeledSampleAndRetrain(FeatureVector features, SlotStatus label) {
        appendLabeledSampleAndRetrain(
            "interactive_image", "LOT_USER_CALIBRATION", null, features, label, "LOCAL_USER", null, false);
    }

    public DatasetManifest.ManifestRecord appendLabeledSampleAndRetrain(
            String sourceImageId,
            String captureGroup,
            ParkingSlot slot,
            FeatureVector features,
            SlotStatus label,
            BufferedImage roiCrop,
            boolean saveCropImage) {
        return appendLabeledSampleAndRetrain(
            sourceImageId, captureGroup, slot, features, label, "LOCAL_USER", roiCrop, saveCropImage);
    }

    public DatasetManifest.ManifestRecord appendLabeledSampleAndRetrain(
            String sourceImageId,
            String captureGroup,
            ParkingSlot slot,
            FeatureVector features,
            SlotStatus label,
            String reviewerId,
            BufferedImage roiCrop,
            boolean saveCropImage) {
        if (frozenTestSnapshot != null && captureGroup != null && frozenTestSnapshot.testGroups().contains(captureGroup.trim())) {
            throw new IllegalStateException(
                "Capture group '" + captureGroup.trim() + "' belongs to the frozen TEST partition and cannot be added to training.");
        }
        if (lastGroupedPartitions != null && captureGroup != null
                && lastGroupedPartitions.validationGroups().contains(captureGroup.trim())) {
            throw new IllegalStateException(
                "Capture group '" + captureGroup.trim() + "' belongs to the VALIDATION partition and cannot be added to training.");
        }
        DatasetLoader.Dataset preservedTestSet = this.lastTestDataset;
        DatasetManifest.ManifestRecord rec = DatasetManifest.recordUserLabeledSample(
            activeDatasetPath, activeManifestPath, sourceImageId, captureGroup,
            slot, features, label, reviewerId, roiCrop, saveCropImage
        );
        retrainModelPreservingTestSet(activeDatasetPath, currentMaxDepth, currentMinSamples, preservedTestSet);
        return rec;
    }

    public void retrainModel(String datasetCsvPath, int maxDepth, int minSamplesPerLeaf) {
        retrainModelPreservingTestSet(datasetCsvPath, maxDepth, minSamplesPerLeaf, this.lastTestDataset);
    }

    public TestSnapshot resetPartitionsAndBumpVersion(String datasetCsvPath, int maxDepth, int minSamplesPerLeaf) {
        this.evaluationVersion++;
        this.frozenTestSnapshot = null;
        retrainModelPreservingTestSet(datasetCsvPath, maxDepth, minSamplesPerLeaf, null);
        return this.frozenTestSnapshot;
    }

    private void retrainModelPreservingTestSet(String datasetCsvPath,
                                               int maxDepth,
                                               int minSamplesPerLeaf,
                                               DatasetLoader.Dataset frozenTestSet) {
        String manifestPath = resolveManifestPathFor(datasetCsvPath);
        DatasetLoader.Dataset fullDataset = DatasetLoader.load(datasetCsvPath);
        List<DatasetManifest.ManifestRecord> manifest = DatasetManifest.loadOrSync(datasetCsvPath, manifestPath);
        DatasetManifest.verifyStrictAlignment(fullDataset, manifest);
        this.lastValidationSummary = DatasetValidator.validateOrThrow(fullDataset, manifest);
        this.manifestRecords       = manifest;

        DatasetLoader.GroupedPartitions partitions = buildPartitions(fullDataset, manifest);

        if (datasetCsvPath.equals(this.activeDatasetPath) && this.frozenTestSnapshot != null) {
            TestSnapshot diskSnapshot = computeTestSnapshot(
                this.evaluationVersion, partitions.test(), partitions.testGroups());
            if (!diskSnapshot.snapshotSha256().equals(this.frozenTestSnapshot.snapshotSha256())
                    || !diskSnapshot.testGroups().equals(this.frozenTestSnapshot.testGroups())
                    || diskSnapshot.testRowCount() != this.frozenTestSnapshot.testRowCount()) {
                throw new IllegalStateException(
                    "Protected TEST partition integrity violation: on-disk TEST partition does not match frozen TestSnapshot " +
                    "(expected hash=" + this.frozenTestSnapshot.snapshotSha256() + ", got=" + diskSnapshot.snapshotSha256() + ").");
            }
        } else {
            this.frozenTestSnapshot = computeTestSnapshot(
                this.evaluationVersion, partitions.test(), partitions.testGroups());
        }

        UncertaintyPolicy prevPolicy = this.predictor != null
            ? this.predictor.getUncertaintyPolicy()
            : UncertaintyPolicy.defaultProvisionalPolicy();

        DecisionTreeTrainer trainer = new DecisionTreeTrainer(maxDepth, minSamplesPerLeaf);
        DecisionTreeModel model = trainer.train(partitions.train().features, partitions.train().labels);
        this.predictor = new DecisionTreePredictor(model, prevPolicy);
        this.localTrainingDatasetLoaded = true;
        this.lastTrainDataset      = partitions.train();
        this.lastValidationDataset = partitions.validation();
        this.lastTestDataset       = (frozenTestSet != null && datasetCsvPath.equals(this.activeDatasetPath))
            ? frozenTestSet
            : partitions.test();
        this.lastGroupedPartitions = partitions;
        this.activeDatasetPath     = datasetCsvPath;
        this.activeManifestPath    = manifestPath;
        this.currentMaxDepth       = maxDepth;
        this.currentMinSamples     = minSamplesPerLeaf;
    }

    public ModelEvaluator.EvaluationReport evaluateModel(String datasetCsvPath) {
        if (datasetCsvPath.equals(this.activeDatasetPath) && this.lastTestDataset != null) {
            ModelEvaluator.EvaluationScope scope = (lastGroupedPartitions != null && lastGroupedPartitions.groupDisjoint())
                ? ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION
                : ModelEvaluator.EvaluationScope.RANDOM_STRATIFIED_HOLDOUT;
            String provNote = lastValidationSummary != null
                ? lastValidationSummary.formatProvenanceSummary()
                : "Synthetic/Demo Calibration Set";
            return ModelEvaluator.evaluate(predictor, lastTestDataset.features, lastTestDataset.labels, scope, provNote);
        }

        String manifestPath = resolveManifestPathFor(datasetCsvPath);
        DatasetLoader.Dataset fullDataset = DatasetLoader.load(datasetCsvPath);
        List<DatasetManifest.ManifestRecord> manifest = DatasetManifest.loadOrSync(datasetCsvPath, manifestPath);
        DatasetManifest.verifyStrictAlignment(fullDataset, manifest);
        DatasetValidator.ValidationSummary valSummary = DatasetValidator.validate(fullDataset, manifest);
        DatasetLoader.GroupedPartitions parts = buildPartitions(fullDataset, manifest);
        this.lastTrainDataset      = parts.train();
        this.lastValidationDataset = parts.validation();
        this.lastTestDataset       = parts.test();
        this.lastGroupedPartitions = parts;
        this.lastValidationSummary = valSummary;
        this.frozenTestSnapshot    = computeTestSnapshot(this.evaluationVersion, parts.test(), parts.testGroups());

        ModelEvaluator.EvaluationScope scope = parts.groupDisjoint()
            ? ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION
            : ModelEvaluator.EvaluationScope.RANDOM_STRATIFIED_HOLDOUT;
        return ModelEvaluator.evaluate(
            predictor, parts.test().features, parts.test().labels, scope, valSummary.formatProvenanceSummary());
    }

    public ModelEvaluator.EvaluationReport evaluateValidationPartition() {
        if (lastValidationDataset == null) {
            return evaluateModel(activeDatasetPath);
        }
        ModelEvaluator.EvaluationScope scope = (lastGroupedPartitions != null && lastGroupedPartitions.groupDisjoint())
            ? ModelEvaluator.EvaluationScope.VALIDATION_PARTITION
            : ModelEvaluator.EvaluationScope.RANDOM_STRATIFIED_VALIDATION;
        return ModelEvaluator.evaluate(
            predictor,
            lastValidationDataset.features,
            lastValidationDataset.labels,
            scope,
            lastValidationSummary != null ? lastValidationSummary.formatProvenanceSummary() : ""
        );
    }

    public ClassifierBenchmark.BenchmarkResult runClassifierBenchmark() {
        if (lastGroupedPartitions == null) {
            evaluateModel(activeDatasetPath);
        }
        return ClassifierBenchmark.compare(lastGroupedPartitions, currentMaxDepth, currentMinSamples, DEFAULT_SPLIT_SEED);
    }

    public boolean isLocalTrainingDatasetLoaded()                         { return localTrainingDatasetLoaded; }
    public int getLastTrainSize()                                         { return lastTrainDataset != null ? lastTrainDataset.size() : 0; }
    public int getLastValidationSize()                                    { return lastValidationDataset != null ? lastValidationDataset.size() : 0; }
    public int getLastTestSize()                                          { return lastTestDataset != null ? lastTestDataset.size() : 0; }
    public DatasetLoader.Dataset getLastTestDataset()                     { return lastTestDataset; }
    public TestSnapshot getFrozenTestSnapshot()                           { return frozenTestSnapshot; }
    public PkLotTrainingSummary getLastPkLotTrainingSummary()             { return lastPkLotTrainingSummary; }
    public int getEvaluationVersion()                                     { return evaluationVersion; }
    public DatasetLoader.GroupedPartitions getLastGroupedPartitions()     { return lastGroupedPartitions; }
    public DatasetValidator.ValidationSummary getLastValidationSummary()  { return lastValidationSummary; }
    public List<DatasetManifest.ManifestRecord> getManifestRecords()      { return manifestRecords; }
    public boolean isUseAdaptiveThreshold()                               { return useAdaptiveThreshold; }
    public void setUseAdaptiveThreshold(boolean useAdaptiveThreshold)     { this.useAdaptiveThreshold = useAdaptiveThreshold; }
    public DetectionStrategy getActiveDetectionStrategy()                  { return activeDetectionStrategy; }
    public void setActiveDetectionStrategy(DetectionStrategy strategy)     {
        this.activeDetectionStrategy = strategy != null ? strategy : DetectionStrategy.ACCURATE;
    }
    public void clearDetectionCache() {
        if (vehicleDetectionService != null) {
            vehicleDetectionService.clearDetectionCache();
        }
    }

    public OccupancyDecisionEngine.AnalysisMode getAnalysisMode()         { return analysisMode; }

    public void setAnalysisMode(OccupancyDecisionEngine.AnalysisMode analysisMode) {
        if (analysisMode == null) {
            throw new IllegalArgumentException("AnalysisMode must not be null.");
        }
        this.analysisMode = analysisMode;
    }

    public VehicleDetectionService getVehicleDetectionService() {
        return vehicleDetectionService;
    }

    public void setVehicleDetectionService(VehicleDetectionService vehicleDetectionService) {
        if (vehicleDetectionService == null) {
            throw new IllegalArgumentException("VehicleDetectionService must not be null.");
        }
        this.vehicleDetectionService = vehicleDetectionService;
    }

    public SlotOverlapAnalyzer.OverlapPolicy getOverlapPolicy() {
        return overlapPolicy;
    }

    public void setOverlapPolicy(SlotOverlapAnalyzer.OverlapPolicy overlapPolicy) {
        if (overlapPolicy == null) {
            throw new IllegalArgumentException("OverlapPolicy must not be null.");
        }
        this.overlapPolicy = overlapPolicy;
    }

    public double getOverlayFillOpacity() {
        return overlayFillOpacity;
    }

    public void setOverlayFillOpacity(double overlayFillOpacity) {
        if (!Double.isFinite(overlayFillOpacity) || overlayFillOpacity < 0.0 || overlayFillOpacity > 0.50) {
            throw new IllegalArgumentException("overlayFillOpacity must be in [0.0, 0.50]; got " + overlayFillOpacity);
        }
        this.overlayFillOpacity = overlayFillOpacity;
    }

    public DecisionTreePredictor getPredictor() {
        return predictor;
    }

    public void setPredictor(DecisionTreePredictor predictor) {
        if (predictor == null) throw new IllegalArgumentException("Predictor must not be null.");
        this.predictor = predictor;
    }
}
