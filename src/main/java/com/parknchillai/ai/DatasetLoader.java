package com.parknchillai.ai;

import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.SlotStatus;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Loads, validates, initializes, and persists capture-group-disjoint partitions
 * for training and evaluation (PNC-FIX-04, PNC-REM-06).
 *
 * <p>Partition stability & immutability invariants enforced in v2.2 (PNC-REM-06):
 * <ul>
 *   <li><strong>Persisted Partition Initialization</strong>: {@link #initializeAndPersistPartitions}
 *       computes the group-aware split once, validates pairwise disjointness, and persists the
 *       {@code TRAIN}, {@code VALIDATION}, and {@code TEST} assignments plus {@code splitId}
 *       to {@code manifest.csv} using the recoverable transaction writer.</li>
 *   <li><strong>No Silent In-Memory Repartitioning</strong>: {@link #loadPartitions} and
 *       {@link #buildValidatedPartitions} read only the persisted partition assignments from
 *       {@code manifest.csv}; if {@code VALIDATION} or {@code TEST} is empty (e.g. an unpartitioned
 *       all-{@code TRAIN} manifest), they fail closed rather than recalculating a split in memory.</li>
 *   <li><strong>Pairwise Disjointness & Complete Coverage</strong>: Verifies
 *       {@code TRAIN ∩ VALIDATION = ∅}, {@code TRAIN ∩ TEST = ∅}, and {@code VALIDATION ∩ TEST = ∅}
 *       and that every row index {@code 0..N-1} is assigned to exactly one partition.</li>
 * </ul>
 */
public final class DatasetLoader {

    private DatasetLoader() { /* utility class */ }

    /**
     * Holds Training, Validation, and Held-Out Test partitions along with capture-group
     * assignments, explicit {@code groupDisjoint} status, and the deterministic {@code splitId}.
     */
    public record GroupedPartitions(
            Dataset train,
            Dataset validation,
            Dataset test,
            Set<String> trainGroups,
            Set<String> validationGroups,
            Set<String> testGroups,
            boolean groupDisjoint,
            String splitId
    ) {
        public static final String INSUFFICIENT_GROUPS_LABEL = "INSUFFICIENT_GROUPS / NON_INDEPENDENT";

        public boolean hasSufficientIndependentGroups() {
            return groupDisjoint && trainGroups != null && !trainGroups.isEmpty()
                && validationGroups != null && !validationGroups.isEmpty()
                && testGroups != null && !testGroups.isEmpty()
                && (trainGroups.size() + validationGroups.size() + testGroups.size() >= 3);
        }

        public GroupedPartitions(
                Dataset train,
                Dataset validation,
                Dataset test,
                Set<String> trainGroups,
                Set<String> validationGroups,
                Set<String> testGroups,
                boolean groupDisjoint) {
            this(train, validation, test, trainGroups, validationGroups, testGroups, groupDisjoint,
                computeDeterministicSplitId(42L, trainGroups, validationGroups, testGroups, groupDisjoint));
        }

        /**
         * Asserts that all three pairwise capture-group intersections are empty when
         * {@code groupDisjoint == true} and that total partitioned rows equal {@code expectedTotalRows}.
         */
        public void assertValidInvariants(int expectedTotalRows) {
            if (train == null || validation == null || test == null) {
                throw new IllegalStateException("Partition datasets must not be null.");
            }
            int sumRows = train.size() + validation.size() + test.size();
            if (sumRows != expectedTotalRows) {
                throw new IllegalStateException(String.format(Locale.ROOT,
                    "Partition row assignment mismatch: train(%d) + validation(%d) + test(%d) = %d != expected %d.",
                    train.size(), validation.size(), test.size(), sumRows, expectedTotalRows));
            }
            if (groupDisjoint) {
                if (trainGroups == null || validationGroups == null || testGroups == null
                        || trainGroups.isEmpty() || validationGroups.isEmpty() || testGroups.isEmpty()) {
                    throw new IllegalStateException("Group-disjoint partitions must have non-empty group sets for TRAIN, VALIDATION, and TEST.");
                }
                if (!Collections.disjoint(trainGroups, validationGroups)) {
                    throw new IllegalStateException("Partition leakage: TRAIN ∩ VALIDATION capture groups are not disjoint.");
                }
                if (!Collections.disjoint(trainGroups, testGroups)) {
                    throw new IllegalStateException("Partition leakage: TRAIN ∩ TEST capture groups are not disjoint.");
                }
                if (!Collections.disjoint(validationGroups, testGroups)) {
                    throw new IllegalStateException("Partition leakage: VALIDATION ∩ TEST capture groups are not disjoint.");
                }
            }
        }
    }

    /**
     * Result container holding the feature matrix and matching label array.
     */
    public static final class Dataset {
        public final double[][]   features;
        public final SlotStatus[] labels;

        public Dataset(double[][] features, SlotStatus[] labels) {
            this.features = features;
            this.labels   = labels;
        }

        public int size() { return labels.length; }

        public int featureCount() {
            return (features.length > 0 && features[0] != null) ? features[0].length : 0;
        }

        public int countClass(SlotStatus target) {
            int c = 0;
            for (SlotStatus s : labels) if (s == target) c++;
            return c;
        }

        public boolean hasBothClasses() {
            return countClass(SlotStatus.OCCUPIED) > 0 && countClass(SlotStatus.AVAILABLE) > 0;
        }

        public Dataset[] trainTestSplit(double trainFraction) {
            return trainTestSplit(trainFraction, 42L);
        }

        public Dataset[] trainTestSplit(double trainFraction, long seed) {
            if (trainFraction <= 0 || trainFraction >= 1 || !Double.isFinite(trainFraction)) {
                throw new IllegalArgumentException("trainFraction must be in (0,1).");
            }
            int total = size();
            int trainSize = (int) Math.round(total * trainFraction);
            if (trainSize <= 0 || trainSize >= total) {
                throw new IllegalArgumentException("Dataset too small (" + total + ") for split fraction " + trainFraction);
            }
            int testSize = total - trainSize;

            List<Integer> occIndices = new ArrayList<>();
            List<Integer> availIndices = new ArrayList<>();
            for (int i = 0; i < total; i++) {
                if (labels[i] == SlotStatus.OCCUPIED) {
                    occIndices.add(i);
                } else {
                    availIndices.add(i);
                }
            }

            Random rng = new Random(seed);
            Collections.shuffle(occIndices, rng);
            Collections.shuffle(availIndices, rng);

            int trainOcc = (int) Math.round(occIndices.size() * trainFraction);
            trainOcc = Math.max(0, Math.min(occIndices.size(), Math.min(trainSize, trainOcc)));
            int trainAvail = trainSize - trainOcc;
            if (trainAvail > availIndices.size()) {
                trainAvail = availIndices.size();
                trainOcc = trainSize - trainAvail;
            }

            List<Integer> trainIdx = new ArrayList<>(trainSize);
            List<Integer> testIdx  = new ArrayList<>(testSize);

            trainIdx.addAll(occIndices.subList(0, trainOcc));
            testIdx.addAll(occIndices.subList(trainOcc, occIndices.size()));

            trainIdx.addAll(availIndices.subList(0, trainAvail));
            testIdx.addAll(availIndices.subList(trainAvail, availIndices.size()));

            Collections.shuffle(trainIdx, new Random(seed ^ 0x5DEECE66DL));
            Collections.shuffle(testIdx,  new Random(seed ^ 0xB504F333L));

            return new Dataset[]{ subset(trainIdx), subset(testIdx) };
        }

        public Dataset subset(List<Integer> indices) {
            double[][] subF = new double[indices.size()][];
            SlotStatus[] subL = new SlotStatus[indices.size()];
            for (int i = 0; i < indices.size(); i++) {
                int idx = indices.get(i);
                subF[i] = features[idx].clone();
                subL[i] = labels[idx];
            }
            return new Dataset(subF, subL);
        }
    }

    /**
     * Explicitly initializes and persists group-disjoint {@code TRAIN}, {@code VALIDATION},
     * and {@code TEST} partition assignments into {@code manifestPath} using the recoverable
     * transaction path (PNC-REM-06).
     */
    public static GroupedPartitions initializeAndPersistPartitions(String featuresCsvPath,
                                                                   String manifestPath,
                                                                   long seed) {
        Dataset fullDataset = load(featuresCsvPath);
        List<DatasetManifest.ManifestRecord> manifest = DatasetManifest.loadOrSync(featuresCsvPath, manifestPath);
        DatasetManifest.verifyStrictAlignment(fullDataset, manifest);

        List<String> captureGroups = new ArrayList<>(manifest.size());
        for (int i = 0; i < manifest.size(); i++) {
            String grp = manifest.get(i).captureGroup();
            if (grp == null || grp.isBlank()
                    || "UNKNOWN".equalsIgnoreCase(grp.trim())
                    || "DEFAULT_GROUP".equalsIgnoreCase(grp.trim())
                    || "LEGACY_UNGROUPED".equalsIgnoreCase(grp.trim())) {
                throw new IllegalStateException(
                    "Cannot initialize group-disjoint partitions: row " + i + " ("
                    + manifest.get(i).sampleId() + ") has invalid/ungrouped captureGroup '" + grp + "'.");
            }
            captureGroups.add(grp.trim());
        }

        Set<String> distinct = new LinkedHashSet<>(captureGroups);
        if (distinct.size() < 3) {
            throw new IllegalStateException(
                "Cannot initialize group-disjoint partitions: requires at least 3 distinct capture groups, got "
                + distinct.size() + ".");
        }

        GroupedPartitions computed = groupAwareSplit(fullDataset, captureGroups, seed);
        if (!computed.groupDisjoint()) {
            throw new IllegalStateException("Calculated split is not group-disjoint.");
        }

        List<DatasetManifest.ManifestRecord> updatedManifest = new ArrayList<>(manifest.size());
        for (DatasetManifest.ManifestRecord r : manifest) {
            String grp = r.captureGroup().trim();
            String assignedPart;
            if (computed.testGroups().contains(grp)) {
                assignedPart = "TEST";
            } else if (computed.validationGroups().contains(grp)) {
                assignedPart = "VALIDATION";
            } else {
                assignedPart = "TRAIN";
            }
            updatedManifest.add(new DatasetManifest.ManifestRecord(
                r.sampleId(), r.sourceImageId(), r.captureGroup(),
                r.slotId(), r.x(), r.y(), r.width(), r.height(),
                r.features(), r.label(), r.schemaVersion(),
                r.reviewStatus(), r.provenance(), assignedPart, r.timestamp()
            ));
        }

        String splitId = computeDeterministicSplitId(
            seed, computed.trainGroups(), computed.validationGroups(), computed.testGroups(), true);
        String splitComment = String.format(Locale.ROOT,
            "SplitMetadata: splitId=%s,seed=%d,protocol=GROUP_DISJOINT_V1", splitId, seed);

        DatasetManifest.commitDatasetAndManifestTransaction(
            featuresCsvPath, manifestPath, fullDataset, updatedManifest,
            "PARTITION_INIT", null, false, splitComment
        );

        return loadPartitions(featuresCsvPath, manifestPath);
    }

    /**
     * Loads the persisted {@code TRAIN}, {@code VALIDATION}, and {@code TEST} partitions
     * from {@code featuresCsvPath} and {@code manifestPath} without recalculating them (PNC-REM-06).
     */
    public static GroupedPartitions loadPartitions(String featuresCsvPath, String manifestPath) {
        Dataset fullDataset = load(featuresCsvPath);
        List<DatasetManifest.ManifestRecord> manifest = DatasetManifest.loadOrSync(featuresCsvPath, manifestPath);
        String meta = DatasetManifest.readPersistedSplitMetadata(manifestPath);
        return buildValidatedPartitions(fullDataset, manifest, 42L, meta);
    }

    /**
     * Reusable partition validator (PNC-FIX-04, PNC-REM-06).
     */
    public static void validatePartitions(Dataset fullDataset,
                                          List<DatasetManifest.ManifestRecord> manifest,
                                          GroupedPartitions partitions) {
        if (fullDataset == null || fullDataset.size() == 0) {
            throw new IllegalArgumentException("Full dataset must not be null or empty.");
        }
        if (manifest == null || manifest.size() != fullDataset.size()) {
            throw new IllegalStateException(
                "Every dataset row must have an aligned manifest record before partitioning (expected "
                + fullDataset.size() + ", got " + (manifest == null ? 0 : manifest.size()) + ").");
        }
        DatasetManifest.verifyStrictAlignment(fullDataset, manifest);

        if (partitions == null) {
            throw new IllegalArgumentException("GroupedPartitions must not be null.");
        }
        partitions.assertValidInvariants(fullDataset.size());

        Map<String, String> groupToPartition = new HashMap<>();
        for (int i = 0; i < manifest.size(); i++) {
            DatasetManifest.ManifestRecord r = manifest.get(i);
            String grp = r.captureGroup();
            if (grp == null || grp.isBlank() || "UNKNOWN".equalsIgnoreCase(grp)
                    || "DEFAULT_GROUP".equalsIgnoreCase(grp) || "LEGACY_UNGROUPED".equalsIgnoreCase(grp)) {
                throw new IllegalStateException(
                    "Row " + i + " (" + r.sampleId() + ") has missing or ungrouped captureGroup '" + grp + "'.");
            }
            String part = r.partition().toUpperCase(Locale.ROOT);
            String prevPart = groupToPartition.putIfAbsent(grp, part);
            if (prevPart != null && !prevPart.equals(part) && partitions.groupDisjoint()) {
                throw new IllegalStateException(String.format(Locale.ROOT,
                    "Capture group '%s' is split across multiple partitions (%s and %s).",
                    grp, prevPart, part));
            }
        }
    }

    /**
     * Builds and validates partitions strictly from persisted manifest assignments (PNC-FIX-04, PNC-REM-06).
     * Never silently recalculates a group split in memory if {@code VALIDATION} or {@code TEST} is empty.
     */
    public static GroupedPartitions buildValidatedPartitions(Dataset fullDataset,
                                                             List<DatasetManifest.ManifestRecord> manifest,
                                                             long seed) {
        return buildValidatedPartitions(fullDataset, manifest, seed, "");
    }

    private static GroupedPartitions buildValidatedPartitions(Dataset fullDataset,
                                                              List<DatasetManifest.ManifestRecord> manifest,
                                                              long seed,
                                                              String persistedSplitMeta) {
        if (fullDataset == null || fullDataset.size() == 0) {
            throw new IllegalArgumentException("Dataset must not be null or empty.");
        }
        if (manifest == null || manifest.size() != fullDataset.size()) {
            throw new IllegalStateException(
                "Manifest record count (" + (manifest == null ? 0 : manifest.size())
                + ") must match dataset row count (" + fullDataset.size() + ").");
        }
        DatasetManifest.verifyStrictAlignment(fullDataset, manifest);

        List<Integer> trainIdx = new ArrayList<>();
        List<Integer> valIdx   = new ArrayList<>();
        List<Integer> testIdx  = new ArrayList<>();
        Set<String> trainGrps  = new LinkedHashSet<>();
        Set<String> valGrps    = new LinkedHashSet<>();
        Set<String> testGrps   = new LinkedHashSet<>();

        for (int i = 0; i < manifest.size(); i++) {
            DatasetManifest.ManifestRecord r = manifest.get(i);
            String grp = r.captureGroup() != null ? r.captureGroup().trim() : "";
            if (grp.isEmpty() || "UNKNOWN".equalsIgnoreCase(grp)
                    || "DEFAULT_GROUP".equalsIgnoreCase(grp)
                    || "LEGACY_UNGROUPED".equalsIgnoreCase(grp)) {
                throw new IllegalStateException(
                    "Row " + i + " (" + r.sampleId() + ") has missing or ungrouped captureGroup '" + grp + "'.");
            }
            String part = r.partition() != null ? r.partition().trim().toUpperCase(Locale.ROOT) : "";
            if ("TEST".equals(part)) {
                testIdx.add(i);
                testGrps.add(grp);
            } else if ("VALIDATION".equals(part)) {
                valIdx.add(i);
                valGrps.add(grp);
            } else if ("TRAIN".equals(part)) {
                trainIdx.add(i);
                trainGrps.add(grp);
            } else {
                throw new IllegalStateException("Row " + i + " has invalid partition '" + part + "'.");
            }
        }

        // Enforce persisted 3-way partition assignment (never silently re-split an all-TRAIN manifest in memory!)
        if (trainIdx.isEmpty() || valIdx.isEmpty() || testIdx.isEmpty()) {
            throw new IllegalStateException(String.format(Locale.ROOT,
                "Manifest does not contain persisted 3-way partition assignments (TRAIN=%d, VALIDATION=%d, TEST=%d). " +
                "Refusing to recalculate partitions in memory; call DatasetLoader.initializeAndPersistPartitions() first.",
                trainIdx.size(), valIdx.size(), testIdx.size()));
        }

        if (!Collections.disjoint(trainGrps, valGrps)) {
            throw new IllegalStateException("Manifest partition violation: TRAIN and VALIDATION share capture group(s).");
        }
        if (!Collections.disjoint(trainGrps, testGrps)) {
            throw new IllegalStateException("Manifest partition violation: TRAIN and TEST share capture group(s).");
        }
        if (!Collections.disjoint(valGrps, testGrps)) {
            throw new IllegalStateException("Manifest partition violation: VALIDATION and TEST share capture group(s).");
        }

        String splitId = extractSplitIdFromMeta(persistedSplitMeta);
        if (splitId.isEmpty()) {
            splitId = computeDeterministicSplitId(seed, trainGrps, valGrps, testGrps, true);
        }

        GroupedPartitions gp = new GroupedPartitions(
            fullDataset.subset(trainIdx),
            fullDataset.subset(valIdx),
            fullDataset.subset(testIdx),
            Collections.unmodifiableSet(trainGrps),
            Collections.unmodifiableSet(valGrps),
            Collections.unmodifiableSet(testGrps),
            true,
            splitId
        );
        gp.assertValidInvariants(fullDataset.size());
        return gp;
    }

    /**
     * Partitions a dataset by capture group into disjoint Training, Validation, and Test sets
     * so that samples from the same capture group never appear in more than one partition.
     *
     * <p>If fewer than 3 distinct capture groups exist, falls back to a 3-way sample-level
     * partition and explicitly sets {@code groupDisjoint = false}.
     */
    public static GroupedPartitions groupAwareSplit(Dataset dataset, List<String> captureGroups, long seed) {
        if (dataset == null || dataset.size() == 0) {
            throw new IllegalArgumentException("Dataset must not be null or empty.");
        }
        if (captureGroups == null || captureGroups.size() != dataset.size()) {
            throw new IllegalArgumentException("captureGroups size must match dataset size.");
        }

        Map<String, List<Integer>> groupToIndices = new LinkedHashMap<>();
        for (int i = 0; i < dataset.size(); i++) {
            String rawGrp = captureGroups.get(i);
            if (rawGrp == null || rawGrp.isBlank()
                    || "UNKNOWN".equalsIgnoreCase(rawGrp.trim())
                    || "DEFAULT_GROUP".equalsIgnoreCase(rawGrp.trim())
                    || "LEGACY_UNGROUPED".equalsIgnoreCase(rawGrp.trim())) {
                throw new IllegalArgumentException(
                    "Capture group at row " + i + " must not be null, blank, UNKNOWN, DEFAULT_GROUP, or LEGACY_UNGROUPED.");
            }
            String grp = rawGrp.trim();
            groupToIndices.computeIfAbsent(grp, k -> new ArrayList<>()).add(i);
        }

        List<String> distinctGroups = new ArrayList<>(groupToIndices.keySet());
        if (distinctGroups.size() < 3) {
            // WS-7 requirement: Groups (capture groups / source images) must NEVER be split across TRAIN, VALIDATION and TEST.
            // When fewer than 3 independent groups exist, independent held-out evaluation cannot proceed.
            // Assign full groups without splitting, and explicitly label as INSUFFICIENT_GROUPS / NON_INDEPENDENT.
            List<Integer> trIdx = new ArrayList<>();
            List<Integer> vaIdx = new ArrayList<>();
            Set<String> trGrps  = new LinkedHashSet<>();
            Set<String> vaGrps  = new LinkedHashSet<>();

            if (distinctGroups.size() == 2) {
                // Group 0 to train, Group 1 to validation, 0 to test
                String g0 = distinctGroups.get(0);
                String g1 = distinctGroups.get(1);
                trIdx.addAll(groupToIndices.get(g0));
                vaIdx.addAll(groupToIndices.get(g1));
                trGrps.add(g0);
                vaGrps.add(g1);
            } else if (distinctGroups.size() == 1) {
                // Single group: all samples to train
                String g0 = distinctGroups.get(0);
                trIdx.addAll(groupToIndices.get(g0));
                trGrps.add(g0);
            }

            Dataset trainDs = dataset.subset(trIdx);
            Dataset valDs   = dataset.subset(vaIdx);
            Dataset testDs  = dataset.subset(Collections.emptyList());

            GroupedPartitions fallback = new GroupedPartitions(
                trainDs,
                valDs,
                testDs,
                Collections.unmodifiableSet(trGrps),
                Collections.unmodifiableSet(vaGrps),
                Collections.emptySet(),
                false,
                GroupedPartitions.INSUFFICIENT_GROUPS_LABEL
            );
            fallback.assertValidInvariants(dataset.size());
            return fallback;
        }

        List<String> occGroups   = new ArrayList<>();
        List<String> availGroups = new ArrayList<>();
        List<String> mixedGroups = new ArrayList<>();

        for (String g : distinctGroups) {
            List<Integer> idxs = groupToIndices.get(g);
            int occ = 0, avail = 0;
            for (int idx : idxs) {
                if (dataset.labels[idx] == SlotStatus.OCCUPIED) occ++;
                else avail++;
            }
            if (occ > 0 && avail > 0) mixedGroups.add(g);
            else if (occ > 0) occGroups.add(g);
            else availGroups.add(g);
        }

        Random rng = new Random(seed);
        Collections.shuffle(occGroups, rng);
        Collections.shuffle(availGroups, rng);
        Collections.shuffle(mixedGroups, rng);

        Set<String> trainGroups = new LinkedHashSet<>();
        Set<String> valGroups   = new LinkedHashSet<>();
        Set<String> testGroups  = new LinkedHashSet<>();

        if (occGroups.size() >= 3 && availGroups.size() >= 3) {
            testGroups.add(occGroups.get(0));
            testGroups.add(availGroups.get(0));
            valGroups.add(occGroups.get(1));
            valGroups.add(availGroups.get(1));
            trainGroups.addAll(occGroups.subList(2, occGroups.size()));
            trainGroups.addAll(availGroups.subList(2, availGroups.size()));
            trainGroups.addAll(mixedGroups);
        } else {
            List<String> shuffled = new ArrayList<>(distinctGroups);
            Collections.shuffle(shuffled, new Random(seed));
            int nGroups = shuffled.size();
            int testGroupCount = Math.max(1, (int) Math.round(nGroups * 0.20));
            int valGroupCount  = Math.max(1, (int) Math.round(nGroups * 0.20));
            if (testGroupCount + valGroupCount >= nGroups) {
                testGroupCount = 1;
                valGroupCount  = 1;
            }
            int trainGroupCount = nGroups - testGroupCount - valGroupCount;
            trainGroups.addAll(shuffled.subList(0, trainGroupCount));
            valGroups.addAll(shuffled.subList(trainGroupCount, trainGroupCount + valGroupCount));
            testGroups.addAll(shuffled.subList(trainGroupCount + valGroupCount, nGroups));
        }

        List<Integer> trainIdx = new ArrayList<>();
        List<Integer> valIdx   = new ArrayList<>();
        List<Integer> testIdx  = new ArrayList<>();

        for (String g : trainGroups) trainIdx.addAll(groupToIndices.get(g));
        for (String g : valGroups)   valIdx.addAll(groupToIndices.get(g));
        for (String g : testGroups)  testIdx.addAll(groupToIndices.get(g));

        String splitId = computeDeterministicSplitId(seed, trainGroups, valGroups, testGroups, true);
        GroupedPartitions result = new GroupedPartitions(
            dataset.subset(trainIdx),
            dataset.subset(valIdx),
            dataset.subset(testIdx),
            Collections.unmodifiableSet(trainGroups),
            Collections.unmodifiableSet(valGroups),
            Collections.unmodifiableSet(testGroups),
            true,
            splitId
        );
        result.assertValidInvariants(dataset.size());
        return result;
    }

    private static String computeDeterministicSplitId(long seed,
                                                      Set<String> trainGroups,
                                                      Set<String> valGroups,
                                                      Set<String> testGroups,
                                                      boolean groupDisjoint) {
        List<String> valSorted  = new ArrayList<>(valGroups != null ? valGroups : Set.of());
        List<String> testSorted = new ArrayList<>(testGroups != null ? testGroups : Set.of());
        Collections.sort(valSorted);
        Collections.sort(testSorted);
        String canonical = "seed=" + seed + "|disjoint=" + groupDisjoint
            + "|VAL=" + String.join(",", valSorted)
            + "|TEST=" + String.join(",", testSorted);
        String hex12 = DatasetManifest.shortSha256Hex(canonical).substring(0, 12);
        return (groupDisjoint ? "SPLIT-GD-" : "SPLIT-FB-") + hex12;
    }

    private static String extractSplitIdFromMeta(String meta) {
        if (meta == null || meta.isBlank()) return "";
        int idx = meta.indexOf("splitId=");
        if (idx < 0) return "";
        int start = idx + 8;
        int end = meta.indexOf(',', start);
        if (end < 0) end = meta.length();
        return meta.substring(start, end).trim();
    }

    /**
     * Reads a feature CSV file (either 8-feature v2.0+ or 5-feature legacy schema) and returns a {@link Dataset}.
     */
    public static Dataset load(String csvPath) {
        if (csvPath == null || csvPath.isBlank()) {
            throw new IllegalArgumentException("Dataset CSV path must not be null or blank.");
        }
        List<double[]>   featureRows = new ArrayList<>();
        List<SlotStatus> labelList   = new ArrayList<>();
        int expectedFeatureCount     = -1;

        try (BufferedReader reader = Files.newBufferedReader(Path.of(csvPath), StandardCharsets.UTF_8)) {
            String line;
            int lineNum = 0;
            boolean headerChecked = false;
            while ((line = reader.readLine()) != null) {
                lineNum++;
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;

                if (!headerChecked) {
                    headerChecked = true;
                    String lower = line.toLowerCase(Locale.ROOT);
                    if (lower.startsWith("mean") || lower.contains("label") || lower.contains("foreground")) {
                        continue;
                    }
                }

                String[] parts = line.split(",");
                int numFeatures = parts.length - 1;
                if (numFeatures != FeatureVector.FEATURE_COUNT
                        && numFeatures != FeatureVector.LEGACY_FEATURE_COUNT && numFeatures != FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT) {
                    throw new IllegalArgumentException(
                        "CSV line " + lineNum + ": expected "
                            + (FeatureVector.FEATURE_COUNT + 1) + " or "
                            + (FeatureVector.LEGACY_FEATURE_COUNT + 1)
                            + " fields; got " + parts.length + " in '" + line + "'");
                }
                if (expectedFeatureCount == -1) {
                    expectedFeatureCount = numFeatures;
                } else if (numFeatures != expectedFeatureCount) {
                    throw new IllegalArgumentException(
                        "CSV line " + lineNum + ": inconsistent feature count (expected "
                            + expectedFeatureCount + ", got " + numFeatures + ")");
                }

                double[] fv = new double[expectedFeatureCount];
                try {
                    for (int i = 0; i < expectedFeatureCount; i++) {
                        double val = Double.parseDouble(parts[i].trim());
                        if (!Double.isFinite(val)) {
                            throw new IllegalArgumentException(
                                "CSV line " + lineNum + ": feature " + i + " must be finite (not NaN/Infinity) in '" + line + "'");
                        }
                        fv[i] = val;
                    }
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                        "CSV line " + lineNum + ": non-numeric feature value in '" + line + "'", e);
                }
                SlotStatus label = SlotStatus.fromString(parts[expectedFeatureCount]);
                if (label == SlotStatus.NEEDS_REVIEW) {
                    throw new IllegalArgumentException(
                        "CSV line " + lineNum + ": NEEDS_REVIEW is not a valid ground-truth training label.");
                }
                featureRows.add(fv);
                labelList.add(label);
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Cannot read dataset CSV: " + csvPath, e);
        }

        if (featureRows.isEmpty()) {
            throw new IllegalArgumentException("Dataset CSV is empty or contains no data rows: " + csvPath);
        }

        return new Dataset(
            featureRows.toArray(new double[0][]),
            labelList.toArray(new SlotStatus[0])
        );
    }

    /**
     * Standalone single-file CSV helper retained for legacy unit tests that operate on a standalone
     * temporary CSV without a companion manifest. Production active-learning workflows use
     * {@link DatasetManifest#recordUserLabeledSample} so {@code features.csv}, {@code manifest.csv},
     * and optional crops are updated in a single aligned transaction.
     */
    public static void appendSample(String csvPath, FeatureVector features, SlotStatus label) {
        if (csvPath == null || csvPath.isBlank()) {
            throw new IllegalArgumentException("Dataset CSV path must not be null or blank.");
        }
        if (features == null) {
            throw new IllegalArgumentException("FeatureVector must not be null.");
        }
        if (label == null || label == SlotStatus.NEEDS_REVIEW) {
            throw new IllegalArgumentException("Ground-truth label must be OCCUPIED or AVAILABLE.");
        }

        Path path = Path.of(csvPath);
        int targetFeatureCount = FeatureVector.FEATURE_COUNT;
        boolean fileExists = Files.exists(path);

        if (fileExists) {
            try {
                Dataset existing = load(csvPath);
                if (existing.featureCount() > 0) {
                    targetFeatureCount = existing.featureCount();
                }
            } catch (Exception ignored) {
                // If file is new or empty, write full 8-feature schema
            }
        }

        String featurePrefix = features.toCsvRow(targetFeatureCount) + ",";
        String fullNewLine   = featurePrefix + label.name();

        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            if (fileExists && Files.size(path) > 0) {
                List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
                boolean foundExisting = false;
                for (int i = 0; i < lines.size(); i++) {
                    String trimmed = lines.get(i).trim();
                    if (trimmed.startsWith(featurePrefix)) {
                        lines.set(i, fullNewLine);
                        foundExisting = true;
                        break;
                    }
                }
                if (foundExisting) {
                    Files.write(path, lines, StandardCharsets.UTF_8, StandardOpenOption.TRUNCATE_EXISTING);
                    return;
                }
            }

            boolean writeHeader = !fileExists || Files.size(path) == 0;
            try (BufferedWriter writer = Files.newBufferedWriter(
                    path,
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND)) {
                if (writeHeader) {
                    writer.write(FeatureVector.csvHeader());
                    writer.newLine();
                }
                writer.write(fullNewLine);
                writer.newLine();
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Cannot append sample to dataset CSV: " + csvPath, e);
        }
    }
}
