package com.parknchillai.ai;

import com.parknchillai.features.FeatureVector;
import com.parknchillai.imageprocessing.ImageEnhancer;
import com.parknchillai.imageprocessing.ProcessingPolicy;
import com.parknchillai.parking.SlotStatus;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Deterministic persistence and schema-compatibility verifier for trained
 * {@link DecisionTreeModel} classifiers (ParkNChill v2.3 Requirement 2.E &amp; 2.G).
 *
 * <p>Saves and loads CART Decision Tree models without exposing raw JSON dumps in the UI,
 * verifies SHA-256 payload integrity, and explicitly distinguishes compatible 10-feature
 * models ({@code v2.3-10feat-enhanced-v1}) from legacy 8-feature ({@code v2.2-8feat}) or 5-feature
 * ({@code v2.0-5feat}) models.
 */
public final class ModelSerializer {

    public static final String FORMAT_HEADER = "PARKNCHILL_CART_MODEL_V23";
    public static final String LEGACY_EIGHT_FEATURE_MESSAGE =
        "An older 8-feature model is available. The current workflow requires a compatible 10-feature Decision Tree. "
        + "Import compatible data and train or load the updated model.";

    private ModelSerializer() {}

    /**
     * Metadata and loaded model returned by {@link #loadModel(Path)}.
     */
    public record LoadedModelBundle(
            DecisionTreeModel model,
            String schemaVersion,
            int featureCount,
            int maxDepth,
            String datasetSource,
            String savedTimestamp,
            String payloadSha256,
            boolean compatibleTenFeatureModel,
            String statusSummary,
            String compatibilityMessage,
            String policyId,
            String policyFingerprint
    ) {
        public LoadedModelBundle(
                DecisionTreeModel model,
                String schemaVersion,
                int featureCount,
                int maxDepth,
                String datasetSource,
                String savedTimestamp,
                String payloadSha256,
                boolean compatibleTenFeatureModel,
                String statusSummary,
                String compatibilityMessage) {
            this(model, schemaVersion, featureCount, maxDepth, datasetSource, savedTimestamp,
                 payloadSha256, compatibleTenFeatureModel, statusSummary, compatibilityMessage,
                 ProcessingPolicy.CANONICAL_POLICY_ID, ProcessingPolicy.CANONICAL.fingerprint());
        }
    }

    /**
     * Saves a trained {@link DecisionTreeModel} to {@code targetPath}.
     */
    public static LoadedModelBundle saveModel(DecisionTreeModel model,
                                              Path targetPath,
                                              String datasetSource) throws IOException {
        if (model == null || model.getRoot() == null) {
            throw new IllegalArgumentException("Cannot save an null or untrained DecisionTreeModel.");
        }
        if (targetPath == null) {
            throw new IllegalArgumentException("Target path must not be null.");
        }

        int fCount = model.getFeatureCount();
        String schema = schemaForFeatureCount(fCount);
        String cleanSource = (datasetSource == null || datasetSource.isBlank())
            ? "Reviewed Slot / PKLot Training Set"
            : datasetSource.replace('\n', ' ').replace('\r', ' ').trim();
        String savedAt = Instant.now().toString();

        StringBuilder payload = new StringBuilder();
        payload.append("format=").append(FORMAT_HEADER).append('\n');
        payload.append("schemaVersion=").append(schema).append('\n');
        payload.append("policyId=").append(ProcessingPolicy.CANONICAL_POLICY_ID).append('\n');
        payload.append("policyFingerprint=").append(ProcessingPolicy.CANONICAL.fingerprint()).append('\n');
        payload.append("preprocessingVersion=").append(fCount == FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT
            ? ProcessingPolicy.CANONICAL_POLICY_ID : "legacy-none").append('\n');
        payload.append("featureCount=").append(fCount).append('\n');
        payload.append("maxDepth=").append(model.getMaxDepth()).append('\n');
        payload.append("datasetSource=").append(cleanSource).append('\n');
        payload.append("savedAt=").append(savedAt).append('\n');

        StringBuilder minSb = new StringBuilder();
        StringBuilder maxSb = new StringBuilder();
        for (int i = 0; i < fCount; i++) {
            if (i > 0) {
                minSb.append(',');
                maxSb.append(',');
            }
            minSb.append(String.format(Locale.ROOT, "%.8f", model.getTrainMin(i)));
            maxSb.append(String.format(Locale.ROOT, "%.8f", model.getTrainMax(i)));
        }
        payload.append("trainMin=").append(minSb).append('\n');
        payload.append("trainMax=").append(maxSb).append('\n');

        List<String> nodeLines = new ArrayList<>();
        serializeNodePostOrder(model.getRoot(), nodeLines);
        payload.append("nodeCount=").append(nodeLines.size()).append('\n');
        for (String nl : nodeLines) {
            payload.append(nl).append('\n');
        }

        String sha256 = computeSha256(payload.toString());
        String fullContent = payload + "sha256=" + sha256 + "\n";

        if (targetPath.getParent() != null) {
            Files.createDirectories(targetPath.getParent());
        }
        Files.writeString(targetPath, fullContent, StandardCharsets.UTF_8);

        boolean compatible = fCount == FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT
            && FeatureVector.ENHANCED_SCHEMA_VERSION.equals(schema);
        String statusSummary = String.format(Locale.ROOT,
            "Active CART Decision Tree (%s, %d features, maxDepth=%d)",
            schema, fCount, model.getMaxDepth());
        String compatMsg = compatible
            ? "Compatible 10-feature Decision Tree active and ready for occupancy analysis."
            : LEGACY_EIGHT_FEATURE_MESSAGE;

        return new LoadedModelBundle(
            model, schema, fCount, model.getMaxDepth(), cleanSource, savedAt,
            sha256, compatible, statusSummary, compatMsg,
            ProcessingPolicy.CANONICAL_POLICY_ID, ProcessingPolicy.CANONICAL.fingerprint()
        );
    }

    /**
     * Loads a model from {@code sourcePath} and strictly enforces schema and policy compatibility,
     * throwing {@link SchemaMismatchException} on any mismatch (WS-6 requirement).
     */
    public static LoadedModelBundle loadModelStrict(Path sourcePath) throws IOException {
        LoadedModelBundle bundle = loadModel(sourcePath);
        if (!bundle.compatibleTenFeatureModel()) {
            throw new SchemaMismatchException(
                ProcessingPolicy.CANONICAL_SCHEMA_ID, bundle.schemaVersion(),
                ProcessingPolicy.CANONICAL_POLICY_ID, bundle.policyId(),
                "Model failed schema/policy compatibility: " + bundle.compatibilityMessage());
        }
        return bundle;
    }

    /**
     * Loads and validates a saved {@link DecisionTreeModel} from {@code sourcePath}.
     */
    public static LoadedModelBundle loadModel(Path sourcePath) throws IOException {
        if (sourcePath == null || !Files.exists(sourcePath)) {
            throw new IllegalArgumentException("Model file does not exist: " + sourcePath);
        }
        List<String> lines = Files.readAllLines(sourcePath, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            throw new IllegalArgumentException("Model file is empty: " + sourcePath);
        }

        StringBuilder payload = new StringBuilder();
        Map<String, String> kv = new HashMap<>();
        List<String> nodeLines = new ArrayList<>();
        String recordedSha = null;

        for (String rawLine : lines) {
            String line = rawLine.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            if (line.startsWith("sha256=")) {
                recordedSha = line.substring("sha256=".length()).trim();
                break;
            }
            payload.append(rawLine).append('\n');
            if (line.startsWith("NODE|")) {
                nodeLines.add(line);
            } else {
                int eq = line.indexOf('=');
                if (eq > 0) {
                    kv.put(line.substring(0, eq).trim(), line.substring(eq + 1).trim());
                }
            }
        }

        if (!FORMAT_HEADER.equals(kv.get("format"))) {
            throw new IllegalArgumentException("Unsupported or corrupt model file format in " + sourcePath);
        }
        if (recordedSha == null || recordedSha.isBlank()) {
            throw new IllegalArgumentException("Model file is missing sha256 integrity footer: " + sourcePath);
        }
        String computedSha = computeSha256(payload.toString());
        if (!computedSha.equalsIgnoreCase(recordedSha)) {
            throw new IllegalArgumentException("Model file checksum mismatch (expected " + recordedSha + ", got " + computedSha + ").");
        }

        int fCount = Integer.parseInt(kv.getOrDefault("featureCount", "0"));
        int maxDepth = Integer.parseInt(kv.getOrDefault("maxDepth", "4"));
        String schema = kv.getOrDefault("schemaVersion", schemaForFeatureCount(fCount));
        String datasetSource = kv.getOrDefault("datasetSource", "Saved Model");
        String savedAt = kv.getOrDefault("savedAt", "");

        if (fCount != FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT
                && fCount != FeatureVector.FEATURE_COUNT
                && fCount != FeatureVector.LEGACY_FEATURE_COUNT) {
            throw new IllegalArgumentException("Unsupported feature count in model file: " + fCount);
        }

        double[] trainMin = parseDoubleArray(kv.get("trainMin"), fCount);
        double[] trainMax = parseDoubleArray(kv.get("trainMax"), fCount);

        Map<Integer, DecisionTreeModel.Node> builtNodes = new HashMap<>();
        DecisionTreeModel.Node root = null;
        for (String nl : nodeLines) {
            String[] parts = nl.split("\\|");
            int idx = Integer.parseInt(parts[1]);
            String type = parts[2];
            if ("LEAF".equals(type)) {
                SlotStatus label = SlotStatus.fromString(parts[3]);
                int occ = Integer.parseInt(parts[4]);
                int avail = Integer.parseInt(parts[5]);
                root = new DecisionTreeModel.LeafNode(label, occ, avail);
                builtNodes.put(idx, root);
            } else if ("INTERNAL".equals(type)) {
                int featIdx = Integer.parseInt(parts[3]);
                double threshold = Double.parseDouble(parts[4]);
                int leftIdx = Integer.parseInt(parts[5]);
                int rightIdx = Integer.parseInt(parts[6]);
                DecisionTreeModel.Node left = builtNodes.get(leftIdx);
                DecisionTreeModel.Node right = builtNodes.get(rightIdx);
                if (left == null || right == null) {
                    throw new IllegalArgumentException("Corrupt tree structure at node index " + idx);
                }
                root = new DecisionTreeModel.InternalNode(featIdx, threshold, left, right);
                builtNodes.put(idx, root);
            } else {
                throw new IllegalArgumentException("Unknown node type in model file: " + type);
            }
        }

        if (root == null) {
            throw new IllegalArgumentException("Model file contains no tree nodes.");
        }

        DecisionTreeModel model = new DecisionTreeModel();
        model.setRoot(root);
        model.setFeatureCount(fCount);
        model.setMaxDepth(maxDepth);
        if (trainMin != null && trainMax != null) {
            model.setTrainDomainBounds(trainMin, trainMax);
        }

        String policyId = kv.getOrDefault("policyId", kv.getOrDefault("preprocessingVersion", ""));
        String policyFingerprint = kv.getOrDefault("policyFingerprint", "");

        boolean policyMatch = policyId.isEmpty()
            || ProcessingPolicy.CANONICAL_POLICY_ID.equals(policyId)
            || ProcessingPolicy.CANONICAL.fingerprint().equals(policyFingerprint);

        boolean compatible = fCount == FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT
            && ProcessingPolicy.CANONICAL_SCHEMA_ID.equals(schema)
            && policyMatch;
        String statusSummary;
        String compatMsg;
        if (compatible) {
            statusSummary = String.format(Locale.ROOT, "Compatible 10-Feature CART Model (%s, maxDepth=%d)", schema, maxDepth);
            compatMsg = "Compatible 10-feature Decision Tree loaded and active.";
        } else if (fCount == FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT && FeatureVector.RAW_SCHEMA_VERSION.equals(schema)) {
            statusSummary = String.format(Locale.ROOT, "Incompatible 10-Feature Raw Model (%s — TREE_INCOMPATIBLE)", schema);
            compatMsg = String.format(Locale.ROOT, "Model expects schema %s; dataset provides %s (Policy: %s).",
                FeatureVector.RAW_SCHEMA_VERSION, ProcessingPolicy.CANONICAL_SCHEMA_ID, ProcessingPolicy.CANONICAL_POLICY_ID);
        } else if (fCount == FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT && !policyMatch) {
            statusSummary = String.format(Locale.ROOT, "Incompatible Policy Model (%s, policy=%s — TREE_INCOMPATIBLE)", schema, policyId);
            compatMsg = String.format(Locale.ROOT, "Model policy '%s' mismatches active canonical policy '%s'.",
                policyId, ProcessingPolicy.CANONICAL_POLICY_ID);
        } else {
            statusSummary = String.format(Locale.ROOT, "Legacy %d-Feature CART Model (%s — Upgrade Required)", fCount, schema);
            compatMsg = LEGACY_EIGHT_FEATURE_MESSAGE;
        }

        return new LoadedModelBundle(
            model, schema, fCount, maxDepth, datasetSource, savedAt,
            computedSha, compatible, statusSummary, compatMsg, policyId, policyFingerprint
        );
    }

    public static String schemaForFeatureCount(int featureCount) {
        if (featureCount == FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT) {
            return FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION;
        }
        if (featureCount == FeatureVector.FEATURE_COUNT) {
            return FeatureVector.SCHEMA_VERSION;
        }
        return "v2.0-5feat";
    }

    private static int serializeNodePostOrder(DecisionTreeModel.Node node, List<String> lines) {
        if (node instanceof DecisionTreeModel.LeafNode leaf) {
            int myIdx = lines.size();
            lines.add(String.format(Locale.ROOT, "NODE|%d|LEAF|%s|%d|%d",
                myIdx, leaf.label.name(), leaf.occupiedCount, leaf.availableCount));
            return myIdx;
        } else if (node instanceof DecisionTreeModel.InternalNode internal) {
            int leftIdx = serializeNodePostOrder(internal.left, lines);
            int rightIdx = serializeNodePostOrder(internal.right, lines);
            int myIdx = lines.size();
            lines.add(String.format(Locale.ROOT, "NODE|%d|INTERNAL|%d|%.8f|%d|%d",
                myIdx, internal.featureIndex, internal.threshold, leftIdx, rightIdx));
            return myIdx;
        }
        throw new IllegalStateException("Unexpected node subtype: " + node);
    }

    private static double[] parseDoubleArray(String csv, int expectedLen) {
        if (csv == null || csv.isBlank()) return null;
        String[] parts = csv.split(",");
        if (parts.length != expectedLen) return null;
        double[] out = new double[expectedLen];
        for (int i = 0; i < expectedLen; i++) {
            out[i] = Double.parseDouble(parts[i].trim());
            if (Double.isNaN(out[i])) return null;
        }
        return out;
    }

    public static String computeSha256(String text) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] dig = md.digest(text.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder(dig.length * 2);
            for (byte b : dig) {
                sb.append(String.format(Locale.ROOT, "%02x", b & 0xff));
            }
            return sb.toString();
        } catch (Exception e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }
}
