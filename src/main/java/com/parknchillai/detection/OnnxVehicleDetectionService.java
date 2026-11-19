package com.parknchillai.detection;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Local ONNX Runtime implementation of {@link VehicleDetectionService} with a verified
 * first-run model setup &amp; validation workflow (ParkNChill v2.3 Requirement 1).
 *
 * <p>Documented Reference Model Specification:
 * <ul>
 *   <li><strong>Primary Reference Model</strong>: YOLOX-s ONNX ({@code yolox_s.onnx})</li>
 *   <li><strong>Original Upstream URL</strong>:
 *       {@code https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx}</li>
 *   <li><strong>License</strong>: Apache License 2.0 (permissive commercial &amp; academic use)</li>
 *   <li><strong>Alternative Compatible Models</strong>: Ultralytics/Paddle RT-DETR-L ({@code rtdetr-l.onnx}, Apache-2.0)
 *       or Alibaba DAMO-YOLO-T ({@code damoyolo_tinynasL20_T.onnx}, MIT License)</li>
 *   <li><strong>Default YOLOX-s input</strong>: {@code [1, 3, 640, 640]} planar BGR {@code float32}
 *       in the {@code [0, 255]} range, aspect-preserving resize, top-left image placement, and padding value 114.
 *       This intentionally matches YOLOX's reference preprocessing rather than normalized RGB preprocessing.</li>
 *   <li><strong>Output Tensor Specification</strong>: Auto-detected across Layout A {@code [1, 4+C, 8400]}
 *       (transposed YOLOv8/RT-DETR), Layout B {@code [1, 8400, 5+C]} / {@code [1, 25200, 85]} (YOLOX/YOLOv5),
 *       and Layout C {@code [1, N, 6]} (End-to-End NMS)</li>
 *   <li><strong>Supported Class Mapping</strong>: COCO 80-class vehicle indices {@code 2 = CAR},
 *       {@code 3 = MOTORCYCLE}, {@code 5 = BUS}, {@code 7 = TRUCK} (or 4-class {@code 0..3})</li>
 * </ul>
 */
public final class OnnxVehicleDetectionService implements VehicleDetectionService, AutoCloseable {

    public static final String DEFAULT_MODEL_PATH = "models/vehicle_detector.onnx";
    public static final String LEGACY_MODEL_PATH  = "models/yolov8n_vehicles.onnx";
    public static final String MODEL_PATH_PROPERTY = "parknchill.onnx.model";

    public static final String DOCUMENTED_MODEL_URL =
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx";
    public static final String DOCUMENTED_MODEL_LICENSE =
        "Apache-2.0 License (Megvii YOLOX-s ONNX / RT-DETR) + MIT License (com.microsoft.onnxruntime:onnxruntime:1.17.1)";
    public static final String DOCUMENTED_TENSOR_SPEC =
        "Default YOLOX input: float32[1,3,640,640] CHW BGR [0..255], top-left letterbox | Output: [1,8400,85] / [1,84,8400] / [1,N,6] | Classes: COCO {2:CAR, 3:MOTORCYCLE, 5:BUS, 7:TRUCK}";
    public static final String DEFAULT_LICENSE_NOTE =
        "Requires a permissively licensed pretrained ONNX vehicle detector (e.g. Apache-2.0 YOLOX-s at "
        + DOCUMENTED_MODEL_URL + ") saved at models/vehicle_detector.onnx + com.microsoft.onnxruntime:onnxruntime:1.17.1 (MIT License)";

    /**
     * Result of running first-run ONNX model installation/verification via {@link #installAndVerifyModel(Path, Path)}.
     */
    public record ModelSetupResult(
            boolean validOnnxHeader,
            Path installedPath,
            long fileSizeBytes,
            String sha256,
            DetectorStatus resultingStatus,
            String message
    ) {}

    @FunctionalInterface
    public interface RawTensorInferenceBackend {
        float[][] runInference(float[] chwTensor, int inputWidth, int inputHeight) throws Exception;
    }

    @FunctionalInterface
    public interface TiledTensorInferenceBackend {
        float[][] runTileInference(float[] chwTensor,
                                   int inputWidth,
                                   int inputHeight,
                                   int tileOffsetX,
                                   int tileOffsetY,
                                   int tileWidth,
                                   int tileHeight) throws Exception;
    }

    private final Path modelPath;
    private final String modelName;
    private final int inputWidth;
    private final int inputHeight;
    private final double confidenceThreshold;
    private final double nmsIouThreshold;
    private final boolean enableTiledInferenceForOverheadLots;
    private final RawTensorInferenceBackend customBackend;
    private final TiledTensorInferenceBackend customTiledBackend;
    private volatile Boolean cachedYoloXInputProfile;
    private final Object sessionLock = new Object();
    private Object cachedSession = null;

    private final DetectionStrategy defaultStrategy = resolveDefaultStrategy();

    private static DetectionStrategy resolveDefaultStrategy() {
        String prop = System.getProperty("parknchill.detection.strategy", "ACCURATE").trim();
        if (prop.equalsIgnoreCase("FAST")) return DetectionStrategy.FAST;
        return DetectionStrategy.ACCURATE;
    }

    public record DetectionCacheKey(
            String imageSha256,
            String modelSha256,
            String profileId,
            int imageWidth,
            int imageHeight,
            double confidenceThreshold,
            double nmsThreshold,
            DetectionStrategy strategy
    ) {}

    private final Map<DetectionCacheKey, DetectionBatchResult> lruDetectionCache =
        Collections.synchronizedMap(new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<DetectionCacheKey, DetectionBatchResult> eldest) {
                return size() > 16;
            }
        });

    private static final AtomicInteger fullPixelHashInvocationCounter = new AtomicInteger(0);

    public static int getFullPixelHashInvocationCount() {
        return fullPixelHashInvocationCounter.get();
    }

    public static void resetFullPixelHashInvocationCount() {
        fullPixelHashInvocationCounter.set(0);
    }

    public void clearDetectionCache() {
        lruDetectionCache.clear();
    }

    public int getDetectionCacheSize() {
        return lruDetectionCache.size();
    }

    /**
     * Computes a collision-resistant SHA-256 hash across all raw decoded pixels
     * of the image, including width, height, and raster format.
     * Rows are streamed without building a full encoded PNG/JPEG in memory.
     */
    public static String computeFullPixelSha256(BufferedImage img) {
        if (img == null) return null;
        fullPixelHashInvocationCounter.incrementAndGet();
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            int w = img.getWidth();
            int h = img.getHeight();
            int type = img.getType();
            java.nio.ByteBuffer header = java.nio.ByteBuffer.allocate(16);
            header.putInt(w).putInt(h).putInt(type);
            md.update(header.array());

            int[] rgb = new int[w];
            byte[] rowBuf = new byte[w * 4];
            for (int y = 0; y < h; y++) {
                img.getRGB(0, y, w, 1, rgb, 0, w);
                int idx = 0;
                for (int p : rgb) {
                    rowBuf[idx++] = (byte) (p >>> 24);
                    rowBuf[idx++] = (byte) (p >>> 16);
                    rowBuf[idx++] = (byte) (p >>> 8);
                    rowBuf[idx++] = (byte) p;
                }
                md.update(rowBuf, 0, idx);
            }
            byte[] dig = md.digest();
            StringBuilder sb = new StringBuilder(dig.length * 2);
            for (byte b : dig) sb.append(String.format(Locale.ROOT, "%02x", b & 0xff));
            return sb.toString();
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * @deprecated Insecure sampled-row digest previously used; kept strictly for
     * collision regression testing against full-pixel hashing.
     */
    @Deprecated
    public static String deprecatedInsecureSampledRowDigest(BufferedImage img) {
        if (img == null) return "NULL_IMAGE";
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            int w = img.getWidth();
            int h = img.getHeight();
            java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(16);
            bb.putInt(w).putInt(h).putInt(img.getType());
            md.update(bb.array());

            int rowStep = Math.max(1, h / 200);
            int[] rgb = new int[w];
            byte[] rowBuf = new byte[w * 4];
            for (int y = 0; y < h; y += rowStep) {
                img.getRGB(0, y, w, 1, rgb, 0, w);
                int idx = 0;
                for (int p : rgb) {
                    rowBuf[idx++] = (byte) (p >>> 24);
                    rowBuf[idx++] = (byte) (p >>> 16);
                    rowBuf[idx++] = (byte) (p >>> 8);
                    rowBuf[idx++] = (byte) p;
                }
                md.update(rowBuf, 0, idx);
            }
            byte[] dig = md.digest();
            StringBuilder sb = new StringBuilder(dig.length * 2);
            for (byte b : dig) sb.append(String.format(Locale.ROOT, "%02x", b & 0xff));
            return sb.toString();
        } catch (Exception e) {
            return "HASH_FALLBACK_" + System.identityHashCode(img);
        }
    }

    public OnnxVehicleDetectionService() {
        this(resolveDefaultModelPath(), DetectionPreprocessor.DEFAULT_INPUT_SIZE,
            DetectionPreprocessor.DEFAULT_INPUT_SIZE,
            DetectionFilter.DEFAULT_CONFIDENCE_THRESHOLD,
            DetectionFilter.DEFAULT_NMS_IOU_THRESHOLD,
            true, null, null);
    }

    public OnnxVehicleDetectionService(Path modelPath) {
        this(modelPath, DetectionPreprocessor.DEFAULT_INPUT_SIZE,
            DetectionPreprocessor.DEFAULT_INPUT_SIZE,
            DetectionFilter.DEFAULT_CONFIDENCE_THRESHOLD,
            DetectionFilter.DEFAULT_NMS_IOU_THRESHOLD,
            true, null, null);
    }

    public OnnxVehicleDetectionService(Path modelPath,
                                       int inputWidth,
                                       int inputHeight,
                                       double confidenceThreshold,
                                       double nmsIouThreshold,
                                       RawTensorInferenceBackend customBackend) {
        this(modelPath, inputWidth, inputHeight, confidenceThreshold, nmsIouThreshold,
            false, customBackend, null);
    }

    public OnnxVehicleDetectionService(Path modelPath,
                                       int inputWidth,
                                       int inputHeight,
                                       double confidenceThreshold,
                                       double nmsIouThreshold,
                                       boolean enableTiledInferenceForOverheadLots,
                                       RawTensorInferenceBackend customBackend,
                                       TiledTensorInferenceBackend customTiledBackend) {
        this.modelPath = modelPath != null ? modelPath : Path.of(DEFAULT_MODEL_PATH);
        this.modelName = this.modelPath.getFileName() != null
            ? this.modelPath.getFileName().toString()
            : "vehicle_detector.onnx";
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.confidenceThreshold = confidenceThreshold;
        this.nmsIouThreshold = nmsIouThreshold;
        this.enableTiledInferenceForOverheadLots = enableTiledInferenceForOverheadLots;
        this.customBackend = customBackend;
        this.customTiledBackend = customTiledBackend;
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            clearDetectionCache();
            close();
        }, "OnnxSessionShutdownHook-" + this.modelName));
    }

    /**
     * First-run ONNX model setup mechanism (Requirement 1):
     * Validates a candidate {@code sourceOnnxFile}, copies it to {@code targetModelPath}
     * (default {@code models/vehicle_detector.onnx}), computes its SHA-256 checksum, and returns
     * the resulting {@link DetectorStatus}. Never fabricates a model or claims readiness if invalid.
     */
    public static ModelSetupResult installAndVerifyModel(Path sourceOnnxFile, Path targetModelPath) throws IOException {
        if (sourceOnnxFile == null || !Files.exists(sourceOnnxFile) || !Files.isRegularFile(sourceOnnxFile)) {
            throw new IllegalArgumentException("Selected ONNX model file does not exist: " + sourceOnnxFile);
        }
        if (!isValidOnnxProtobufFile(sourceOnnxFile)) {
            return new ModelSetupResult(
                false,
                targetModelPath,
                Files.size(sourceOnnxFile),
                "INVALID_ONNX_HEADER",
                DetectorStatus.MODEL_INCOMPATIBLE,
                "Rejected '" + sourceOnnxFile.getFileName() + "': file is not a valid ONNX ModelProto binary (missing wire tag 0x08 / ir_version)."
            );
        }

        Path dest = targetModelPath != null ? targetModelPath : Path.of(DEFAULT_MODEL_PATH);
        if (dest.getParent() != null) {
            Files.createDirectories(dest.getParent());
        }
        if (!sourceOnnxFile.toAbsolutePath().normalize().equals(dest.toAbsolutePath().normalize())) {
            Files.copy(sourceOnnxFile, dest, StandardCopyOption.REPLACE_EXISTING);
        }

        OnnxVehicleDetectionService probe = new OnnxVehicleDetectionService(dest);
        String sha = probe.computeModelSha256();
        DetectorStatus st = probe.getStatus();
        long size = Files.size(dest);
        String msg = String.format(Locale.ROOT,
            "Installed ONNX model '%s' (%d bytes, sha256=%s) -> Status: %s",
            dest, size, sha, st.name());
        return new ModelSetupResult(true, dest, size, sha, st, msg);
    }

    private static Path resolveDefaultModelPath() {
        String prop = System.getProperty(MODEL_PATH_PROPERTY);
        if (prop != null && !prop.isBlank()) {
            return Path.of(prop.trim());
        }
        String env = System.getenv("PARKNCHILL_ONNX_MODEL");
        if (env != null && !env.isBlank()) {
            return Path.of(env.trim());
        }
        Path preferred = Path.of(DEFAULT_MODEL_PATH);
        if (Files.exists(preferred)) {
            return preferred;
        }
        Path legacy = Path.of(LEGACY_MODEL_PATH);
        if (Files.exists(legacy)) {
            return legacy;
        }
        return preferred;
    }

    public static OnnxVehicleDetectionService createWithTestBackend(
            String modelName,
            int inputWidth,
            int inputHeight,
            double confidenceThreshold,
            double nmsIouThreshold,
            RawTensorInferenceBackend backend) {
        return createWithTestBackend(modelName, inputWidth, inputHeight, confidenceThreshold, nmsIouThreshold, false, backend);
    }

    public static OnnxVehicleDetectionService createWithTestBackend(
            String modelName,
            int inputWidth,
            int inputHeight,
            double confidenceThreshold,
            double nmsIouThreshold,
            boolean enableTiling,
            RawTensorInferenceBackend backend) {
        if (backend == null) {
            throw new IllegalArgumentException("Test inference backend must not be null.");
        }
        return new OnnxVehicleDetectionService(
            Path.of(modelName != null ? modelName : "test_vehicle_detector.onnx"),
            inputWidth, inputHeight, confidenceThreshold, nmsIouThreshold, enableTiling, backend, null
        );
    }

    public static OnnxVehicleDetectionService createWithTiledTestBackend(
            String modelName,
            int inputWidth,
            int inputHeight,
            double confidenceThreshold,
            double nmsIouThreshold,
            TiledTensorInferenceBackend tiledBackend) {
        if (tiledBackend == null) {
            throw new IllegalArgumentException("Tiled test inference backend must not be null.");
        }
        return new OnnxVehicleDetectionService(
            Path.of(modelName != null ? modelName : "test_tiled_vehicle_detector.onnx"),
            inputWidth, inputHeight, confidenceThreshold, nmsIouThreshold, true, null, tiledBackend
        );
    }

    public boolean isUsingTestBackend() {
        return customBackend != null || customTiledBackend != null;
    }

    public Path getModelPath() {
        return modelPath;
    }

    @Override
    public DetectorStatus getStatus() {
        if (customBackend != null || customTiledBackend != null) {
            return DetectorStatus.READY;
        }
        if (!Files.exists(modelPath) || !Files.isRegularFile(modelPath)) {
            return DetectorStatus.MODEL_MISSING;
        }
        if (!isValidOnnxProtobufFile(modelPath)) {
            return DetectorStatus.MODEL_INCOMPATIBLE;
        }
        if (!isOnnxRuntimePresentOnClasspath()) {
            return DetectorStatus.RUNTIME_MISSING;
        }
        return DetectorStatus.READY;
    }

    public static boolean isValidOnnxProtobufFile(Path path) {
        if (path == null || !Files.exists(path) || !Files.isRegularFile(path)) {
            return false;
        }
        try {
            long size = Files.size(path);
            if (size < 128) {
                return false;
            }
            try (InputStream in = Files.newInputStream(path)) {
                byte[] header = in.readNBytes(128);
                return header.length >= 128 && header[0] == 0x08 && header[1] > 0 && header[1] <= 20;
            }
        } catch (IOException e) {
            return false;
        }
    }

    public String computeModelSha256() {
        if (!Files.exists(modelPath) || !Files.isRegularFile(modelPath)) {
            return "MODEL_MISSING";
        }
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            try (InputStream is = Files.newInputStream(modelPath);
                 DigestInputStream dis = new DigestInputStream(is, md)) {
                byte[] buffer = new byte[65536];
                while (dis.read(buffer) != -1) {
                    // stream through to compute digest without loading full 36MB into heap
                }
            }
            byte[] digest = md.digest();
            StringBuilder sb = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                sb.append(String.format(Locale.ROOT, "%02x", b & 0xff));
            }
            return sb.toString();
        } catch (Exception e) {
            return "HASH_ERROR";
        }
    }

    @Override
    public String describeConfiguration() {
        DetectorStatus st = getStatus();
        return switch (st) {
            case READY -> String.format(Locale.ROOT,
                "Vehicle Detector READY: model='%s' (%dx%d, conf>=%.2f, NMS IoU=%.2f, tiledOverhead=%s, sha256=%s)",
                modelName, inputWidth, inputHeight, confidenceThreshold, nmsIouThreshold,
                enableTiledInferenceForOverheadLots ? "ENABLED" : "OFF", computeModelSha256());
            case MODEL_MISSING -> String.format(Locale.ROOT,
                "Vehicle Detector UNAVAILABLE (MODEL_MISSING): No ONNX vehicle model found at '%s'. "
                + "Click 'Setup ONNX Model...' or run scripts/setup_onnx_model.ps1 to download %s (%s) to '%s'.",
                modelPath, DOCUMENTED_MODEL_URL, DOCUMENTED_MODEL_LICENSE, DEFAULT_MODEL_PATH);
            case RUNTIME_MISSING -> String.format(Locale.ROOT,
                "Vehicle Detector UNAVAILABLE (RUNTIME_MISSING): Valid ONNX model '%s' (sha256=%s) found, "
                + "but 'ai.onnxruntime.OrtEnvironment' (com.microsoft.onnxruntime:onnxruntime:1.17.1) is not on the JVM classpath.",
                modelPath, computeModelSha256());
            case MODEL_INCOMPATIBLE -> String.format(Locale.ROOT,
                "Vehicle Detector UNAVAILABLE (MODEL_INCOMPATIBLE): File at '%s' is not a valid ONNX ModelProto or has an unsupported output tensor shape.",
                modelPath);
            case INFERENCE_FAILED -> String.format(Locale.ROOT,
                "Vehicle Detector UNAVAILABLE (INFERENCE_FAILED): Runtime error while executing '%s'.",
                modelPath);
        };
    }

    @Override
    public DetectionBatchResult detectVehicles(BufferedImage fullImage) {
        return detectVehicles(fullImage, null, defaultStrategy);
    }

    @Override
    public DetectionBatchResult detectVehicles(BufferedImage fullImage, DetectionStrategy strategy) {
        return detectVehicles(fullImage, null, strategy);
    }

    @Override
    public DetectionBatchResult detectVehicles(BufferedImage fullImage, String explicitImageSha256, DetectionStrategy strategy) {
        if (fullImage == null) {
            throw new IllegalArgumentException("Input image must not be null.");
        }
        long t0 = System.currentTimeMillis();
        DetectorStatus currentStatus = getStatus();
        DetectionStrategy effectiveStrategy = strategy != null ? strategy : defaultStrategy;

        if (currentStatus != DetectorStatus.READY) {
            return new DetectionBatchResult(
                currentStatus,
                modelName,
                DEFAULT_LICENSE_NOTE,
                describeConfiguration(),
                List.of(),
                List.of(),
                System.currentTimeMillis() - t0,
                effectiveStrategy,
                0,
                false
            );
        }

        boolean yoloXProfile = useYoloXInputProfile();
        String profileId = yoloXProfile ? "YOLOX_BGR_255" : "RGB_NORMALIZED";
        String imageSha = (explicitImageSha256 != null && !explicitImageSha256.isBlank())
            ? explicitImageSha256.trim()
            : computeFullPixelSha256(fullImage);

        DetectionCacheKey cacheKey = (imageSha != null)
            ? new DetectionCacheKey(
                imageSha,
                computeModelSha256(),
                profileId,
                fullImage.getWidth(),
                fullImage.getHeight(),
                confidenceThreshold,
                nmsIouThreshold,
                effectiveStrategy
            )
            : null;

        if (cacheKey != null) {
            DetectionBatchResult cached = lruDetectionCache.get(cacheKey);
            if (cached != null) {
                return cached.withCacheHit(System.currentTimeMillis() - t0);
            }
        }

        try {
            List<VehicleDetection> allRawCandidates = new ArrayList<>();
            int onnxPasses = 0;

            DetectionPreprocessor.LetterboxResult fullLb = preprocessForConfiguredModel(
                fullImage, inputWidth, inputHeight, yoloXProfile);
            float[][] fullOutput = executeSinglePass(
                fullLb.chwNormalizedTensor(), 0, 0, fullImage.getWidth(), fullImage.getHeight());
            onnxPasses++;

            allRawCandidates.addAll(yoloXProfile
                ? DetectionFilter.decodeYoloX(fullOutput, fullLb, inputWidth, inputHeight, confidenceThreshold, modelName)
                : DetectionFilter.decodeAuto(fullOutput, fullLb, confidenceThreshold, modelName));

            boolean runTiles = (effectiveStrategy == DetectionStrategy.ACCURATE)
                && enableTiledInferenceForOverheadLots
                && (customTiledBackend != null || (fullImage.getWidth() >= 600 && fullImage.getHeight() >= 300));
            if (runTiles) {
                List<DetectionPreprocessor.ImageTile> tiles = yoloXProfile
                    ? DetectionPreprocessor.generateOverlappingTilesYoloX(
                        fullImage, 2, 2, DetectionPreprocessor.DEFAULT_TILE_OVERLAP_FRACTION,
                        inputWidth, inputHeight)
                    : DetectionPreprocessor.generateOverlappingTiles(
                        fullImage, 2, 2, DetectionPreprocessor.DEFAULT_TILE_OVERLAP_FRACTION,
                        inputWidth, inputHeight);
                for (DetectionPreprocessor.ImageTile tile : tiles) {
                    float[][] tileOutput = executeSinglePass(
                        tile.letterbox().chwNormalizedTensor(),
                        tile.offsetX(), tile.offsetY(), tile.width(), tile.height()
                    );
                    onnxPasses++;
                    List<VehicleDetection> tileDecoded = yoloXProfile
                        ? DetectionFilter.decodeYoloX(tileOutput, tile.letterbox(), inputWidth, inputHeight, confidenceThreshold, modelName)
                        : DetectionFilter.decodeAuto(tileOutput, tile.letterbox(), confidenceThreshold, modelName);
                    for (VehicleDetection td : tileDecoded) {
                        double gx0 = Math.max(0.0, Math.min(fullImage.getWidth(), tile.offsetX() + td.x()));
                        double gy0 = Math.max(0.0, Math.min(fullImage.getHeight(), tile.offsetY() + td.y()));
                        double gx1 = Math.max(gx0, Math.min(fullImage.getWidth(), tile.offsetX() + td.x() + td.width()));
                        double gy1 = Math.max(gy0, Math.min(fullImage.getHeight(), tile.offsetY() + td.y() + td.height()));
                        double gw = gx1 - gx0;
                        double gh = gy1 - gy0;
                        if (gw >= DetectionFilter.MIN_BOX_SIDE_PX && gh >= DetectionFilter.MIN_BOX_SIDE_PX) {
                            String tileMeta = String.format(Locale.ROOT,
                                "pass=tile-%d,tileOffset=(%d,%d),tileSize=%dx%d,tileScale=%.3f,fullOrig=%dx%d",
                                tile.tileIndex(), tile.offsetX(), tile.offsetY(),
                                tile.width(), tile.height(), tile.letterbox().scale(),
                                fullImage.getWidth(), fullImage.getHeight());
                            allRawCandidates.add(new VehicleDetection(
                                td.detectionId() + "-T" + tile.tileIndex(),
                                td.category(), gx0, gy0, gw, gh, td.detectionScore(), modelName,
                                inputWidth, inputHeight, tileMeta
                            ));
                        }
                    }
                }
            }

            List<VehicleDetection> filteredDetections = DetectionFilter.mergeTiledDetectionsWithNms(
                allRawCandidates, confidenceThreshold, nmsIouThreshold, DetectionFilter.CROSS_TILE_IOMIN_THRESHOLD);

            long elapsed = System.currentTimeMillis() - t0;
            String inputProfile = yoloXProfile ? "YOLOX BGR/0..255, top-left letterbox" : "RGB/0..1, centered letterbox";
            String msg = filteredDetections.isEmpty()
                ? String.format(Locale.ROOT,
                    "Detector READY (%s) [%s, Passes=%d]: 0 vehicles detected in image above conf %.2f (input=%s, %d ms).",
                    modelName, effectiveStrategy.name(), onnxPasses, confidenceThreshold, inputProfile, elapsed)
                : String.format(Locale.ROOT,
                    "Detector READY (%s) [%s, Passes=%d]: Detected %d vehicle(s) after NMS (%d candidates, input=%s, %d ms).",
                    modelName, effectiveStrategy.name(), onnxPasses, filteredDetections.size(), allRawCandidates.size(), inputProfile, elapsed);

            DetectionBatchResult result = new DetectionBatchResult(
                DetectorStatus.READY,
                modelName,
                DEFAULT_LICENSE_NOTE,
                msg,
                List.copyOf(allRawCandidates),
                filteredDetections,
                elapsed,
                effectiveStrategy,
                onnxPasses,
                false
            );
            if (cacheKey != null) {
                lruDetectionCache.put(cacheKey, result);
            }
            return result;
        } catch (IllegalArgumentException iae) {
            return new DetectionBatchResult(
                DetectorStatus.MODEL_INCOMPATIBLE,
                modelName,
                DEFAULT_LICENSE_NOTE,
                "Incompatible model output tensor: " + iae.getMessage(),
                List.of(),
                List.of(),
                System.currentTimeMillis() - t0,
                effectiveStrategy,
                0,
                false
            );
        } catch (Exception ex) {
            return new DetectionBatchResult(
                DetectorStatus.INFERENCE_FAILED,
                modelName,
                DEFAULT_LICENSE_NOTE,
                "Vehicle inference failed without fallback: " + ex.getMessage(),
                List.of(),
                List.of(),
                System.currentTimeMillis() - t0,
                effectiveStrategy,
                0,
                false
            );
        }
    }

    /**
     * The bundled setup script installs YOLOX-s, whose reference preprocessing differs
     * from common YOLOv5/YOLOv8 exports. Override with -Dparknchill.onnx.inputProfile=RGB_NORMALIZED
     * when the replacement model specifically expects RGB float values in [0,1] and centered padding.
     */
    private boolean useYoloXInputProfile() {
        if (customBackend != null || customTiledBackend != null) {
            // Keep injected test backends on the original normalized convention for deterministic unit tests.
            return false;
        }
        String configured = System.getProperty("parknchill.onnx.inputProfile", "").trim();
        if (!configured.isEmpty()) {
            if (configured.equalsIgnoreCase("YOLOX_BGR_255")) return true;
            if (configured.equalsIgnoreCase("RGB_NORMALIZED")) return false;
            throw new IllegalArgumentException("Unsupported -Dparknchill.onnx.inputProfile=" + configured
                + ". Supported profiles: YOLOX_BGR_255, RGB_NORMALIZED.");
        }
        String filename = modelPath.getFileName() == null ? "" : modelPath.getFileName().toString().toLowerCase(Locale.ROOT);
        if (filename.contains("yolox")) return true;
        if (filename.contains("yolov5") || filename.contains("yolov8") || filename.contains("yolov10")
                || filename.contains("rtdetr") || filename.contains("rt-detr")) {
            return false;
        }

        // The common configured name vehicle_detector.onnx does not identify the architecture.
        // Inspect the model's declared output shape so an older YOLOv5 model copied into this path
        // is not accidentally fed YOLOX preprocessing. The official setup script downloads YOLOX-s,
        // so YOLOX remains the safe fallback if metadata cannot be inspected.
        Boolean cached = cachedYoloXInputProfile;
        if (cached != null) return cached;
        Boolean detected = detectYoloXProfileFromModelMetadata(modelPath);
        if (detected == null) detected = Boolean.TRUE;
        cachedYoloXInputProfile = detected;
        return detected;
    }

    /** Returns true for the official raw YOLOX [1,8400,85] output, false for known other layouts. */
    private static Boolean detectYoloXProfileFromModelMetadata(Path path) {
        if (!isOnnxRuntimePresentOnClasspath() || path == null || !Files.isRegularFile(path)) return null;
        Object session = null;
        try {
            Class<?> envClass = Class.forName("ai.onnxruntime.OrtEnvironment");
            Object env = envClass.getMethod("getEnvironment").invoke(null);
            session = envClass.getMethod("createSession", String.class).invoke(env, path.toAbsolutePath().toString());
            Object infoObj = session.getClass().getMethod("getOutputInfo").invoke(session);
            if (!(infoObj instanceof Map<?, ?> outputInfo) || outputInfo.isEmpty()) return null;
            Object nodeInfo = outputInfo.values().iterator().next();
            Object tensorInfo = nodeInfo.getClass().getMethod("getInfo").invoke(nodeInfo);
            Object shapeObj = tensorInfo.getClass().getMethod("getShape").invoke(tensorInfo);
            if (!(shapeObj instanceof long[] shape) || shape.length < 2) return null;
            long anchors = shape[shape.length - 2];
            long channels = shape[shape.length - 1];
            if (channels == 85 && anchors == 8400) return Boolean.TRUE;
            if ((channels == 85 && anchors == 25200)
                    || (channels == 84 && anchors == 8400)
                    || (anchors == 8400 && channels == 84)) return Boolean.FALSE;
            // Transposed YOLOv8 [1,84,N] also uses RGB-normalized preprocessing.
            if (shape.length >= 2 && shape[shape.length - 2] == 84) return Boolean.FALSE;
            return null;
        } catch (Exception ignored) {
            return null;
        } finally {
            if (session != null) {
                try { session.getClass().getMethod("close").invoke(session); } catch (Exception ignored) { }
            }
        }
    }

    private static DetectionPreprocessor.LetterboxResult preprocessForConfiguredModel(
            BufferedImage image, int width, int height, boolean yoloXProfile) {
        return yoloXProfile
            ? DetectionPreprocessor.preprocessYoloX(image, width, height)
            : DetectionPreprocessor.preprocess(image, width, height);
    }

    private float[][] executeSinglePass(float[] chwTensor,
                                        int offsetX,
                                        int offsetY,
                                        int tileW,
                                        int tileH) throws Exception {
        if (customTiledBackend != null) {
            return customTiledBackend.runTileInference(
                chwTensor, inputWidth, inputHeight, offsetX, offsetY, tileW, tileH);
        }
        if (customBackend != null) {
            return customBackend.runInference(chwTensor, inputWidth, inputHeight);
        }
        return runCachedReflectiveSession(modelPath, chwTensor, inputWidth, inputHeight);
    }

    public static boolean isOnnxRuntimePresentOnClasspath() {
        try {
            Class.forName("ai.onnxruntime.OrtEnvironment");
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    private float[][] runCachedReflectiveSession(Path onnxPath,
                                                 float[] chwTensor,
                                                 int width,
                                                 int height) throws Exception {
        Class<?> envClass = Class.forName("ai.onnxruntime.OrtEnvironment");
        Class<?> sessionClass = Class.forName("ai.onnxruntime.OrtSession");
        Class<?> tensorClass = Class.forName("ai.onnxruntime.OnnxTensor");
        Class<?> valueClass = Class.forName("ai.onnxruntime.OnnxValue");

        Object session;
        synchronized (sessionLock) {
            if (cachedSession == null) {
                Object env = envClass.getMethod("getEnvironment").invoke(null);
                Method createSession = envClass.getMethod("createSession", String.class);
                cachedSession = createSession.invoke(env, onnxPath.toAbsolutePath().toString());
            }
            session = cachedSession;
        }

        try {
            @SuppressWarnings("unchecked")
            java.util.Set<String> inputNames = (java.util.Set<String>) sessionClass.getMethod("getInputNames").invoke(session);
            if (inputNames.isEmpty()) {
                throw new IllegalStateException("ONNX model has no input nodes.");
            }
            String inputName = inputNames.iterator().next();
            long[] shape = new long[]{1L, 3L, height, width};
            Object env = envClass.getMethod("getEnvironment").invoke(null);
            Method createTensor = tensorClass.getMethod("createTensor", envClass, FloatBuffer.class, long[].class);
            Object inputTensor = createTensor.invoke(null, env, FloatBuffer.wrap(chwTensor), shape);
            try {
                Method runMethod = sessionClass.getMethod("run", Map.class);
                Object result = runMethod.invoke(session, Map.of(inputName, inputTensor));
                try {
                    Method getMethod = result.getClass().getMethod("get", int.class);
                    Object firstOutput = getMethod.invoke(result, 0);
                    Object rawVal = valueClass.getMethod("getValue").invoke(firstOutput);
                    if (rawVal instanceof float[][][] batch3d && batch3d.length > 0) {
                        return batch3d[0];
                    }
                    if (rawVal instanceof float[][] mat2d) {
                        return mat2d;
                    }
                    throw new IllegalArgumentException("Unsupported ONNX output tensor Java type: "
                        + (rawVal == null ? "null" : rawVal.getClass().getName()));
                } finally {
                    result.getClass().getMethod("close").invoke(result);
                }
            } finally {
                tensorClass.getMethod("close").invoke(inputTensor);
            }
        } catch (Exception ex) {
            close();
            throw ex;
        }
    }

    public static float[][] runReflectiveOnnxSession(Path onnxPath,
                                                     float[] chwTensor,
                                                     int width,
                                                     int height) throws Exception {
        Class<?> envClass = Class.forName("ai.onnxruntime.OrtEnvironment");
        Class<?> sessionClass = Class.forName("ai.onnxruntime.OrtSession");
        Class<?> tensorClass = Class.forName("ai.onnxruntime.OnnxTensor");
        Class<?> valueClass = Class.forName("ai.onnxruntime.OnnxValue");

        Object env = envClass.getMethod("getEnvironment").invoke(null);
        Method createSession = envClass.getMethod("createSession", String.class);
        Object session = createSession.invoke(env, onnxPath.toAbsolutePath().toString());
        try {
            @SuppressWarnings("unchecked")
            java.util.Set<String> inputNames = (java.util.Set<String>) sessionClass.getMethod("getInputNames").invoke(session);
            if (inputNames.isEmpty()) {
                throw new IllegalStateException("ONNX model has no input nodes.");
            }
            String inputName = inputNames.iterator().next();
            long[] shape = new long[]{1L, 3L, height, width};
            Method createTensor = tensorClass.getMethod("createTensor", envClass, FloatBuffer.class, long[].class);
            Object inputTensor = createTensor.invoke(null, env, FloatBuffer.wrap(chwTensor), shape);
            try {
                Method runMethod = sessionClass.getMethod("run", Map.class);
                Object result = runMethod.invoke(session, Map.of(inputName, inputTensor));
                try {
                    Method getMethod = result.getClass().getMethod("get", int.class);
                    Object firstOutput = getMethod.invoke(result, 0);
                    Object rawVal = valueClass.getMethod("getValue").invoke(firstOutput);
                    if (rawVal instanceof float[][][] batch3d && batch3d.length > 0) {
                        return batch3d[0];
                    }
                    if (rawVal instanceof float[][] mat2d) {
                        return mat2d;
                    }
                    throw new IllegalArgumentException("Unsupported ONNX output tensor Java type: "
                        + (rawVal == null ? "null" : rawVal.getClass().getName()));
                } finally {
                    result.getClass().getMethod("close").invoke(result);
                }
            } finally {
                tensorClass.getMethod("close").invoke(inputTensor);
            }
        } finally {
            sessionClass.getMethod("close").invoke(session);
        }
    }

    @Override
    public void close() {
        synchronized (sessionLock) {
            if (cachedSession != null) {
                try {
                    cachedSession.getClass().getMethod("close").invoke(cachedSession);
                } catch (Throwable ignored) {
                } finally {
                    cachedSession = null;
                }
            }
        }
    }
}
