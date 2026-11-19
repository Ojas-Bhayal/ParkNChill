package com.parknchillai.detection;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

/**
 * Responsibility 3 of the Vehicle Detection Pipeline:
 * <ul>
 *   <li>Auto-detects and decodes the three standard ONNX object-detection output tensor layouts:
 *     <ol>
 *       <li><strong>YOLOv8 / YOLOv11 / RT-DETR Transposed</strong> {@code [4 + C][N]} (e.g. {@code [84][8400]} or {@code [8][8400]}),
 *           where rows {@code 0..3} are {@code cx, cy, w, h} and rows {@code 4..4+C-1} are class scores</li>
 *       <li><strong>YOLOv5 / YOLOv7 Anchor Matrix</strong> {@code [N][5 + C]} (e.g. {@code [25200][85]} or {@code [25200][9]}),
 *           where columns {@code 0..3} are {@code cx, cy, w, h}, column {@code 4} is {@code objectness},
 *           and columns {@code 5..5+C-1} are conditional class probabilities ({@code score = obj * clsProb})</li>
 *       <li><strong>YOLOX raw output</strong> {@code [N][5+C]} (e.g. {@code [8400][85]}), with grid/stride decoding before coordinate mapping</li>
 *       <li><strong>End-to-End NMS / YOLOv10 / SSD Corner Matrix</strong> {@code [N][6]} (e.g. {@code [300][6]}),
 *           where each row is {@code [x1, y1, x2, y2, score, classId]}</li>
 *     </ol>
 *   </li>
 *   <li>Supports both pixel-space coordinates {@code [0, inputDim]} and normalized coordinates {@code [0, 1]}</li>
 *   <li>Filters by supported {@link VehicleCategory} ({@code CAR, MOTORCYCLE, BUS, TRUCK}),
 *       minimum detection score, and minimum bounding-box dimensions</li>
 *   <li>Applies greedy Non-Maximum Suppression ({@code NMS}) and cross-tile containment suppression
 *       ({@code IoMin >= 0.75}) to eliminate duplicate detections across overlapping tiles</li>
 * </ul>
 */
public final class DetectionFilter {

    public static final double DEFAULT_CONFIDENCE_THRESHOLD = 0.30;
    public static final double DEFAULT_NMS_IOU_THRESHOLD = 0.45;
    public static final double CROSS_TILE_IOMIN_THRESHOLD = 0.75;
    public static final double MIN_BOX_SIDE_PX = 6.0;

    private DetectionFilter() { /* utility class */ }

    public enum OutputTensorLayout {
        YOLOV8_TRANSPOSED_4_PLUS_C_BY_N,
        YOLOV5_ANCHOR_N_BY_5_PLUS_C,
        END_TO_END_NMS_N_BY_6
    }

    /**
     * Inspects the 2D output matrix dimensions and values to determine its ONNX output layout,
     * throwing {@link IllegalArgumentException} if the shape is incompatible with all supported
     * vehicle-detection formats.
     */
    public static OutputTensorLayout detectOutputLayout(float[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0) {
            throw new IllegalArgumentException("ONNX output tensor matrix is null or empty.");
        }
        int rows = matrix.length;
        int cols = matrix[0].length;
        for (float[] r : matrix) {
            if (r == null || r.length != cols) {
                throw new IllegalArgumentException("Ragged ONNX output tensor matrix.");
            }
        }

        // Layout C: [N][6] where cols == 6 and rows != 5 and rows != 8 and rows != 84
        if (cols == 6 && rows != 5 && rows != 8 && rows != 84) {
            return OutputTensorLayout.END_TO_END_NMS_N_BY_6;
        }

        // Layout A: YOLOv8 transposed [4 + C][N] where rows is 5 (1-class), 8 (4-class), or 84 (80-class COCO)
        if (rows == 5 || rows == 8 || rows == 84) {
            return OutputTensorLayout.YOLOV8_TRANSPOSED_4_PLUS_C_BY_N;
        }

        // Layout B: YOLOv5 anchor [N][5 + C] where cols is 6 (1-class), 9 (4-class), or 85 (80-class COCO)
        if (cols == 9 || cols == 85) {
            return OutputTensorLayout.YOLOV5_ANCHOR_N_BY_5_PLUS_C;
        }

        throw new IllegalArgumentException(String.format(Locale.ROOT,
            "Incompatible ONNX output tensor shape [%d x %d]: expected YOLOv8 [84|8|5 x N], YOLOv5 [N x 85|9], or End-to-End NMS [N x 6].",
            rows, cols));
    }

    /**
     * Automatically detects the output tensor layout and decodes detections mapped to
     * {@code letterboxMeta}'s original image coordinates.
     */
    public static List<VehicleDetection> decodeAuto(
            float[][] outputMatrix,
            DetectionPreprocessor.LetterboxResult letterboxMeta,
            double minScoreThreshold,
            String modelId) {
        OutputTensorLayout layout = detectOutputLayout(outputMatrix);
        return switch (layout) {
            case YOLOV8_TRANSPOSED_4_PLUS_C_BY_N ->
                decodeYoloV8Transposed(outputMatrix, letterboxMeta, minScoreThreshold, modelId);
            case YOLOV5_ANCHOR_N_BY_5_PLUS_C ->
                decodeYoloV5Anchors(outputMatrix, letterboxMeta, minScoreThreshold, modelId);
            case END_TO_END_NMS_N_BY_6 ->
                decodeEndToEndNmsCorners(outputMatrix, letterboxMeta, minScoreThreshold, modelId);
        };
    }

    /**
     * Decodes a YOLOv8-style output matrix of shape {@code [4 + numClasses][numPredictions]}
     * where rows {@code 0..3} are {@code cx, cy, w, h} in letterbox coordinates (or normalized {@code [0,1]})
     * and rows {@code 4..4+numClasses-1} are class scores in {@code [0, 1]}.
     */
    public static List<VehicleDetection> decodeYoloV8Transposed(
            float[][] channelsByAnchors,
            DetectionPreprocessor.LetterboxResult letterboxMeta,
            double minScoreThreshold,
            String modelId) {
        if (channelsByAnchors == null || channelsByAnchors.length < 5) {
            throw new IllegalArgumentException("YOLOv8 output tensor must have at least 5 rows (4 box + >=1 class).");
        }
        int numChannels = channelsByAnchors.length;
        int numClasses = numChannels - 4;
        if (numClasses != 1 && numClasses != 4 && numClasses != 80) {
            throw new IllegalArgumentException("Unsupported YOLOv8 class count: " + numClasses + " (expected 1, 4, or 80).");
        }
        int numAnchors = channelsByAnchors[0].length;
        for (float[] row : channelsByAnchors) {
            if (row == null || row.length != numAnchors) {
                throw new IllegalArgumentException("Inconsistent anchor length in YOLOv8 output tensor.");
            }
        }

        boolean normalizedCoords = isNormalizedBoxRange(
            channelsByAnchors[0], channelsByAnchors[1], channelsByAnchors[2], channelsByAnchors[3]);
        String meta = buildProcessingMetadata("YOLOv8-transposed", letterboxMeta, normalizedCoords);

        List<VehicleDetection> raw = new ArrayList<>();
        int seq = 1;
        for (int a = 0; a < numAnchors; a++) {
            double cx = channelsByAnchors[0][a];
            double cy = channelsByAnchors[1][a];
            double bw = channelsByAnchors[2][a];
            double bh = channelsByAnchors[3][a];
            if (!Double.isFinite(cx) || !Double.isFinite(cy) || !Double.isFinite(bw) || !Double.isFinite(bh)) {
                continue;
            }
            if (normalizedCoords) {
                cx *= letterboxMeta.inputWidth();
                cy *= letterboxMeta.inputHeight();
                bw *= letterboxMeta.inputWidth();
                bh *= letterboxMeta.inputHeight();
            }
            if (bw <= 1.0 || bh <= 1.0) continue;

            double bestScore = -1.0;
            VehicleCategory bestCat = null;
            for (int c = 0; c < numClasses; c++) {
                Optional<VehicleCategory> catOpt = VehicleCategory.fromModelClassIndex(c, numClasses);
                if (catOpt.isEmpty()) continue;
                float score = channelsByAnchors[4 + c][a];
                if (Float.isFinite(score) && score >= minScoreThreshold && score > bestScore) {
                    bestScore = Math.min(1.0, score);
                    bestCat = catOpt.get();
                }
            }

            if (bestCat != null) {
                double lx = cx - bw / 2.0;
                double ly = cy - bh / 2.0;
                double[] mapped = DetectionPreprocessor.mapBoxToOriginalImage(lx, ly, bw, bh, letterboxMeta);
                if (mapped[2] >= MIN_BOX_SIDE_PX && mapped[3] >= MIN_BOX_SIDE_PX) {
                    String detId = String.format(Locale.ROOT, "VEH-%03d", seq++);
                    raw.add(new VehicleDetection(
                        detId, bestCat, mapped[0], mapped[1], mapped[2], mapped[3], bestScore, modelId,
                        letterboxMeta.inputWidth(), letterboxMeta.inputHeight(), meta
                    ));
                }
            }
        }
        return raw;
    }

    /**
     * Decodes a YOLOv5/YOLOv7 output matrix of shape {@code [numAnchors][5 + numClasses]}
     * where columns {@code 0..3} are {@code cx, cy, w, h}, column {@code 4} is {@code objectness},
     * and columns {@code 5..5+numClasses-1} are class probabilities.
     */
    public static List<VehicleDetection> decodeYoloV5Anchors(
            float[][] anchorsByCols,
            DetectionPreprocessor.LetterboxResult letterboxMeta,
            double minScoreThreshold,
            String modelId) {
        if (anchorsByCols == null || anchorsByCols.length == 0 || anchorsByCols[0].length < 6) {
            throw new IllegalArgumentException("YOLOv5 output tensor must have shape [N][>=6].");
        }
        int numCols = anchorsByCols[0].length;
        int numClasses = numCols - 5;
        if (numClasses != 1 && numClasses != 4 && numClasses != 80) {
            throw new IllegalArgumentException("Unsupported YOLOv5 class count: " + numClasses + " (expected 1, 4, or 80).");
        }

        double maxCoord = 0.0;
        for (float[] row : anchorsByCols) {
            if (row == null || row.length != numCols) {
                throw new IllegalArgumentException("Inconsistent row width in YOLOv5 output tensor.");
            }
            maxCoord = Math.max(maxCoord, Math.max(row[2], row[3]));
        }
        boolean normalizedCoords = maxCoord > 0.0 && maxCoord <= 1.5;
        String meta = buildProcessingMetadata("YOLOv5-anchor", letterboxMeta, normalizedCoords);

        List<VehicleDetection> raw = new ArrayList<>();
        int seq = 1;
        for (float[] row : anchorsByCols) {
            float objConf = row[4];
            if (!Float.isFinite(objConf) || objConf < minScoreThreshold) continue;

            double cx = row[0];
            double cy = row[1];
            double bw = row[2];
            double bh = row[3];
            if (!Double.isFinite(cx) || !Double.isFinite(cy) || !Double.isFinite(bw) || !Double.isFinite(bh)) {
                continue;
            }
            if (normalizedCoords) {
                cx *= letterboxMeta.inputWidth();
                cy *= letterboxMeta.inputHeight();
                bw *= letterboxMeta.inputWidth();
                bh *= letterboxMeta.inputHeight();
            }
            if (bw <= 1.0 || bh <= 1.0) continue;

            double bestScore = -1.0;
            VehicleCategory bestCat = null;
            for (int c = 0; c < numClasses; c++) {
                Optional<VehicleCategory> catOpt = VehicleCategory.fromModelClassIndex(c, numClasses);
                if (catOpt.isEmpty()) continue;
                double combined = (double) objConf * row[5 + c];
                if (Double.isFinite(combined) && combined >= minScoreThreshold && combined > bestScore) {
                    bestScore = Math.min(1.0, combined);
                    bestCat = catOpt.get();
                }
            }

            if (bestCat != null) {
                double lx = cx - bw / 2.0;
                double ly = cy - bh / 2.0;
                double[] mapped = DetectionPreprocessor.mapBoxToOriginalImage(lx, ly, bw, bh, letterboxMeta);
                if (mapped[2] >= MIN_BOX_SIDE_PX && mapped[3] >= MIN_BOX_SIDE_PX) {
                    String detId = String.format(Locale.ROOT, "VEH-%03d", seq++);
                    raw.add(new VehicleDetection(
                        detId, bestCat, mapped[0], mapped[1], mapped[2], mapped[3], bestScore, modelId,
                        letterboxMeta.inputWidth(), letterboxMeta.inputHeight(), meta
                    ));
                }
            }
        }
        return raw;
    }


    /**
     * Decodes the raw anchor matrix returned by the official YOLOX-s ONNX export
     * (normally [8400][85] for a 640x640 input). Unlike YOLOv5, raw YOLOX box
     * coordinates are grid-relative and width/height are log-space; they must be
     * decoded with strides 8, 16, and 32 before mapping back to the source image.
     * Objectness and class scores are already sigmoid probabilities in the official
     * YOLOX inference output.
     */
    public static List<VehicleDetection> decodeYoloX(
            float[][] anchorsByCols,
            DetectionPreprocessor.LetterboxResult letterboxMeta,
            int inputWidth,
            int inputHeight,
            double minScoreThreshold,
            String modelId) {
        if (anchorsByCols == null || anchorsByCols.length == 0 || anchorsByCols[0] == null) {
            throw new IllegalArgumentException("YOLOX output tensor is null or empty.");
        }
        if (letterboxMeta == null) throw new IllegalArgumentException("YOLOX letterbox metadata is required.");
        if (inputWidth < 32 || inputHeight < 32 || inputWidth % 32 != 0 || inputHeight % 32 != 0) {
            throw new IllegalArgumentException("YOLOX input dimensions must be multiples of 32.");
        }
        int numCols = anchorsByCols[0].length;
        int numClasses = numCols - 5;
        if (numClasses != 1 && numClasses != 4 && numClasses != 80) {
            throw new IllegalArgumentException("Unsupported YOLOX class count: " + numClasses + " (expected 1, 4, or 80).");
        }
        int expectedRows = 0;
        for (int stride : new int[]{8, 16, 32}) {
            expectedRows += (inputWidth / stride) * (inputHeight / stride);
        }
        if (anchorsByCols.length != expectedRows) {
            throw new IllegalArgumentException("YOLOX raw output should contain " + expectedRows
                + " rows for input " + inputWidth + "x" + inputHeight + ", but got " + anchorsByCols.length
                + ". If this model already decodes boxes inside its graph, use a model-specific decoded-output profile instead.");
        }
        for (float[] row : anchorsByCols) {
            if (row == null || row.length != numCols) {
                throw new IllegalArgumentException("Inconsistent row width in YOLOX output tensor.");
            }
        }

        List<VehicleDetection> raw = new ArrayList<>();
        int seq = 1;
        int anchor = 0;
        for (int stride : new int[]{8, 16, 32}) {
            int gridW = inputWidth / stride;
            int gridH = inputHeight / stride;
            for (int gy = 0; gy < gridH; gy++) {
                for (int gx = 0; gx < gridW; gx++, anchor++) {
                    float[] row = anchorsByCols[anchor];
                    double tx = row[0], ty = row[1], tw = row[2], th = row[3];
                    double objectness = row[4];
                    if (!Double.isFinite(tx) || !Double.isFinite(ty) || !Double.isFinite(tw)
                            || !Double.isFinite(th) || !Double.isFinite(objectness)) continue;
                    if (objectness < 0.0 || objectness > 1.0) {
                        throw new IllegalArgumentException("YOLOX objectness values are not probabilities; verify model export/output profile.");
                    }
                    double cx = (tx + gx) * stride;
                    double cy = (ty + gy) * stride;
                    double bw = Math.exp(Math.max(-20.0, Math.min(20.0, tw))) * stride;
                    double bh = Math.exp(Math.max(-20.0, Math.min(20.0, th))) * stride;
                    if (!Double.isFinite(cx) || !Double.isFinite(cy) || !Double.isFinite(bw) || !Double.isFinite(bh)
                            || bw <= 1.0 || bh <= 1.0) continue;

                    double bestScore = -1.0;
                    VehicleCategory bestCategory = null;
                    for (int cls = 0; cls < numClasses; cls++) {
                        Optional<VehicleCategory> category = VehicleCategory.fromModelClassIndex(cls, numClasses);
                        if (category.isEmpty()) continue;
                        float clsScore = row[5 + cls];
                        if (!Float.isFinite(clsScore) || clsScore < 0.0f || clsScore > 1.0f) {
                            throw new IllegalArgumentException("YOLOX class scores are not probabilities; verify model export/output profile.");
                        }
                        double score = objectness * clsScore;
                        if (score >= minScoreThreshold && score > bestScore) {
                            bestScore = score;
                            bestCategory = category.get();
                        }
                    }
                    if (bestCategory == null) continue;
                    double[] mapped = DetectionPreprocessor.mapBoxToOriginalImage(
                        cx - bw / 2.0, cy - bh / 2.0, bw, bh, letterboxMeta);
                    if (mapped[2] >= MIN_BOX_SIDE_PX && mapped[3] >= MIN_BOX_SIDE_PX) {
                        raw.add(new VehicleDetection(
                            String.format(Locale.ROOT, "YOLOX-%03d", seq++), bestCategory,
                            mapped[0], mapped[1], mapped[2], mapped[3], bestScore, modelId,
                            inputWidth, inputHeight,
                            String.format(Locale.ROOT, "YOLOX-raw,stride=%d,grid=(%d,%d)", stride, gx, gy)));
                    }
                }
            }
        }
        return raw;
    }

    /**
     * Decodes an End-to-End NMS / YOLOv10 / SSD output matrix of shape {@code [N][6]}
     * where each row is {@code [x1, y1, x2, y2, score, classId]}.
     */
    public static List<VehicleDetection> decodeEndToEndNmsCorners(
            float[][] rowsBySix,
            DetectionPreprocessor.LetterboxResult letterboxMeta,
            double minScoreThreshold,
            String modelId) {
        if (rowsBySix == null || rowsBySix.length == 0 || rowsBySix[0].length != 6) {
            throw new IllegalArgumentException("End-to-End NMS output tensor must have shape [N][6].");
        }
        double maxCoord = 0.0;
        for (float[] row : rowsBySix) {
            if (row == null || row.length != 6) {
                throw new IllegalArgumentException("Inconsistent row width in [N][6] output tensor.");
            }
            maxCoord = Math.max(maxCoord, Math.max(row[2], row[3]));
        }
        boolean normalizedCoords = maxCoord > 0.0 && maxCoord <= 1.5;
        String meta = buildProcessingMetadata("EndToEnd-NMS", letterboxMeta, normalizedCoords);

        List<VehicleDetection> raw = new ArrayList<>();
        int seq = 1;
        for (float[] row : rowsBySix) {
            double x1 = row[0];
            double y1 = row[1];
            double x2 = row[2];
            double y2 = row[3];
            double score = row[4];
            int clsId = Math.round(row[5]);

            if (!Double.isFinite(x1) || !Double.isFinite(y1) || !Double.isFinite(x2) || !Double.isFinite(y2)
                    || !Double.isFinite(score) || score < minScoreThreshold) {
                continue;
            }
            if (normalizedCoords) {
                x1 *= letterboxMeta.inputWidth();
                y1 *= letterboxMeta.inputHeight();
                x2 *= letterboxMeta.inputWidth();
                y2 *= letterboxMeta.inputHeight();
            }
            double bw = x2 - x1;
            double bh = y2 - y1;
            if (bw <= 1.0 || bh <= 1.0) continue;

            Optional<VehicleCategory> catOpt = VehicleCategory.fromCocoClassId(clsId);
            if (catOpt.isEmpty() && clsId >= 0 && clsId < 4) {
                catOpt = VehicleCategory.fromModelClassIndex(clsId, 4);
            }
            if (catOpt.isEmpty()) continue;

            double[] mapped = DetectionPreprocessor.mapBoxToOriginalImage(x1, y1, bw, bh, letterboxMeta);
            if (mapped[2] >= MIN_BOX_SIDE_PX && mapped[3] >= MIN_BOX_SIDE_PX) {
                String detId = String.format(Locale.ROOT, "VEH-%03d", seq++);
                raw.add(new VehicleDetection(
                    detId, catOpt.get(), mapped[0], mapped[1], mapped[2], mapped[3],
                    Math.min(1.0, score), modelId,
                    letterboxMeta.inputWidth(), letterboxMeta.inputHeight(), meta
                ));
            }
        }
        return raw;
    }

    private static String buildProcessingMetadata(String layoutName,
                                                  DetectionPreprocessor.LetterboxResult lb,
                                                  boolean normalizedCoords) {
        return String.format(Locale.ROOT,
            "layout=%s,orig=%dx%d,inf=%dx%d,scale=%.3f,pad=(%d,%d),coords=%s",
            layoutName, lb.originalWidth(), lb.originalHeight(),
            lb.inputWidth(), lb.inputHeight(), lb.scale(), lb.padX(), lb.padY(),
            normalizedCoords ? "normalized" : "pixel");
    }

    private static boolean isNormalizedBoxRange(float[] cx, float[] cy, float[] w, float[] h) {
        double max = 0.0;
        for (int i = 0; i < cx.length; i++) {
            if (Float.isFinite(w[i]) && w[i] > max) max = w[i];
            if (Float.isFinite(h[i]) && h[i] > max) max = h[i];
            if (Float.isFinite(cx[i]) && cx[i] > max) max = cx[i];
            if (Float.isFinite(cy[i]) && cy[i] > max) max = cy[i];
        }
        return max > 0.0 && max <= 1.5;
    }

    /**
     * Filters candidate detections by minimum score and applies greedy Non-Maximum Suppression (NMS).
     */
    public static List<VehicleDetection> filterAndSuppressDuplicates(
            List<VehicleDetection> candidates,
            double minScoreThreshold,
            double nmsIouThreshold) {
        return mergeTiledDetectionsWithNms(candidates, minScoreThreshold, nmsIouThreshold, CROSS_TILE_IOMIN_THRESHOLD);
    }

    /**
     * Merges candidate detections from full-frame and overlapping tile passes using greedy NMS
     * ({@code iou >= nmsIouThreshold}) AND containment suppression
     * ({@code intersection / min(areaA, areaB) >= ioMinThreshold}) so partial boxes at tile borders
     * are merged cleanly into the full vehicle box.
     */
    public static List<VehicleDetection> mergeTiledDetectionsWithNms(
            List<VehicleDetection> candidates,
            double minScoreThreshold,
            double nmsIouThreshold,
            double ioMinThreshold) {
        if (candidates == null || candidates.isEmpty()) {
            return List.of();
        }
        if (!Double.isFinite(minScoreThreshold) || minScoreThreshold < 0.0 || minScoreThreshold > 1.0) {
            throw new IllegalArgumentException("minScoreThreshold must be in [0.0, 1.0].");
        }
        if (!Double.isFinite(nmsIouThreshold) || nmsIouThreshold <= 0.0 || nmsIouThreshold > 1.0) {
            throw new IllegalArgumentException("nmsIouThreshold must be in (0.0, 1.0].");
        }

        List<VehicleDetection> eligible = new ArrayList<>();
        for (VehicleDetection det : candidates) {
            if (det != null
                    && det.detectionScore() >= minScoreThreshold
                    && det.width() >= MIN_BOX_SIDE_PX
                    && det.height() >= MIN_BOX_SIDE_PX) {
                eligible.add(det);
            }
        }

        eligible.sort(Comparator.comparingDouble(VehicleDetection::detectionScore).reversed());

        List<VehicleDetection> kept = new ArrayList<>();
        boolean[] suppressed = new boolean[eligible.size()];
        int reseq = 1;

        for (int i = 0; i < eligible.size(); i++) {
            if (suppressed[i]) continue;
            VehicleDetection current = eligible.get(i);
            kept.add(new VehicleDetection(
                String.format(Locale.ROOT, "VEH-%03d", reseq++),
                current.category(),
                current.x(),
                current.y(),
                current.width(),
                current.height(),
                current.detectionScore(),
                current.modelId(),
                current.inferenceWidth(),
                current.inferenceHeight(),
                current.processingMetadata() + ",nms=kept"
            ));

            for (int j = i + 1; j < eligible.size(); j++) {
                if (suppressed[j]) continue;
                VehicleDetection other = eligible.get(j);
                double iou = current.iouWith(other);
                double inter = current.intersectionArea(other.x(), other.y(), other.width(), other.height());
                double minArea = Math.min(current.area(), other.area());
                double ioMin = minArea > 0.0 ? inter / minArea : 0.0;
                if (iou >= nmsIouThreshold || ioMin >= ioMinThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        return List.copyOf(kept);
    }
}
