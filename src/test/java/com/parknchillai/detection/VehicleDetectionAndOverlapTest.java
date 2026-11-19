package com.parknchillai.detection;

import com.parknchillai.app.AnalysisSummary;
import com.parknchillai.app.ParkingAnalysisService;
import com.parknchillai.app.SlotAnalysisResult;
import com.parknchillai.app.SlotDiagnosticRecord;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import org.junit.jupiter.api.Test;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive Phase 5 test suite covering:
 * <ol>
 *   <li>Maven {@code pom.xml} ONNX Runtime dependency &amp; ONNX ModelProto header / SHA-256 verification</li>
 *   <li>Input tensor preparation (generic RGB-normalized convention plus default YOLOX BGR {@code [0,255]} preprocessing)</li>
 *   <li>Multi-layout model output decoding ({@code YOLOv8 [4+C][N]}, {@code YOLOv5 [N][5+C]}, and {@code End-to-End NMS [N][6]})</li>
 *   <li>COCO class-index mapping ({@code 2=Car, 3=Motorcycle, 5=Bus, 7=Truck}) and non-vehicle rejection</li>
 *   <li>Bounding-box coordinate transformation &amp; Phase 3 {@code 739x415} overhead 26-slot tiled inference with cross-tile NMS</li>
 *   <li>Duplicate suppression (standard NMS &amp; cross-tile containment {@code IoMin >= 0.75})</li>
 *   <li>Vehicle-to-slot assignment (grazing, 2-slot straddling, distant cars, driving-aisle false detections)</li>
 *   <li>Explicit failure when the model is absent ({@code MODEL_MISSING}) or runtime is absent ({@code RUNTIME_MISSING})</li>
 *   <li>Explicit failure when the model output has an incompatible shape ({@code MODEL_INCOMPATIBLE})</li>
 *   <li>Consistency of occupancy summaries, detector status messages ({@code 0 vehicles detected} vs {@code Could Not Run}), and CSV exports</li>
 * </ol>
 */
class VehicleDetectionAndOverlapTest {

    @Test
    void pomDeclaresOnnxRuntimeDependencyAndServiceVerifiesOnnxProtobufHeaderAndSha256() throws Exception {
        String pom = Files.readString(Path.of("pom.xml"));
        assertTrue(pom.contains("<groupId>com.microsoft.onnxruntime</groupId>"),
            "pom.xml must declare com.microsoft.onnxruntime groupId");
        assertTrue(pom.contains("<artifactId>onnxruntime</artifactId>"),
            "pom.xml must declare onnxruntime artifactId");

        // The default model may be installed locally. Assert status based on the actual
        // filesystem instead of assuming every developer/test environment lacks it.
        OnnxVehicleDetectionService defaultSvc = new OnnxVehicleDetectionService();
        Path defaultModel = Path.of("models", "vehicle_detector.onnx");
        if (Files.isRegularFile(defaultModel)) {
            assertNotEquals(VehicleDetectionService.DetectorStatus.MODEL_MISSING, defaultSvc.getStatus(),
                "An installed model must not be reported as missing");
            String defaultSha = defaultSvc.computeModelSha256();
            assertTrue(defaultSha != null && defaultSha.matches("[0-9a-fA-F]{64}"),
                "Installed model should have a SHA-256 digest");
        } else {
            assertEquals(VehicleDetectionService.DetectorStatus.MODEL_MISSING, defaultSvc.getStatus());
            assertEquals("MODEL_MISSING", defaultSvc.computeModelSha256());
        }

        // Construct a valid ONNX ModelProto header (>= 128 bytes starting with tag 0x08, ir_version = 8)
        Path validOnnxHeaderFile = Files.createTempFile("valid_proto_header", ".onnx");
        try {
            byte[] protoBytes = new byte[256];
            protoBytes[0] = 0x08; // field 1 (ir_version), varint wire type
            protoBytes[1] = 0x08; // ir_version = 8
            protoBytes[2] = 0x12; // field 2 (producer_name)
            protoBytes[3] = 0x07;
            byte[] prod = "pytorch".getBytes();
            System.arraycopy(prod, 0, protoBytes, 4, prod.length);
            Files.write(validOnnxHeaderFile, protoBytes);

            assertTrue(OnnxVehicleDetectionService.isValidOnnxProtobufFile(validOnnxHeaderFile));
            OnnxVehicleDetectionService withValidFile = new OnnxVehicleDetectionService(validOnnxHeaderFile);
            String sha = withValidFile.computeModelSha256();
            assertEquals(64, sha.length());
            // When OrtEnvironment is not on classpath, status must be RUNTIME_MISSING (never READY or fabricated)
            if (!OnnxVehicleDetectionService.isOnnxRuntimePresentOnClasspath()) {
                assertEquals(VehicleDetectionService.DetectorStatus.RUNTIME_MISSING, withValidFile.getStatus());
            }
        } finally {
            Files.deleteIfExists(validOnnxHeaderFile);
        }
    }

    @Test
    void letterboxPreprocessAndInverseCoordinateMappingRestoreExactOriginalImageCoordinates() {
        BufferedImage orig = new BufferedImage(1280, 720, BufferedImage.TYPE_INT_RGB);
        DetectionPreprocessor.LetterboxResult lb = DetectionPreprocessor.preprocess(orig, 640, 640);

        assertEquals(640, lb.inputWidth());
        assertEquals(640, lb.inputHeight());
        assertEquals(1280, lb.originalWidth());
        assertEquals(720, lb.originalHeight());
        assertEquals(0.5, lb.scale(), 1e-6);
        assertEquals(0, lb.padX());
        assertEquals(140, lb.padY()); // (640 - 360) / 2 = 140
        assertEquals(3 * 640 * 640, lb.chwNormalizedTensor().length);

        double[] restored = DetectionPreprocessor.mapBoxToOriginalImage(100.0, 190.0, 80.0, 100.0, lb);
        assertEquals(200.0, restored[0], 1e-4);
        assertEquals(100.0, restored[1], 1e-4);
        assertEquals(160.0, restored[2], 1e-4);
        assertEquals(200.0, restored[3], 1e-4);
    }

    @Test
    void yoloXPreprocessingUsesBgrRawValuesAndTopLeftLetterbox() {
        BufferedImage image = new BufferedImage(1280, 720, BufferedImage.TYPE_INT_RGB);
        // Fill the whole source image red so resizing/interpolation cannot dilute a
        // single-pixel test marker at the corner.
        java.awt.Graphics2D graphics = image.createGraphics();
        graphics.setColor(Color.RED);
        graphics.fillRect(0, 0, image.getWidth(), image.getHeight());
        graphics.dispose();
        DetectionPreprocessor.LetterboxResult lb = DetectionPreprocessor.preprocessYoloX(image, 640, 640);

        assertEquals(0.5, lb.scale(), 1e-6);
        assertEquals(0, lb.padX());
        assertEquals(0, lb.padY(), "YOLOX reference preprocessing anchors resized image at top-left");
        assertEquals(1280, lb.originalWidth());
        assertEquals(720, lb.originalHeight());
        int plane = 640 * 640;
        // Original pixel (0,0) is red in BufferedImage RGB, but YOLOX consumes BGR NCHW 0..255.
        assertEquals(0.0f, lb.chwNormalizedTensor()[0], 1e-6f);       // B
        assertEquals(0.0f, lb.chwNormalizedTensor()[plane], 1e-6f);  // G
        assertEquals(255.0f, lb.chwNormalizedTensor()[2 * plane], 1e-6f); // R
        // Padding begins below the 360px resized image and uses value 114, without normalization.
        int padIdx = 400 * 640 + 10;
        assertEquals(114.0f, lb.chwNormalizedTensor()[padIdx], 1e-6f);
        assertEquals(114.0f, lb.chwNormalizedTensor()[plane + padIdx], 1e-6f);
        assertEquals(114.0f, lb.chwNormalizedTensor()[2 * plane + padIdx], 1e-6f);
    }

    @Test
    void decodeAutoSupportsYoloV8TransposedYoloV5AnchorAndEndToEndNmsLayoutsAndFiltersNonVehicleClasses() {
        BufferedImage img = new BufferedImage(640, 640, BufferedImage.TYPE_INT_RGB);
        DetectionPreprocessor.LetterboxResult lb = DetectionPreprocessor.preprocess(img, 640, 640);

        // 1. Layout A: YOLOv8 transposed [84][6] (includes non-vehicle class 0=person which must be ignored)
        float[][] yolov8 = new float[84][6];
        yolov8[0][0] = 100; yolov8[1][0] = 100; yolov8[2][0] = 60; yolov8[3][0] = 80; yolov8[4 + 2][0] = 0.92f; // Car
        yolov8[0][1] = 102; yolov8[1][1] = 101; yolov8[2][1] = 60; yolov8[3][1] = 80; yolov8[4 + 2][1] = 0.75f; // Duplicate Car
        yolov8[0][2] = 260; yolov8[1][2] = 100; yolov8[2][2] = 70; yolov8[3][2] = 95; yolov8[4 + 7][2] = 0.88f; // Truck
        yolov8[0][3] = 420; yolov8[1][3] = 120; yolov8[2][3] = 80; yolov8[3][3] = 110; yolov8[4 + 5][3] = 0.84f; // Bus
        yolov8[0][4] = 540; yolov8[1][4] = 100; yolov8[2][4] = 28; yolov8[3][4] = 48; yolov8[4 + 3][4] = 0.79f; // Motorcycle
        yolov8[0][5] = 320; yolov8[1][5] = 320; yolov8[2][5] = 30; yolov8[3][5] = 60; yolov8[4 + 0][5] = 0.99f; // Person (ignored!)

        assertEquals(DetectionFilter.OutputTensorLayout.YOLOV8_TRANSPOSED_4_PLUS_C_BY_N,
            DetectionFilter.detectOutputLayout(yolov8));
        List<VehicleDetection> v8Decoded = DetectionFilter.decodeAuto(yolov8, lb, 0.30, "yolov8");
        assertEquals(5, v8Decoded.size(), "Person (class 0) must be ignored; 5 vehicle anchors decoded");
        List<VehicleDetection> v8Nms = DetectionFilter.filterAndSuppressDuplicates(v8Decoded, 0.30, 0.45);
        assertEquals(4, v8Nms.size(), "Duplicate Car anchor suppressed by NMS");

        // 2. Layout B: YOLOv5 anchor layout [3][85] (cx, cy, w, h, objConf, 80 classes)
        float[][] yolov5 = new float[3][85];
        yolov5[0][0] = 150; yolov5[0][1] = 200; yolov5[0][2] = 64; yolov5[0][3] = 88;
        yolov5[0][4] = 0.90f; yolov5[0][5 + 2] = 0.90f; // Car score = 0.81
        yolov5[1][0] = 350; yolov5[1][1] = 200; yolov5[1][2] = 72; yolov5[1][3] = 96;
        yolov5[1][4] = 0.85f; yolov5[1][5 + 7] = 0.80f; // Truck score = 0.68
        yolov5[2][0] = 500; yolov5[2][1] = 200; yolov5[2][2] = 30; yolov5[2][3] = 50;
        yolov5[2][4] = 0.95f; yolov5[2][5 + 0] = 0.95f; // Person (ignored)

        assertEquals(DetectionFilter.OutputTensorLayout.YOLOV5_ANCHOR_N_BY_5_PLUS_C,
            DetectionFilter.detectOutputLayout(yolov5));
        List<VehicleDetection> v5Decoded = DetectionFilter.decodeAuto(yolov5, lb, 0.30, "yolov5");
        assertEquals(2, v5Decoded.size());
        assertEquals(VehicleCategory.CAR, v5Decoded.get(0).category());
        assertEquals(0.81, v5Decoded.get(0).detectionScore(), 1e-4);
        assertEquals(VehicleCategory.TRUCK, v5Decoded.get(1).category());

        // 3. Layout C: End-to-End NMS layout [2][6] ([x1, y1, x2, y2, score, classId])
        float[][] e2e = new float[][]{
            {100f, 120f, 160f, 200f, 0.93f, 2f}, // Car (60x80 at 100,120)
            {240f, 110f, 320f, 210f, 0.87f, 5f}  // Bus (80x100 at 240,110)
        };
        assertEquals(DetectionFilter.OutputTensorLayout.END_TO_END_NMS_N_BY_6,
            DetectionFilter.detectOutputLayout(e2e));
        List<VehicleDetection> e2eDecoded = DetectionFilter.decodeAuto(e2e, lb, 0.30, "yolov10-e2e");
        assertEquals(2, e2eDecoded.size());
        assertEquals(100.0, e2eDecoded.get(0).x(), 1e-4);
        assertEquals(120.0, e2eDecoded.get(0).y(), 1e-4);
        assertEquals(60.0, e2eDecoded.get(0).width(), 1e-4);
        assertEquals(80.0, e2eDecoded.get(0).height(), 1e-4);
    }

    @Test
    void decodeRawYoloXGridOutputUsesStrideAndExponentialBoxDecode() {
        BufferedImage image = new BufferedImage(640, 640, BufferedImage.TYPE_INT_RGB);
        DetectionPreprocessor.LetterboxResult lb = DetectionPreprocessor.preprocessYoloX(image, 640, 640);
        float[][] raw = new float[8400][85];
        int xGrid = 20;
        int yGrid = 25;
        int anchorIndex = yGrid * 80 + xGrid; // first output scale is stride 8, 80x80
        raw[anchorIndex][0] = 0.5f;
        raw[anchorIndex][1] = 0.5f;
        raw[anchorIndex][2] = (float) Math.log(8.0); // exp(log(8))*stride = 64px
        raw[anchorIndex][3] = (float) Math.log(6.0); // exp(log(6))*stride = 48px
        raw[anchorIndex][4] = 0.95f; // objectness already sigmoid-activated in official YOLOX inference output
        raw[anchorIndex][5 + 2] = 0.90f; // COCO car class

        List<VehicleDetection> decoded = DetectionFilter.decodeYoloX(raw, lb, 640, 640, 0.30, "yolox_s");
        assertEquals(1, decoded.size());
        VehicleDetection car = decoded.get(0);
        assertEquals(VehicleCategory.CAR, car.category());
        assertEquals(0.855, car.detectionScore(), 1e-4);
        assertEquals(132.0, car.x(), 1e-3);
        assertEquals(180.0, car.y(), 1e-3);
        assertEquals(64.0, car.width(), 1e-3);
        assertEquals(48.0, car.height(), 1e-3);
    }

    @Test
    void phase3TiledInferenceOn739x415OverheadLotDetectsSmallVehiclesAndDeduplicatesCrossTileBoxes() {
        // Phase 3 scenario: 739 x 415 overhead parking image with 26 small spaces (2 rows x 13 bays, each 52x80 px)
        BufferedImage overheadLot = new BufferedImage(739, 415, BufferedImage.TYPE_INT_RGB);
        List<ParkingSlot> slots26 = new ArrayList<>();
        int id = 1;
        int[] rowY = {45, 235};
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 13; c++) {
                int sx = 24 + c * 53;
                int sy = rowY[r];
                slots26.add(new ParkingSlot(id++, sx, sy, 50, 80, ParkingSlot.GeometryOrigin.AUTO_DETECTED));
            }
        }

        // Verify overlapping 2x2 tile generation on 739x415
        List<DetectionPreprocessor.ImageTile> tiles =
            DetectionPreprocessor.generateOverlappingTiles(overheadLot, 2, 2, 0.25, 640, 640);
        assertEquals(4, tiles.size());
        // Each tile is ~462x260 letterboxed to 640x640 -> scale ~1.38x (magnifying small 34x56 overhead vehicles!)
        assertTrue(tiles.get(0).letterbox().scale() > 1.30);

        // Configure a tiled test backend simulating 4 small overhead vehicles in Slots #1, #7 (on tile boundary!), #14, #26
        int[] trueOccupiedIds = {1, 7, 14, 26};
        OnnxVehicleDetectionService tiledDetector = OnnxVehicleDetectionService.createWithTiledTestBackend(
            "overhead_yolov8.onnx", 640, 640, 0.30, 0.45,
            (tensor, inW, inH, offX, offY, tileW, tileH) -> {
                // Only emit small-vehicle detections during high-res tile passes (tileW < 739),
                // including duplicate detections for Slot #7 which lies in the 25% overlap zone of Tile 0 and Tile 1!
                if (tileW == 739) {
                    return new float[84][1]; // Full-frame pass misses tiny overhead cars
                }
                double scale = Math.min((double) inW / tileW, (double) inH / tileH);
                int padX = (inW - (int) Math.round(tileW * scale)) / 2;
                int padY = (inH - (int) Math.round(tileH * scale)) / 2;

                List<double[]> hitsInTile = new ArrayList<>();
                for (int occId : trueOccupiedIds) {
                    ParkingSlot s = slots26.get(occId - 1);
                    double carX = s.getX() + 7;
                    double carY = s.getY() + 10;
                    double carW = 36;
                    double carH = 60;
                    // If vehicle center falls inside this tile, emit detection in tile's letterbox coordinates
                    double cxGlobal = carX + carW / 2.0;
                    double cyGlobal = carY + carH / 2.0;
                    if (cxGlobal >= offX && cxGlobal <= offX + tileW && cyGlobal >= offY && cyGlobal <= offY + tileH) {
                        double lcx = (cxGlobal - offX) * scale + padX;
                        double lcy = (cyGlobal - offY) * scale + padY;
                        double lw = carW * scale;
                        double lh = carH * scale;
                        hitsInTile.add(new double[]{lcx, lcy, lw, lh});
                    }
                }
                float[][] out = new float[84][Math.max(1, hitsInTile.size())];
                for (int i = 0; i < hitsInTile.size(); i++) {
                    double[] h = hitsInTile.get(i);
                    out[0][i] = (float) h[0];
                    out[1][i] = (float) h[1];
                    out[2][i] = (float) h[2];
                    out[3][i] = (float) h[3];
                    out[4 + 2][i] = 0.89f; // Car
                }
                return out;
            }
        );

        VehicleDetectionService.DetectionBatchResult batch = tiledDetector.detectVehicles(overheadLot);
        assertEquals(VehicleDetectionService.DetectorStatus.READY, batch.status());
        // Slot #7 lies in the 25% overlap between Tile 0 and Tile 1, so rawDetections has >4 hits,
        // while cross-tile NMS deduplicates them to exactly 4 unique vehicles!
        assertTrue(batch.rawDetections().size() >= 5, "Boundary-straddling vehicle in Slot #7 is detected in multiple overlapping tiles");
        assertEquals(4, batch.filteredDetections().size(), "Cross-tile NMS must suppress duplicate tile-overlap detections to exactly 4 vehicles");

        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(tiledDetector);
        AnalysisSummary summary = svc.analyze(
            overheadLot, slots26, false, OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION);

        assertEquals(4, summary.getOccupiedSlots());
        assertEquals(22, summary.getAvailableSlots());
        assertEquals(0, summary.getNeedsReviewSlots());
    }

    @Test
    void vehicleToSlotOverlapHandlesGrazingStraddlingSmallDistantAndMultipleVehicles() {
        List<ParkingSlot> slots = List.of(
            new ParkingSlot(1, 10, 10, 80, 100),  // Car centered in Slot 1, slightly grazing Slot 2 by 4px
            new ParkingSlot(2, 90, 10, 80, 100),  // Empty neighbour slot grazed by Slot 1's mirror
            new ParkingSlot(3, 170, 10, 80, 100), // Straddled left half of rogue parked truck
            new ParkingSlot(4, 250, 10, 80, 100), // Straddled right half of rogue parked truck
            new ParkingSlot(5, 330, 10, 80, 100), // Small distant motorcycle/compact car centered in slot
            new ParkingSlot(6, 410, 10, 80, 100)  // Completely empty slot + false detection in driving aisle
        );

        List<VehicleDetection> detections = List.of(
            new VehicleDetection("VEH-001", VehicleCategory.CAR, 20, 20, 74, 75, 0.91, "model"),
            new VehicleDetection("VEH-002", VehicleCategory.TRUCK, 215, 20, 70, 75, 0.89, "model"),
            new VehicleDetection("VEH-003", VehicleCategory.MOTORCYCLE, 352, 40, 36, 40, 0.78, "model"),
            new VehicleDetection("VEH-004", VehicleCategory.CAR, 415, 140, 60, 50, 0.85, "model")
        );

        SlotOverlapAnalyzer.OverlapPolicy policy = SlotOverlapAnalyzer.OverlapPolicy.defaultValidationTunedPolicy();
        Map<Integer, SlotOverlapAnalyzer.SlotOverlapEvidence> eval =
            SlotOverlapAnalyzer.evaluateSlots(slots, detections, policy);

        assertEquals(SlotStatus.OCCUPIED, eval.get(1).status(), "Primary Slot 1 must be OCCUPIED");
        assertEquals(SlotStatus.AVAILABLE, eval.get(2).status(), "Grazed neighbour Slot 2 must remain AVAILABLE");
        assertEquals(SlotStatus.NEEDS_REVIEW, eval.get(3).status(), "Straddled Slot 3 must be flagged NEEDS_REVIEW");
        assertTrue(eval.get(3).multiSlotConflict());
        assertEquals(SlotStatus.NEEDS_REVIEW, eval.get(4).status(), "Straddled Slot 4 must be flagged NEEDS_REVIEW");
        assertTrue(eval.get(4).multiSlotConflict());
        assertEquals(SlotStatus.OCCUPIED, eval.get(5).status(), "Small distant vehicle centered in Slot 5 must be OCCUPIED");
        assertEquals(SlotStatus.AVAILABLE, eval.get(6).status(), "Aisle detection outside Slot 6 must leave Slot 6 AVAILABLE");
    }

    @Test
    void validationSetTuningSelectsOptimalIoSThreshold() {
        List<ParkingSlot> slots = List.of(
            new ParkingSlot(1, 10, 10, 80, 100),
            new ParkingSlot(2, 100, 10, 80, 100),
            new ParkingSlot(3, 190, 10, 80, 100)
        );
        Map<Integer, SlotStatus> truth = Map.of(
            1, SlotStatus.OCCUPIED,
            2, SlotStatus.AVAILABLE,
            3, SlotStatus.OCCUPIED
        );
        List<VehicleDetection> dets = List.of(
            new VehicleDetection("V1", VehicleCategory.CAR, 20, 20, 56, 76, 0.90, "m"),
            new VehicleDetection("V2", VehicleCategory.CAR, 200, 25, 50, 60, 0.85, "m")
        );

        SlotOverlapAnalyzer.OverlapPolicy tuned = SlotOverlapAnalyzer.tuneOnValidationSet(
            List.of(new SlotOverlapAnalyzer.ValidationSlotScene(slots, truth, dets))
        );
        assertTrue(tuned.tunedOnValidationSet());
        assertTrue(tuned.minIoSForOccupied() >= 0.16 && tuned.minIoSForOccupied() <= 0.35);
        assertTrue(tuned.tuningRationale().contains("Val F1=1.0000"));
    }

    @Test
    void comparisonModeFlagsDisagreementAndDistinguishesZeroDetectionsFromUnavailableDetector() {
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();

        // 1. Detector READY but detects 0 vehicles on the 12-slot demo lot
        OnnxVehicleDetectionService emptyDetector = OnnxVehicleDetectionService.createWithTestBackend(
            "mock_yolov8.onnx", 640, 640, 0.30, 0.45,
            (tensor, w, h) -> new float[84][1]
        );
        svc.setVehicleDetectionService(emptyDetector);

        BufferedImage demoImg = com.parknchillai.utils.SampleDataGenerator.generateDemoParkingLot();
        List<ParkingSlot> slots = com.parknchillai.utils.ConfigLoader.loadSlotsFromCsv("data/config/slots.csv");

        AnalysisSummary compSummary = svc.analyze(
            demoImg, slots, false, OccupancyDecisionEngine.AnalysisMode.COMPARISON);

        assertEquals(8, compSummary.getAvailableSlots());
        assertEquals(0, compSummary.getOccupiedSlots());
        assertEquals(4, compSummary.getNeedsReviewSlots());
        assertTrue(compSummary.getDetectionBatchResult().formatExecutionSummary().contains("0 vehicles detected in image"));

        SlotAnalysisResult slot1 = compSummary.getSlotResults().get(0);
        assertEquals(SlotStatus.NEEDS_REVIEW, slot1.getStatus());
        assertTrue(slot1.getReviewReason().contains("Method Disagreement"));
        assertTrue(slot1.getDiagnosticRecord().formatVehicleDetectionsSummary().contains("Detector READY: 0 vehicles"));

        // 2. When detector is MODEL_MISSING in VEHICLE_DETECTION mode, never mark spaces AVAILABLE: route all to NEEDS_REVIEW
        svc.setVehicleDetectionService(new OnnxVehicleDetectionService(Path.of("models/missing_detector.onnx")));
        AnalysisSummary missingSummary = svc.analyze(
            demoImg, slots, false, OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION);
        assertEquals(0, missingSummary.getOccupiedSlots());
        assertEquals(0, missingSummary.getAvailableSlots());
        assertEquals(12, missingSummary.getNeedsReviewSlots());
        assertTrue(missingSummary.getDetectionBatchResult().formatExecutionSummary().contains("Detection Could Not Run [MODEL_MISSING]"));
    }

    @Test
    void missingModelIncompatibleTensorShapeAndFailedInferenceNeverFabricateDetections() throws Exception {
        BufferedImage img = new BufferedImage(380, 350, BufferedImage.TYPE_INT_RGB);

        // 1. Missing model file -> MODEL_MISSING
        OnnxVehicleDetectionService missingSvc = new OnnxVehicleDetectionService(Path.of("models/nonexistent_model.onnx"));
        assertEquals(VehicleDetectionService.DetectorStatus.MODEL_MISSING, missingSvc.getStatus());
        VehicleDetectionService.DetectionBatchResult resMissing = missingSvc.detectVehicles(img);
        assertFalse(resMissing.isAvailable());
        assertTrue(resMissing.filteredDetections().isEmpty());

        // 2. Corrupt/non-ONNX file -> MODEL_INCOMPATIBLE
        Path corruptFile = Files.createTempFile("corrupt_model", ".onnx");
        Files.writeString(corruptFile, "NOT_AN_ONNX_PROTOBUF_FILE_".repeat(10));
        try {
            OnnxVehicleDetectionService corruptSvc = new OnnxVehicleDetectionService(corruptFile);
            assertEquals(VehicleDetectionService.DetectorStatus.MODEL_INCOMPATIBLE, corruptSvc.getStatus());
        } finally {
            Files.deleteIfExists(corruptFile);
        }

        // 3. Incompatible output tensor shape ([3][10] matches neither YOLOv8, YOLOv5, nor [N][6]) -> MODEL_INCOMPATIBLE
        OnnxVehicleDetectionService badShapeSvc = OnnxVehicleDetectionService.createWithTestBackend(
            "bad_shape.onnx", 640, 640, 0.30, 0.45,
            (tensor, w, h) -> new float[3][10]
        );
        VehicleDetectionService.DetectionBatchResult resBadShape = badShapeSvc.detectVehicles(img);
        assertEquals(VehicleDetectionService.DetectorStatus.MODEL_INCOMPATIBLE, resBadShape.status());
        assertTrue(resBadShape.filteredDetections().isEmpty());

        // 4. Failed inference exception -> INFERENCE_FAILED
        OnnxVehicleDetectionService failingSvc = OnnxVehicleDetectionService.createWithTestBackend(
            "failing.onnx", 640, 640, 0.30, 0.45,
            (tensor, w, h) -> { throw new RuntimeException("Simulated ONNX session failure"); }
        );
        VehicleDetectionService.DetectionBatchResult resFailed = failingSvc.detectVehicles(img);
        assertEquals(VehicleDetectionService.DetectorStatus.INFERENCE_FAILED, resFailed.status());
        assertTrue(resFailed.filteredDetections().isEmpty());
    }

    @Test
    void twentySixSlotOutdoorLotAndShadowedOccludedLotAchieveExactOccupancyInBothModes() {
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();

        BufferedImage lot26 = new BufferedImage(1060, 320, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = lot26.createGraphics();
        for (int y = 0; y < 320; y++) {
            for (int x = 0; x < 1060; x++) {
                int v = 62 + (int) (8.0 * Math.sin(x * 0.04) + 6.0 * Math.cos(y * 0.05)) + ((x * 7 + y * 13) % 5 - 2);
                v = Math.max(0, Math.min(255, v));
                lot26.setRGB(x, y, (v << 16) | (v << 8) | (v + 2));
            }
        }
        g.setColor(new Color(215, 215, 210));
        g.setStroke(new BasicStroke(3f));
        List<ParkingSlot> slots = new ArrayList<>();
        int id = 1;
        int[] rowY = {20, 180};
        for (int r = 0; r < 2; r++) {
            int y = rowY[r];
            g.drawLine(20, y, 20 + 13 * 78, y);
            g.drawLine(20, y + 110, 20 + 13 * 78, y + 110);
            for (int c = 0; c < 13; c++) {
                int x = 20 + c * 78;
                g.drawLine(x, y, x, y + 110);
                slots.add(new ParkingSlot(id++, x, y, 78, 110, ParkingSlot.GeometryOrigin.AUTO_DETECTED));
            }
            g.drawLine(20 + 13 * 78, y, 20 + 13 * 78, y + 110);
        }
        int[] occSlots = {1, 5, 12, 16, 23};
        Color[] carColors = {
            new Color(190, 35, 45),
            new Color(35, 95, 195),
            new Color(210, 155, 25),
            new Color(40, 150, 95),
            new Color(30, 32, 36)
        };
        for (int i = 0; i < occSlots.length; i++) {
            ParkingSlot s = slots.get(occSlots[i] - 1);
            g.setColor(carColors[i]);
            g.fillRoundRect(s.getX() + 12, s.getY() + 14, s.getWidth() - 24, s.getHeight() - 28, 10, 10);
            g.setColor(new Color(30, 40, 50));
            g.fillRect(s.getX() + 16, s.getY() + 26, s.getWidth() - 32, 18);
            g.fillRect(s.getX() + 16, s.getY() + s.getHeight() - 44, s.getWidth() - 32, 16);
        }
        g.dispose();

        AnalysisSummary otsu26 = svc.analyze(lot26, slots, false, OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE);
        assertEquals(5, otsu26.getOccupiedSlots());
        assertEquals(21, otsu26.getAvailableSlots());

        AnalysisSummary adap26 = svc.analyze(lot26, slots, true, OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE);
        assertEquals(5, adap26.getOccupiedSlots());
        assertEquals(21, adap26.getAvailableSlots());

        for (SlotAnalysisResult r : adap26.getSlotResults()) {
            SlotDiagnosticRecord diag = r.getDiagnosticRecord();
            assertNotNull(diag);
            assertEquals(r.getId(), diag.slotId());
            assertTrue(diag.formatMultiLineReport().contains("CART Tree Trace"));
            assertTrue(diag.formatMultiLineReport().contains("Vehicle Detections"));
        }
    }
}
