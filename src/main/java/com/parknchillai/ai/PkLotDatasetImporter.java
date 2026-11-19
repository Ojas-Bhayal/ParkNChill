package com.parknchillai.ai;

import com.parknchillai.detection.SlotOverlapAnalyzer;
import com.parknchillai.detection.VehicleDetectionService;
import com.parknchillai.features.FeatureExtractor;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.imageprocessing.EnhancedImage;
import com.parknchillai.imageprocessing.ImageEnhancementService;
import com.parknchillai.imageprocessing.OriginalImage;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.RoiExtractor;
import com.parknchillai.parking.SlotStatus;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import javax.imageio.ImageIO;
import javax.xml.parsers.DocumentBuilderFactory;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;
import java.util.function.DoubleConsumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Partition-aware PKLot dataset importer and read-only annotation inspector
 * (ParkNChill v2.3 Requirements 2, 3, 4 &amp; 5):
 * <ul>
 *   <li><strong>Explicit {@code IMPORTED_GROUND_TRUTH} Provenance</strong>:
 *       Imported records carry {@code provenance = IMPORTED_GROUND_TRUTH},
 *       {@code sourceDataset = "PKLot"}, {@code annotationFilePath}, {@code sourceImageId},
 *       {@code sourceImageSha256}, and {@code reviewerId = ""} (never marked {@code HUMAN_VERIFIED}
 *       and never assigned a fabricated reviewer ID such as {@code PKLOT_ANNOTATION}).</li>
 *   <li><strong>Actual Polygon / Rotated Quadrilateral Geometry Preservation</strong>:
 *       {@link #parsePkLotXml(Path)} preserves the original 4-point or multi-point {@code <contour>}
 *       vertices and {@code <rotatedRect>} angle in {@link ParkingSlot} ({@link ParkingSlot#hasCustomPolygon()}),
 *       enabling exact Sutherland-Hodgman polygon vehicle overlap in {@link SlotOverlapAnalyzer}
 *       and perspective-rectified ROI feature extraction in {@link RoiExtractor}.</li>
 *   <li><strong>Cancellation &amp; Non-Destructive Staging</strong>:
 *       Supports cooperative cancellation via {@link BooleanSupplier} so a cancelled or failed
 *       import never partially overwrites an active dataset.</li>
 * </ul>
 */
public final class PkLotDatasetImporter {

    private static final Pattern DATE_PATTERN = Pattern.compile("(20\\d{2}[-_]\\d{2}[-_]\\d{2})");

    private PkLotDatasetImporter() {}

    /**
     * Resolves a default PKLot directory on the user's workstation (checking {@code -DPKLOT_ROOT},
     * {@code PKLOT_ROOT} environment variable, and {@code ~/Desktop/PKLot} or {@code ~/Desktop})
     * if present on disk.
     */
    public static Path resolveDefaultPkLotDirectory() {
        String sysProp = System.getProperty("PKLOT_ROOT");
        if (sysProp != null && !sysProp.isBlank()) {
            Path p = Path.of(sysProp.trim());
            if (Files.exists(p) && Files.isDirectory(p)) return p;
        }
        String env = System.getenv("PKLOT_ROOT");
        if (env != null && !env.isBlank()) {
            Path p = Path.of(env.trim());
            if (Files.exists(p) && Files.isDirectory(p)) return p;
        }
        String userHome = System.getProperty("user.home");
        if (userHome != null && !userHome.isBlank()) {
            for (String candidateRel : new String[]{"Desktop/PKLot", "Desktop/pklot", "Desktop/PKLot_Dataset", "Desktop"}) {
                Path candidate = Path.of(userHome, candidateRel.split("/"));
                if (Files.exists(candidate) && Files.isDirectory(candidate)) {
                    return candidate;
                }
            }
        }
        return null;
    }

    /**
     * Parsed single parking-space annotation from a PKLot XML or CSV sidecar file,
     * preserving both the enclosing bounding box and the exact polygon/quadrilateral contour.
     */
    public record ParsedSlotAnnotation(
            int slotId,
            int x,
            int y,
            int width,
            int height,
            List<double[]> polygonVertices,
            Double rotationAngleDegrees,
            ParkingSlot.GeometryOrigin geometryOrigin,
            SlotStatus groundTruthLabel,
            boolean valid,
            String invalidReason
    ) {
        public ParkingSlot toParkingSlot() {
            if (polygonVertices != null && polygonVertices.size() >= 3) {
                return ParkingSlot.fromPolygon(slotId, polygonVertices, geometryOrigin, rotationAngleDegrees);
            }
            return new ParkingSlot(slotId, x, y, width, height,
                geometryOrigin != null ? geometryOrigin : ParkingSlot.GeometryOrigin.CSV_CONFIG);
        }
    }

    /**
     * Read-only inspection summary returned by {@link #inspectSampleAnnotations(Path, int)}.
     */
    public record AnnotationInspectionReport(
            Path rootDirectory,
            int annotationFilesFound,
            int pairedImagesFound,
            int totalSlotsInspected,
            int occupiedAnnotations,
            int availableAnnotations,
            int invalidAnnotations,
            int polygonContourCount,
            List<String> sampleEntries,
            String conciseSummary
    ) {}

    /**
     * Result of importing a PKLot directory via {@link #importDirectory}.
     */
    public record PkLotImportResult(
            Path rootDirectory,
            int validImageCount,
            int validSlotCount,
            int occupiedCount,
            int availableCount,
            int invalidRecordCount,
            int detectorFailureCount,
            List<SlotTrainingRecord> eligibleRecords,
            Map<String, String> groupPartitionMap,
            String conciseSummary,
            String recommendedNextAction
    ) {
        public boolean hasEligibleRecords() {
            return eligibleRecords != null && !eligibleRecords.isEmpty()
                && occupiedCount > 0 && availableCount > 0;
        }
    }

    /**
     * Read-only inspection function for {@code Sample Annotations...} (Requirement 5).
     * Never fabricates training features or adds records to the training dataset.
     */
    public static AnnotationInspectionReport inspectSampleAnnotations(Path rootDir, int maxSamplesToList) throws IOException {
        if (rootDir == null || !Files.exists(rootDir)) {
            throw new IllegalArgumentException("PKLot directory does not exist: " + rootDir);
        }
        List<Path> annotationFiles = findAnnotationFiles(rootDir);
        int pairedImages = 0;
        int totalSlots = 0;
        int occCount = 0;
        int availCount = 0;
        int invalidCount = 0;
        int polyCount = 0;
        List<String> sampleLines = new ArrayList<>();

        for (Path annFile : annotationFiles) {
            Path imgPath = resolvePairedImage(annFile);
            boolean hasImg = imgPath != null && Files.exists(imgPath);
            if (hasImg) pairedImages++;
            String captureGroup = deriveCaptureGroup(rootDir, annFile);

            List<ParsedSlotAnnotation> parsed = parseAnnotationFile(annFile);
            for (ParsedSlotAnnotation slot : parsed) {
                totalSlots++;
                if (!slot.valid() || (slot.groundTruthLabel() != SlotStatus.OCCUPIED && slot.groundTruthLabel() != SlotStatus.AVAILABLE)) {
                    invalidCount++;
                    continue;
                }
                if (slot.groundTruthLabel() == SlotStatus.OCCUPIED) occCount++;
                else availCount++;
                if (slot.polygonVertices() != null && slot.polygonVertices().size() >= 3) {
                    polyCount++;
                }

                if (sampleLines.size() < Math.max(1, maxSamplesToList)) {
                    ParkingSlot ps = slot.toParkingSlot();
                    sampleLines.add(String.format(Locale.ROOT,
                        "%s | Group=%s | Slot #%d (%d,%d,%dx%d, poly=%s) | Label=%s | Image=%s",
                        annFile.getFileName(), captureGroup, slot.slotId(),
                        slot.x(), slot.y(), slot.width(), slot.height(),
                        ps.hasCustomPolygon() ? "ANGLED_QUAD" : "RECT",
                        slot.groundTruthLabel(), hasImg ? imgPath.getFileName().toString() : "MISSING"));
                }
            }
        }

        String summary = String.format(Locale.ROOT,
            "Inspected %d annotation file(s) (%d paired image(s)): %d valid slot(s) (%d Occupied, %d Available; %d polygon contours), %d invalid record(s). "
            + "[Read-only inspection: no training features were fabricated.]",
            annotationFiles.size(), pairedImages, occCount + availCount, occCount, availCount, polyCount, invalidCount);

        return new AnnotationInspectionReport(
            rootDir,
            annotationFiles.size(),
            pairedImages,
            totalSlots,
            occCount,
            availCount,
            invalidCount,
            polyCount,
            Collections.unmodifiableList(sampleLines),
            summary
        );
    }

    /**
     * Imports a PKLot directory using genuine decoded images, real 8-feature IVP extraction
     * (with perspective rectification on angled quadrilateral contours), and live
     * {@link VehicleDetectionService} inference to build verified 10-feature
     * ({@code v2.3-10feat-enhanced-v1}) {@link SlotTrainingRecord}s with {@code IMPORTED_GROUND_TRUTH}
     * provenance while preserving camera/date group partitions.
     */
    public static PkLotImportResult importDirectory(
            Path rootDir,
            VehicleDetectionService detector,
            SlotOverlapAnalyzer.OverlapPolicy overlapPolicy,
            boolean useAdaptiveThreshold,
            String ignoredReviewerId,
            DoubleConsumer progressCallback) throws IOException {
        return importDirectory(rootDir, detector, overlapPolicy, useAdaptiveThreshold, progressCallback, () -> false);
    }

    /**
     * Cancellable overload of {@link #importDirectory} supporting cooperative background cancellation
     * (Requirement 4).
     */
    public static PkLotImportResult importDirectory(
            Path rootDir,
            VehicleDetectionService detector,
            SlotOverlapAnalyzer.OverlapPolicy overlapPolicy,
            boolean useAdaptiveThreshold,
            DoubleConsumer progressCallback,
            BooleanSupplier isCancelled) throws IOException {
        if (rootDir == null || !Files.exists(rootDir)) {
            throw new IllegalArgumentException("PKLot directory does not exist: " + rootDir);
        }
        SlotOverlapAnalyzer.OverlapPolicy activePolicy = overlapPolicy != null
            ? overlapPolicy
            : SlotOverlapAnalyzer.OverlapPolicy.defaultValidationTunedPolicy();

        List<Path> annotationFiles = findAnnotationFiles(rootDir);
        if (annotationFiles.isEmpty()) {
            String msg = "No PKLot XML or CSV annotation files found in " + rootDir.getFileName();
            return new PkLotImportResult(
                rootDir, 0, 0, 0, 0, 0, 0, List.of(), Map.of(),
                msg, "Select a folder containing PKLot .xml/.csv annotations paired with .jpg/.png parking images."
            );
        }

        List<String> distinctGroups = annotationFiles.stream()
            .map(p -> deriveCaptureGroup(rootDir, p))
            .distinct()
            .sorted()
            .collect(Collectors.toList());
        Map<String, String> groupPartitionMap = assignCameraDatePartitions(distinctGroups);

        int validImages = 0;
        int validSlots = 0;
        int occupiedCount = 0;
        int availableCount = 0;
        int invalidRecords = 0;
        int detectorFailures = 0;
        List<SlotTrainingRecord> eligibleRecords = new ArrayList<>();

        int totalFiles = annotationFiles.size();
        for (int idx = 0; idx < totalFiles; idx++) {
            if (isCancelled != null && isCancelled.getAsBoolean()) {
                throw new CancellationException("PKLot import cancelled by user after " + idx + " of " + totalFiles + " file(s).");
            }
            Path annFile = annotationFiles.get(idx);
            if (progressCallback != null) {
                progressCallback.accept((double) idx / totalFiles);
            }

            List<ParsedSlotAnnotation> parsedSlots = parseAnnotationFile(annFile);
            Path imgPath = resolvePairedImage(annFile);
            if (imgPath == null || !Files.exists(imgPath)) {
                invalidRecords += Math.max(1, parsedSlots.size());
                continue;
            }

            BufferedImage img;
            String imgSha256;
            try {
                img = ImageIO.read(imgPath.toFile());
                if (img == null || img.getWidth() < 10 || img.getHeight() < 10) {
                    invalidRecords += Math.max(1, parsedSlots.size());
                    continue;
                }
                imgSha256 = DatasetManifest.computeFileSha256OrEmpty(imgPath);
            } catch (Exception ex) {
                invalidRecords += Math.max(1, parsedSlots.size());
                continue;
            }

            List<ParkingSlot> inBoundsSlots = new ArrayList<>();
            Map<Integer, SlotStatus> groundTruthBySlot = new LinkedHashMap<>();

            for (ParsedSlotAnnotation psa : parsedSlots) {
                if (!psa.valid()
                        || (psa.groundTruthLabel() != SlotStatus.OCCUPIED && psa.groundTruthLabel() != SlotStatus.AVAILABLE)) {
                    invalidRecords++;
                    continue;
                }
                ParkingSlot slot;
                try {
                    slot = psa.toParkingSlot();
                } catch (Exception ex) {
                    invalidRecords++;
                    continue;
                }
                if (slot.getX() < 0 || slot.getY() < 0
                        || slot.getWidth() < 8 || slot.getHeight() < 8
                        || slot.getX() + slot.getWidth() > img.getWidth()
                        || slot.getY() + slot.getHeight() > img.getHeight()
                        || groundTruthBySlot.containsKey(slot.getId())) {
                    invalidRecords++;
                    continue;
                }
                inBoundsSlots.add(slot);
                groundTruthBySlot.put(slot.getId(), psa.groundTruthLabel());
            }

            if (inBoundsSlots.isEmpty()) {
                continue;
            }

            // Run live ONNX vehicle detection on the image (never fabricate detections!)
            VehicleDetectionService.DetectionBatchResult detBatch = detector != null
                ? detector.detectVehicles(img)
                : new VehicleDetectionService.DetectionBatchResult(
                    VehicleDetectionService.DetectorStatus.MODEL_MISSING,
                    "unconfigured", "", "Detector is null", List.of(), List.of(), 0L);

            if (!detBatch.isAvailable()) {
                detectorFailures++;
                continue;
            }

            validImages++;
            Map<Integer, SlotOverlapAnalyzer.SlotOverlapEvidence> overlapMap =
                SlotOverlapAnalyzer.evaluateSlots(inBoundsSlots, detBatch.filteredDetections(), activePolicy);

            String captureGroup = deriveCaptureGroup(rootDir, annFile);
            String targetPartition = groupPartitionMap.getOrDefault(captureGroup, "TRAIN");
            String sourceImageId = rootDir.relativize(imgPath).toString().replace('\\', '/');
            String relAnnPath = rootDir.relativize(annFile).toString().replace('\\', '/');

            OriginalImage origImg = new OriginalImage(img);
            EnhancedImage enhancedImg = ImageEnhancementService.getDefault().enhance(origImg);

            for (ParkingSlot slot : inBoundsSlots) {
                SlotStatus gt = groundTruthBySlot.get(slot.getId());
                BufferedImage rawRoi = RoiExtractor.extract(origImg.getInternalBuffer(), slot);
                BufferedImage enhancedRoi = RoiExtractor.extract(enhancedImg.getInternalBuffer(), slot);
                FeatureExtractor.ProcessedRoi proc = FeatureExtractor.processWithStages(rawRoi, enhancedRoi, useAdaptiveThreshold);
                SlotOverlapAnalyzer.SlotOverlapEvidence ev = overlapMap.get(slot.getId());

                double ios = ev != null ? ev.bestIoS() : 0.0;
                double iou = ev != null ? ev.bestIou() : 0.0;
                boolean centerIn = ev != null && ev.centerInsideSlot();
                boolean matched = ev != null && (ev.status() == SlotStatus.OCCUPIED
                    || (!ev.contributingDetections().isEmpty() && ios >= activePolicy.minIoSForOccupied()));

                FeatureVector fv10 = proc.getFeatures().withDetectionFeatures(ios, matched);
                String detEv = (ev == null || ev.contributingDetections().isEmpty())
                    ? "NONE"
                    : ev.assignedVehicleId();

                SlotTrainingRecord rec = SlotTrainingRecord.createImportedGroundTruthRecord(
                    "PKLot",
                    sourceImageId,
                    relAnnPath,
                    imgSha256,
                    captureGroup,
                    slot,
                    detEv,
                    ios,
                    iou,
                    centerIn,
                    matched,
                    fv10,
                    gt,
                    "PKLot-ONNX-v2.3",
                    targetPartition
                );

                eligibleRecords.add(rec);
                validSlots++;
                if (gt == SlotStatus.OCCUPIED) occupiedCount++;
                else availableCount++;
            }
        }

        if (progressCallback != null) {
            progressCallback.accept(1.0);
        }

        Set<String> eligibleTrainGroups = new java.util.LinkedHashSet<>();
        Set<String> eligibleValidationGroups = new java.util.LinkedHashSet<>();
        Set<String> eligibleTestGroups = new java.util.LinkedHashSet<>();
        int eligibleTrainOccupied = 0;
        int eligibleTrainAvailable = 0;
        for (SlotTrainingRecord record : eligibleRecords) {
            String part = record.partition() == null ? "" : record.partition().trim().toUpperCase(Locale.ROOT);
            if ("TRAIN".equals(part)) {
                eligibleTrainGroups.add(record.captureGroup());
                if (record.occupancyLabel() == SlotStatus.OCCUPIED) eligibleTrainOccupied++;
                if (record.occupancyLabel() == SlotStatus.AVAILABLE) eligibleTrainAvailable++;
            } else if ("VALIDATION".equals(part) || "VAL".equals(part)) {
                eligibleValidationGroups.add(record.captureGroup());
            } else if ("TEST".equals(part)) {
                eligibleTestGroups.add(record.captureGroup());
            }
        }
        int eligiblePartitionGroups = eligibleTrainGroups.size() + eligibleValidationGroups.size() + eligibleTestGroups.size();

        String summary = String.format(Locale.ROOT,
            "PKLot Import Summary: %d valid image(s), %d valid 10-feature slot(s) (%d Occupied, %d Available; Provenance=IMPORTED_GROUND_TRUTH), %d invalid record(s), %d detector failure(s). "
                + "Capture groups discovered=%d; eligible partitions: TRAIN=%d, VALIDATION=%d, TEST=%d.",
            validImages, validSlots, occupiedCount, availableCount, invalidRecords, detectorFailures,
            distinctGroups.size(), eligibleTrainGroups.size(), eligibleValidationGroups.size(), eligibleTestGroups.size());

        String nextAction;
        if (detectorFailures > 0 && validSlots == 0) {
            nextAction = "Vehicle detector was unavailable (" + detectorFailures
                + " image(s) failed inference). Install a compatible models/vehicle_detector.onnx model and re-import.";
        } else if (eligibleTrainGroups.size() > 0 && eligibleValidationGroups.size() > 0
                && eligibleTestGroups.size() > 0 && eligibleTrainOccupied > 0 && eligibleTrainAvailable > 0
                && eligiblePartitionGroups >= 3 && validSlots >= 6) {
            nextAction = "Ready for partition-aware training. Click 'Train Decision Tree' (TRAIN only), select settings on VALIDATION, and evaluate once on the frozen TEST partition.";
        } else {
            nextAction = String.format(Locale.ROOT,
                "Not ready to train: usable groups are TRAIN=%d, VALIDATION=%d, TEST=%d; TRAIN classes are Occupied=%d, Available=%d. "
                    + "In Import PKLot, select the TOP-LEVEL extracted PKLot folder—not one weather/date folder. Include multiple dates and camera folders (e.g. PUCPR, UFPR04, UFPR05), then re-import.",
                eligibleTrainGroups.size(), eligibleValidationGroups.size(), eligibleTestGroups.size(),
                eligibleTrainOccupied, eligibleTrainAvailable);
        }

        return new PkLotImportResult(
            rootDir,
            validImages,
            validSlots,
            occupiedCount,
            availableCount,
            invalidRecords,
            detectorFailures,
            Collections.unmodifiableList(eligibleRecords),
            Collections.unmodifiableMap(groupPartitionMap),
            summary,
            nextAction
        );
    }

    public static Map<String, String> assignCameraDatePartitions(List<String> sortedGroups) {
        Map<String, String> map = new LinkedHashMap<>();
        int n = sortedGroups.size();
        if (n == 0) return map;
        if (n == 1) {
            map.put(sortedGroups.get(0), "TRAIN");
            return map;
        }
        if (n == 2) {
            map.put(sortedGroups.get(0), "TRAIN");
            map.put(sortedGroups.get(1), "TEST");
            return map;
        }
        // Interleave sorted camera/date groups across partitions rather than placing the
        // alphabetically first dates only in TRAIN and the last camera's dates only in TEST.
        // Each whole group remains indivisible, preventing slots from a scene/date leaking.
        String[] partitionCycle = {"TRAIN", "VALIDATION", "TEST"};
        for (int i = 0; i < n; i++) {
            map.put(sortedGroups.get(i), partitionCycle[i % partitionCycle.length]);
        }
        return map;
    }

    static String deriveCaptureGroup(Path rootDir, Path annFile) {
        Path rel = rootDir.relativize(annFile);
        String relStr = rel.toString().replace('\\', '/');
        String upper = relStr.toUpperCase(Locale.ROOT);
        String camera = "CAM";
        if (upper.contains("UFPR04")) camera = "UFPR04";
        else if (upper.contains("UFPR05")) camera = "UFPR05";
        else if (upper.contains("PUCPR")) camera = "PUCPR";
        else if (rel.getParent() != null && rel.getParent().getFileName() != null) {
            camera = rel.getParent().getFileName().toString().replaceAll("[^a-zA-Z0-9_-]", "_");
        }

        Matcher m = DATE_PATTERN.matcher(relStr);
        if (m.find()) {
            return "PKLOT_" + camera + "_" + m.group(1).replace('_', '-');
        }
        String baseName = stripExtension(annFile.getFileName().toString());
        int underscore = baseName.indexOf('_');
        String prefix = underscore > 0 ? baseName.substring(0, underscore) : baseName;
        return "PKLOT_" + camera + "_" + prefix;
    }

    private static List<Path> findAnnotationFiles(Path rootDir) throws IOException {
        try (Stream<Path> walk = Files.walk(rootDir)) {
            return walk.filter(Files::isRegularFile)
                .filter(p -> {
                    String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
                    return name.endsWith(".xml") || (name.endsWith(".csv") && !name.equals("features.csv") && !name.endsWith("manifest.csv"));
                })
                .sorted(Comparator.comparing(Path::toString))
                .collect(Collectors.toList());
        }
    }

    private static Path resolvePairedImage(Path annFile) {
        String base = stripExtension(annFile.getFileName().toString());
        Path parent = annFile.getParent();
        if (parent == null) return null;
        for (String ext : new String[]{".jpg", ".png", ".jpeg", ".bmp", ".JPG", ".PNG"}) {
            Path candidate = parent.resolve(base + ext);
            if (Files.exists(candidate)) return candidate;
        }
        return null;
    }

    private static String stripExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    public static List<ParsedSlotAnnotation> parseAnnotationFile(Path file) {
        String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".xml")) {
            return parsePkLotXml(file);
        } else {
            return parsePkLotCsv(file);
        }
    }

    /**
     * Parses PKLot XML files while preserving the exact multi-point {@code <contour>} polygon
     * vertices and {@code <rotatedRect>} angle (Requirement 5).
     */
    public static List<ParsedSlotAnnotation> parsePkLotXml(Path xmlFile) {
        List<ParsedSlotAnnotation> out = new ArrayList<>();
        try {
            DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
            dbf.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            Document doc = dbf.newDocumentBuilder().parse(xmlFile.toFile());
            NodeList spaces = doc.getElementsByTagName("space");
            for (int i = 0; i < spaces.getLength(); i++) {
                Element sp = (Element) spaces.item(i);
                String idStr = sp.getAttribute("id");
                String occStr = sp.getAttribute("occupied");
                if (idStr == null || idStr.isBlank() || occStr == null || occStr.isBlank()) {
                    out.add(new ParsedSlotAnnotation(
                        i + 1, 0, 0, 0, 0, List.of(), null,
                        ParkingSlot.GeometryOrigin.PKLOT_POLYGON, SlotStatus.NEEDS_REVIEW, false,
                        "Missing id or occupied attribute"));
                    continue;
                }
                int id = Integer.parseInt(idStr.trim());
                int occFlag = Integer.parseInt(occStr.trim());
                SlotStatus status = occFlag == 1
                    ? SlotStatus.OCCUPIED
                    : (occFlag == 0 ? SlotStatus.AVAILABLE : SlotStatus.NEEDS_REVIEW);

                Double angleDeg = null;
                NodeList angles = sp.getElementsByTagName("angle");
                if (angles.getLength() > 0) {
                    String dAttr = ((Element) angles.item(0)).getAttribute("d");
                    if (dAttr != null && !dAttr.isBlank()) {
                        angleDeg = Double.parseDouble(dAttr.trim());
                    }
                }

                List<double[]> polygonVertices = new ArrayList<>();
                ParkingSlot.GeometryOrigin geomOrigin = ParkingSlot.GeometryOrigin.PKLOT_POLYGON;

                NodeList points = sp.getElementsByTagName("point");
                if (points.getLength() >= 3) {
                    for (int p = 0; p < points.getLength(); p++) {
                        Element pt = (Element) points.item(p);
                        double px = Double.parseDouble(pt.getAttribute("x").trim());
                        double py = Double.parseDouble(pt.getAttribute("y").trim());
                        polygonVertices.add(new double[]{px, py});
                    }
                } else {
                    NodeList centers = sp.getElementsByTagName("center");
                    NodeList sizes = sp.getElementsByTagName("size");
                    if (centers.getLength() > 0 && sizes.getLength() > 0) {
                        Element cEl = (Element) centers.item(0);
                        Element sEl = (Element) sizes.item(0);
                        double cx = Double.parseDouble(cEl.getAttribute("x").trim());
                        double cy = Double.parseDouble(cEl.getAttribute("y").trim());
                        double sw = Double.parseDouble(sEl.getAttribute("w").trim());
                        double sh = Double.parseDouble(sEl.getAttribute("h").trim());
                        double deg = angleDeg != null ? angleDeg : 0.0;
                        polygonVertices = computeRotatedRectCorners(cx, cy, sw, sh, deg);
                        geomOrigin = ParkingSlot.GeometryOrigin.PKLOT_ROTATED_RECT;
                    }
                }

                if (polygonVertices.size() < 3) {
                    out.add(new ParsedSlotAnnotation(
                        id, 0, 0, 0, 0, List.of(), angleDeg, geomOrigin, status, false,
                        "Insufficient contour or rotatedRect coordinates"));
                    continue;
                }

                double minX = Double.POSITIVE_INFINITY, minY = Double.POSITIVE_INFINITY;
                double maxX = Double.NEGATIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
                for (double[] pt : polygonVertices) {
                    minX = Math.min(minX, pt[0]);
                    minY = Math.min(minY, pt[1]);
                    maxX = Math.max(maxX, pt[0]);
                    maxY = Math.max(maxY, pt[1]);
                }
                int bx = (int) Math.floor(minX);
                int by = (int) Math.floor(minY);
                int bw = (int) Math.ceil(maxX) - bx;
                int bh = (int) Math.ceil(maxY) - by;

                boolean valid = (status == SlotStatus.OCCUPIED || status == SlotStatus.AVAILABLE)
                    && bx >= 0 && by >= 0 && bw >= 8 && bh >= 8
                    && ParkingSlot.computePolygonArea(polygonVertices) >= 32.0;
                out.add(new ParsedSlotAnnotation(
                    id, bx, by, bw, bh, Collections.unmodifiableList(polygonVertices),
                    angleDeg, geomOrigin, status, valid, valid ? "" : "Invalid coordinates or label"
                ));
            }
        } catch (Exception ex) {
            out.add(new ParsedSlotAnnotation(
                0, 0, 0, 0, 0, List.of(), null, ParkingSlot.GeometryOrigin.PKLOT_POLYGON,
                SlotStatus.NEEDS_REVIEW, false, "XML parse error: " + ex.getMessage()));
        }
        return out;
    }

    private static List<double[]> computeRotatedRectCorners(double cx, double cy, double w, double h, double angleDegrees) {
        double rad = Math.toRadians(angleDegrees);
        double cos = Math.cos(rad);
        double sin = Math.sin(rad);
        double hw = w / 2.0;
        double hh = h / 2.0;
        double[][] local = new double[][]{
            {-hw, -hh},
            { hw, -hh},
            { hw,  hh},
            {-hw,  hh}
        };
        List<double[]> corners = new ArrayList<>(4);
        for (double[] lc : local) {
            double rx = cx + lc[0] * cos - lc[1] * sin;
            double ry = cy + lc[0] * sin + lc[1] * cos;
            corners.add(new double[]{rx, ry});
        }
        return corners;
    }

    private static List<ParsedSlotAnnotation> parsePkLotCsv(Path csvFile) {
        List<ParsedSlotAnnotation> out = new ArrayList<>();
        try {
            List<String> lines = Files.readAllLines(csvFile, StandardCharsets.UTF_8);
            for (String raw : lines) {
                String line = raw.trim();
                if (line.isEmpty() || line.startsWith("#")
                        || line.toLowerCase(Locale.ROOT).startsWith("id,")
                        || line.toLowerCase(Locale.ROOT).startsWith("slot_id,")) {
                    continue;
                }
                String[] parts = line.split(",");
                if (parts.length < 6) {
                    out.add(new ParsedSlotAnnotation(
                        0, 0, 0, 0, 0, List.of(), null, ParkingSlot.GeometryOrigin.CSV_CONFIG,
                        SlotStatus.NEEDS_REVIEW, false, "Expected 6 CSV columns"));
                    continue;
                }
                try {
                    int id = Integer.parseInt(parts[0].trim());
                    int x = Integer.parseInt(parts[1].trim());
                    int y = Integer.parseInt(parts[2].trim());
                    int w = Integer.parseInt(parts[3].trim());
                    int h = Integer.parseInt(parts[4].trim());
                    String rawLbl = parts[5].trim();
                    SlotStatus st;
                    if ("1".equals(rawLbl) || "OCCUPIED".equalsIgnoreCase(rawLbl)) {
                        st = SlotStatus.OCCUPIED;
                    } else if ("0".equals(rawLbl) || "AVAILABLE".equalsIgnoreCase(rawLbl) || "EMPTY".equalsIgnoreCase(rawLbl)) {
                        st = SlotStatus.AVAILABLE;
                    } else {
                        out.add(new ParsedSlotAnnotation(
                            id, x, y, w, h, List.of(), null, ParkingSlot.GeometryOrigin.CSV_CONFIG,
                            SlotStatus.NEEDS_REVIEW, false, "Invalid occupancy label: " + rawLbl));
                        continue;
                    }
                    boolean valid = id > 0 && x >= 0 && y >= 0 && w >= 8 && h >= 8;
                    List<double[]> rectCorners = List.of(
                        new double[]{x, y},
                        new double[]{x + w, y},
                        new double[]{x + w, y + h},
                        new double[]{x, y + h}
                    );
                    out.add(new ParsedSlotAnnotation(
                        id, x, y, w, h, rectCorners, 0.0, ParkingSlot.GeometryOrigin.CSV_CONFIG,
                        st, valid, valid ? "" : "Out-of-range coordinates"));
                } catch (NumberFormatException nfe) {
                    out.add(new ParsedSlotAnnotation(
                        0, 0, 0, 0, 0, List.of(), null, ParkingSlot.GeometryOrigin.CSV_CONFIG,
                        SlotStatus.NEEDS_REVIEW, false, "Malformed number"));
                }
            }
        } catch (IOException ex) {
            out.add(new ParsedSlotAnnotation(
                0, 0, 0, 0, 0, List.of(), null, ParkingSlot.GeometryOrigin.CSV_CONFIG,
                SlotStatus.NEEDS_REVIEW, false, "CSV read error"));
        }
        return out;
    }
}
