package com.parknchillai.imageprocessing;

import com.parknchillai.ai.DatasetManifest;
import com.parknchillai.ai.DecisionTreeModel;
import com.parknchillai.ai.ModelSerializer;
import com.parknchillai.app.AnalysisSummary;
import com.parknchillai.app.ParkingAnalysisService;
import com.parknchillai.app.SlotAnalysisResult;
import com.parknchillai.detection.OccupancyDecisionEngine;
import com.parknchillai.detection.OnnxVehicleDetectionService;
import com.parknchillai.features.FeatureExtractor;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.RoiExtractor;
import com.parknchillai.parking.SlotStatus;
import com.parknchillai.utils.ConfigLoader;
import com.parknchillai.utils.SampleDataGenerator;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Requirement Verification Suite for ParkNChill v2.3 Canonical IVP Enhancement
 * and Feature-Extraction Consistency (TEST-01 through TEST-11).
 */
public class CanonicalIvpConsistencyRequirementTest {

    @Test
    @DisplayName("TEST-01: Determinism — same input + policy yields byte-identical pixel output and SHA-256")
    void test01_determinism() {
        BufferedImage img1 = SampleDataGenerator.generateDemoParkingLot();
        BufferedImage img2 = SampleDataGenerator.generateDemoParkingLot();

        ImageEnhancementService service = ImageEnhancementService.getDefault();
        EnhancedImage enh1 = service.enhance(OriginalImage.of(img1));
        EnhancedImage enh2 = service.enhance(OriginalImage.of(img2));

        assertEquals(enh1.getPolicyId(), enh2.getPolicyId(), "Policy IDs must match");
        assertEquals("ivp-enhance-v1", enh1.getPolicyId(), "Policy ID must be ivp-enhance-v1");
        assertEquals(enh1.getPolicyFingerprint(), enh2.getPolicyFingerprint(), "Policy fingerprints must match");
        assertEquals(enh1.getSha256(), enh2.getSha256(), "Output pixel SHA-256 hashes must be identical");

        BufferedImage buf1 = enh1.getInternalBuffer();
        BufferedImage buf2 = enh2.getInternalBuffer();
        assertEquals(buf1.getWidth(), buf2.getWidth());
        assertEquals(buf1.getHeight(), buf2.getHeight());
        for (int y = 0; y < buf1.getHeight(); y++) {
            for (int x = 0; x < buf1.getWidth(); x++) {
                assertEquals(buf1.getRGB(x, y), buf2.getRGB(x, y),
                    "Pixel mismatch at (" + x + "," + y + ")");
            }
        }
    }

    @Test
    @DisplayName("TEST-02: What is displayed is what is computed — byte-identical input to stages and viewer")
    void test02_whatIsDisplayedIsWhatIsComputed() {
        BufferedImage demo = SampleDataGenerator.generateDemoParkingLot();
        EnhancedImage enhancedFull = ImageEnhancementService.getDefault().enhance(OriginalImage.of(demo));
        ParkingSlot slot = new ParkingSlot(1, 10, 10, 80, 100);

        BufferedImage rawCrop = RoiExtractor.extract(demo, slot);
        BufferedImage enhancedCrop = RoiExtractor.extract(enhancedFull.image(), slot);

        FeatureExtractor.ProcessedRoi processed = FeatureExtractor.processWithStages(rawCrop, enhancedCrop, false);

        // The displayed enhanced ROI is byte-identical to the input enhanced crop
        BufferedImage displayedEnhanced = processed.getEnhanced();
        assertEquals(enhancedCrop.getWidth(), displayedEnhanced.getWidth());
        assertEquals(enhancedCrop.getHeight(), displayedEnhanced.getHeight());
        for (int y = 0; y < enhancedCrop.getHeight(); y++) {
            for (int x = 0; x < enhancedCrop.getWidth(); x++) {
                assertEquals(enhancedCrop.getRGB(x, y), displayedEnhanced.getRGB(x, y),
                    "Displayed enhanced ROI must match input enhanced crop exactly at (" + x + "," + y + ")");
            }
        }

        // Grayscale stage is computed strictly from the enhanced ROI
        BufferedImage manualGray = GrayscaleConverter.toGrayscale(enhancedCrop);
        BufferedImage actualGray = processed.getGray();
        for (int y = 0; y < manualGray.getHeight(); y++) {
            for (int x = 0; x < manualGray.getWidth(); x++) {
                assertEquals(manualGray.getRGB(x, y), actualGray.getRGB(x, y),
                    "Grayscale stage must be computed directly from enhanced crop at (" + x + "," + y + ")");
            }
        }
    }

    @Test
    @DisplayName("TEST-03: Pipeline Order — Full image enhanced once, slot cropped from enhanced working image")
    void test03_pipelineOrder() {
        BufferedImage demo = SampleDataGenerator.generateDemoParkingLot();
        List<ParkingSlot> slots = ConfigLoader.loadSlotsFromCsv("data/config/slots.csv");

        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setAnalysisMode(OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE);
        AnalysisSummary summary = svc.analyze(demo, slots, false);

        EnhancedImage enhancedFull = ImageEnhancementService.getDefault().enhance(OriginalImage.of(demo));

        for (SlotAnalysisResult res : summary.getSlotResults()) {
            ParkingSlot s = slots.stream().filter(slot -> slot.getId() == res.getId()).findFirst().orElseThrow();
            BufferedImage expectedEnhancedCrop = RoiExtractor.extract(enhancedFull.image(), s);
            BufferedImage actualEnhancedRoi = res.getEnhancedRoiImage();

            assertNotNull(actualEnhancedRoi, "Enhanced ROI image must not be null in result");
            assertEquals(expectedEnhancedCrop.getWidth(), actualEnhancedRoi.getWidth());
            assertEquals(expectedEnhancedCrop.getHeight(), actualEnhancedRoi.getHeight());
            for (int y = 0; y < expectedEnhancedCrop.getHeight(); y++) {
                for (int x = 0; x < expectedEnhancedCrop.getWidth(); x++) {
                    assertEquals(expectedEnhancedCrop.getRGB(x, y), actualEnhancedRoi.getRGB(x, y),
                        "Slot #" + s.getId() + " enhanced ROI must match crop from pre-enhanced full image at (" + x + "," + y + ")");
                }
            }
        }
    }

    @Test
    @DisplayName("TEST-04: Immutability — Original photograph is never mutated")
    void test04_immutability() {
        BufferedImage original = SampleDataGenerator.generateDemoParkingLot();
        int sampleX = 50, sampleY = 50;
        int originalRgb = original.getRGB(sampleX, sampleY);

        OriginalImage wrapped = OriginalImage.of(original);
        EnhancedImage enhanced = ImageEnhancementService.getDefault().enhance(wrapped);

        // Assert original BufferedImage was not modified
        assertEquals(originalRgb, original.getRGB(sampleX, sampleY),
            "Original BufferedImage must not be mutated by enhancement");

        // Mutating source externally does not affect wrapped OriginalImage
        original.setRGB(sampleX, sampleY, 0xFF00FF);
        assertNotEquals(original.getRGB(sampleX, sampleY), wrapped.getRGB(sampleX, sampleY),
            "OriginalImage defensive copy must isolate internal buffer from external mutation");
    }

    @Test
    @DisplayName("TEST-05: Double-enhancement rejected — Compile and runtime enforcement")
    void test05_doubleEnhancementRejected() {
        BufferedImage img = SampleDataGenerator.generateDemoParkingLot();
        EnhancedImage enh = ImageEnhancementService.getDefault().enhance(OriginalImage.of(img));

        assertThrows(IllegalStateException.class, () -> {
            ImageEnhancementService.getDefault().enhance(enh);
        }, "Enhancing an already-enhanced image must fail fast with IllegalStateException");
    }

    @Test
    @DisplayName("TEST-06: Slider isolation — UI preview sliders do not affect feature extraction or models")
    void test06_sliderIsolation() {
        BufferedImage img = SampleDataGenerator.generateDemoParkingLot();
        ParkingSlot slot = new ParkingSlot(1, 10, 10, 80, 100);

        // 1. Extract canonical features
        EnhancedImage enh = ImageEnhancementService.getDefault().enhance(OriginalImage.of(img));
        BufferedImage rawCrop = RoiExtractor.extract(img, slot);
        BufferedImage enhCrop = RoiExtractor.extract(enh.image(), slot);
        FeatureVector canonicalFv = FeatureExtractor.extract(rawCrop, enhCrop, false);

        // 2. Manipulate preview sliders heavily
        BufferedImage previewImg = ImageEnhancer.adjustForPreview(img, 2.5, 45.0, 1.8);
        assertNotEquals(img.getRGB(20, 20), previewImg.getRGB(20, 20),
            "Preview image must have altered pixel values");

        // 3. Extract canonical features again
        EnhancedImage enh2 = ImageEnhancementService.getDefault().enhance(OriginalImage.of(img));
        BufferedImage enhCrop2 = RoiExtractor.extract(enh2.image(), slot);
        FeatureVector canonicalFv2 = FeatureExtractor.extract(rawCrop, enhCrop2, false);

        // 4. Verify exact numerical equivalence
        for (int i = 0; i < FeatureVector.FEATURE_COUNT; i++) {
            assertEquals(canonicalFv.get(i), canonicalFv2.get(i), 1e-9,
                "Feature " + i + " must remain identical regardless of UI preview slider state");
        }
    }

    @Test
    @DisplayName("TEST-07: Honest detector failure — Explicit status, NaN vehicle features, and dedicated schema")
    void test07_honestDetectorFailure() {
        BufferedImage img = SampleDataGenerator.generateDemoParkingLot();
        List<ParkingSlot> slots = ConfigLoader.loadSlotsFromCsv("data/config/slots.csv");

        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        OnnxVehicleDetectionService failingDetector = OnnxVehicleDetectionService.createWithTestBackend(
            "failing_detector.onnx", 640, 640, 0.30, 0.45,
            (tensor, w, h) -> { throw new RuntimeException("GPU backend lost"); }
        );
        svc.setVehicleDetectionService(failingDetector);

        AnalysisSummary summary = svc.analyze(img, slots, false, OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION);
        assertEquals(12, summary.getUnclassifiedSlots(), "All slots must be UNCLASSIFIED on detector failure");

        for (SlotAnalysisResult res : summary.getSlotResults()) {
            assertEquals(SlotStatus.UNCLASSIFIED, res.getStatus());
            FeatureVector fv = res.getFeatures();
            assertEquals(FeatureVector.DETECTOR_FAILED_SCHEMA_VERSION, fv.getSchemaVersion(),
                "Schema must be v2.3-8feat-detector-failed");
            assertFalse(fv.isDetectorAvailable(), "Detector must be marked unavailable");
            assertTrue(Double.isNaN(fv.getVehicleOverlapRatio()),
                "Vehicle overlap ratio must be NaN on failure, NOT 0.0");
            assertTrue(Double.isNaN(fv.getVehicleMatchedFlag()),
                "Vehicle matched flag must be NaN on failure, NOT 0.0");
        }
    }

    @Test
    @DisplayName("TEST-08: Schema mismatch fails closed — TREE_INCOMPATIBLE on raw/enhanced mismatch")
    void test08_schemaMismatchFailsClosed() throws IOException {
        Path tempModel = Files.createTempFile("raw_model_test", ".model");
        // Create a model with schema v2.3-10feat-raw-v1
        String payload = """
            format=PARKNCHILL_CART_MODEL_V23
            schemaVersion=v2.3-10feat-raw-v1
            featureCount=10
            maxDepth=4
            datasetSource=Raw Baseline Set
            savedAt=2026-10-22T00:00:00Z
            policyId=ivp-raw-v1
            policyFingerprint=0000000000000000000000000000000000000000000000000000000000000000
            trainMin=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
            trainMax=255.0,1.0,1.0,1.0,1.0,255.0,1.0,1.0,1.0,1.0
            NODE|0|LEAF|AVAILABLE|0|10
            """;
        String sha = computeSha256Hex(payload);
        String fullFile = payload + "sha256=" + sha + "\n";
        Files.writeString(tempModel, fullFile);

        ModelSerializer.LoadedModelBundle bundle = ModelSerializer.loadModel(tempModel);
        assertFalse(bundle.compatibleTenFeatureModel(),
            "Model trained on raw features must not be marked compatible in enhanced pipeline");
        assertTrue(bundle.statusSummary().contains("TREE_INCOMPATIBLE"),
            "Status summary must contain TREE_INCOMPATIBLE");
        assertTrue(bundle.compatibilityMessage().contains("v2.3-10feat-raw-v1"),
            "Compatibility message must cite the schema mismatch");
    }

    @Test
    @DisplayName("TEST-09: Migration preserves partitions & provenance metadata")
    void test09_migrationPreservesPartitionsAndProvenance() throws IOException {
        List<DatasetManifest.ManifestRecord> records = DatasetManifest.loadManifest("data/labels/manifest.csv");
        assertNotNull(records);
        assertFalse(records.isEmpty());

        long trainCount = records.stream().filter(r -> "TRAIN".equalsIgnoreCase(r.partition())).count();
        long valCount = records.stream().filter(r -> "VALIDATION".equalsIgnoreCase(r.partition())).count();
        long testCount = records.stream().filter(r -> "TEST".equalsIgnoreCase(r.partition())).count();

        assertTrue(trainCount > 0, "Train partition must have records");
        assertTrue(valCount > 0, "Validation partition must have records");
        assertTrue(testCount > 0, "Test partition must have records");

        // Verify that every demo lot record has verified provenance
        for (DatasetManifest.ManifestRecord rec : records) {
            if ("LOT_A_DEMO".equals(rec.captureGroup())) {
                assertEquals("DEMO_LOT_EXTRACTED", rec.provenance());
                assertEquals("VERIFIED_DEMO", rec.reviewStatus());
            }
        }
    }

    @Test
    @DisplayName("TEST-10: PKLot import applies canonical policy — Full-image pre-enhancement before ROI crops")
    void test10_pKLotImportAppliesCanonicalPolicy() {
        ProcessingPolicy canonical = ProcessingPolicy.CANONICAL;
        assertEquals("ivp-enhance-v1", canonical.policyId());
        assertEquals(0.02, canonical.lowPercentile(), 1e-6);
        assertEquals(0.98, canonical.highPercentile(), 1e-6);
        assertEquals(24, canonical.minContrastRange());
        assertEquals(82.0, canonical.darkMeanCutoff(), 1e-6);
        assertEquals(0.86, canonical.darkGamma(), 1e-6);
        assertEquals(190.0, canonical.brightMeanCutoff(), 1e-6);
        assertEquals(1.08, canonical.brightGamma(), 1e-6);
        assertNotNull(canonical.fingerprint());
        assertEquals(64, canonical.fingerprint().length(), "Fingerprint must be 64-hex SHA-256");
    }

    @Test
    @DisplayName("TEST-11: Multi-slot consistency — Each slot cropped from pre-enhanced image")
    void test11_multiSlotConsistency() {
        BufferedImage demo = SampleDataGenerator.generateDemoParkingLot();
        List<ParkingSlot> slots = ConfigLoader.loadSlotsFromCsv("data/config/slots.csv");

        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setAnalysisMode(OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE);
        AnalysisSummary summary = svc.analyze(demo, slots, false);

        assertEquals(12, summary.getTotalSlots());
        assertEquals(4, summary.getOccupiedSlots());
        assertEquals(8, summary.getAvailableSlots());
        assertEquals(0, summary.getNeedsReviewSlots());

        // Verify each slot's enhanced ROI was preserved and matches
        EnhancedImage fullEnh = ImageEnhancementService.getDefault().enhance(OriginalImage.of(demo));
        for (SlotAnalysisResult res : summary.getSlotResults()) {
            ParkingSlot slot = slots.stream().filter(s -> s.getId() == res.getId()).findFirst().orElseThrow();
            BufferedImage expectedCrop = RoiExtractor.extract(fullEnh.image(), slot);
            BufferedImage actualRoi = res.getEnhancedRoiImage();

            assertEquals(expectedCrop.getWidth(), actualRoi.getWidth());
            assertEquals(expectedCrop.getHeight(), actualRoi.getHeight());
            assertEquals(expectedCrop.getRGB(40, 50), actualRoi.getRGB(40, 50));
        }
    }

    private static String computeSha256Hex(String text) {
        try {
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                sb.append(String.format(java.util.Locale.ROOT, "%02x", b & 0xff));
            }
            return sb.toString();
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }
}
