package com.parknchillai.ai;

import com.parknchillai.app.ParkingAnalysisService;
import com.parknchillai.detection.OnnxVehicleDetectionService;
import com.parknchillai.detection.SlotOverlapAnalyzer;
import com.parknchillai.detection.VehicleDetectionService;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Genuine ONNX &amp; External {@code PKLOT_ROOT} Full-Archive Integration Test Suite
 * (ParkNChill v2.3 Requirement 7).
 *
 * <p>Kept strictly separate from fake-detector unit tests:
 * <ul>
 *   <li>{@link #realImageIntegrationWithIndependentXmlAnnotationsWhenCompatibleModelAvailable()}
 *       executes live {@link OnnxVehicleDetectionService} inference against the checked-in
 *       independent XML/image fixtures ({@code data/fixtures/pklot}) when a compatible ONNX model
 *       is present at {@code models/vehicle_detector.onnx}, and explicitly reports {@code SKIPPED}
 *       when the model is absent.</li>
 *   <li>{@link #fullPkLotArchiveAndRealOnnxDetectorEndToEndImportAndPartitionTraining()}
 *       requires both an external {@code PKLOT_ROOT} archive (via {@code -DPKLOT_ROOT},
 *       {@code PKLOT_ROOT} env var, or {@code ~/Desktop/PKLot}) and a live {@code READY} ONNX model,
 *       explicitly reporting {@code SKIPPED} when either is absent.</li>
 * </ul>
 */
class RealPkLotArchiveIntegrationTest {

    @Test
    void realImageIntegrationWithIndependentXmlAnnotationsWhenCompatibleModelAvailable() throws Exception {
        Path fixtureRoot = Paths.get("data/fixtures/pklot");
        OnnxVehicleDetectionService realDetector = new OnnxVehicleDetectionService();

        boolean fixturesPresent = Files.isDirectory(fixtureRoot);
        boolean onnxReady = realDetector.getStatus() == VehicleDetectionService.DetectorStatus.READY;

        Assumptions.assumeTrue(
            fixturesPresent && onnxReady,
            "SKIPPED: Real-image ONNX integration test requires a compatible pretrained ONNX model at "
            + OnnxVehicleDetectionService.DEFAULT_MODEL_PATH + " (current detectorStatus=" + realDetector.getStatus()
            + ", fixturesPresent=" + fixturesPresent + "). Run scripts/setup_onnx_model.ps1 or .sh first."
        );

        PkLotDatasetImporter.PkLotImportResult result = PkLotDatasetImporter.importDirectory(
            fixtureRoot,
            realDetector,
            SlotOverlapAnalyzer.OverlapPolicy.defaultValidationTunedPolicy(),
            false,
            p -> {},
            () -> false
        );

        assertEquals(3, result.validImageCount(), "All 3 checked-in fixture images must be processed with live ONNX");
        assertEquals(12, result.validSlotCount(), "The three fixture XML files contain four slot annotations each; all 12 must yield 10-feature records");
        assertEquals(0, result.detectorFailureCount(), "Live ONNX detector must have 0 inference failures");
    }

    @Test
    void fullPkLotArchiveAndRealOnnxDetectorEndToEndImportAndPartitionTraining() throws Exception {
        Path pkLotRoot = resolveExternalPkLotRootOnly();
        OnnxVehicleDetectionService realDetector = new OnnxVehicleDetectionService();

        boolean pkLotPresent = pkLotRoot != null && Files.isDirectory(pkLotRoot)
            && !PkLotDatasetImporter.inspectSampleAnnotations(pkLotRoot, 1).sampleEntries().isEmpty();
        boolean onnxReady = realDetector.getStatus() == VehicleDetectionService.DetectorStatus.READY;

        Assumptions.assumeTrue(
            pkLotPresent && onnxReady,
            "SKIPPED: Full-archive integration test requires both an external PKLOT_ROOT archive "
            + "(e.g. ~/Desktop/PKLot or -DPKLOT_ROOT=<path>) and a live READY ONNX model at "
            + "models/vehicle_detector.onnx (current externalPkLotPresent=" + pkLotPresent
            + ", detectorStatus=" + realDetector.getStatus() + ")."
        );

        PkLotDatasetImporter.PkLotImportResult result = PkLotDatasetImporter.importDirectory(
            pkLotRoot,
            realDetector,
            SlotOverlapAnalyzer.OverlapPolicy.defaultValidationTunedPolicy(),
            false,
            p -> {},
            () -> false
        );

        assertTrue(result.validImageCount() > 0, "Real PKLot import must process at least 1 valid image");
        assertTrue(result.validSlotCount() > 0, "Real PKLot import must produce valid 10-feature slot records");
        assertEquals(0, result.detectorFailureCount(), "Real ONNX detector must succeed without inference failures");

        ParkingAnalysisService service = ParkingAnalysisService.createWithDefaultModel();
        service.setVehicleDetectionService(realDetector);
        ParkingAnalysisService.PkLotTrainingSummary summary = service.trainFromPartitionedPkLotRecords(
            result.eligibleRecords(), 4, 2, pkLotRoot.resolve("pklot_frozen_test.meta")
        );
        assertNotNull(summary);
        assertTrue(summary.heldOutTestReport().accuracy >= 0.70,
            "Expected held-out test accuracy >= 70% on real PKLot archive; got " + summary.heldOutTestReport().accuracy);
    }

    private static Path resolveExternalPkLotRootOnly() {
        String sysProp = System.getProperty("PKLOT_ROOT");
        if (sysProp != null && !sysProp.isBlank()) {
            Path p = Paths.get(sysProp.trim());
            if (Files.isDirectory(p)) return p.toAbsolutePath().normalize();
        }
        String env = System.getenv("PKLOT_ROOT");
        if (env != null && !env.isBlank()) {
            Path p = Paths.get(env.trim());
            if (Files.isDirectory(p)) return p.toAbsolutePath().normalize();
        }
        String userHome = System.getProperty("user.home", "");
        List<Path> desktopCandidates = List.of(
            Paths.get(userHome, "Desktop", "PKLot"),
            Paths.get(userHome, "Desktop", "PKLot_dataset"),
            Paths.get("data", "pklot")
        );
        for (Path candidate : desktopCandidates) {
            if (Files.isDirectory(candidate)) {
                return candidate.toAbsolutePath().normalize();
            }
        }
        return null;
    }
}
