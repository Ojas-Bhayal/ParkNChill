package com.parknchillai.ui;

import com.parknchillai.ai.DecisionTreeModel;
import com.parknchillai.ai.DecisionTreeTrainer;
import com.parknchillai.ai.ModelEvaluator;
import com.parknchillai.ai.ModelSerializer;
import com.parknchillai.ai.PkLotDatasetImporter;
import com.parknchillai.ai.SlotTrainingRecord;
import com.parknchillai.app.AppState;
import com.parknchillai.app.ParkingAnalysisService;
import com.parknchillai.detection.OnnxVehicleDetectionService;
import com.parknchillai.detection.VehicleCategory;
import com.parknchillai.detection.VehicleDetection;
import com.parknchillai.detection.VehicleDetectionService;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import com.parknchillai.utils.SampleDataGenerator;
import javafx.scene.Scene;
import javafx.scene.control.TableColumn;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive UI Workflow, Layout Simplification, PKLot Integration, and Model Persistence
 * Acceptance Test Suite (ParkNChill v2.3 Requirements 1–7).
 */
class UiWorkflowAndPkLotV23Test {
    static {
        initJavaFxToolkit();
    }

    private static synchronized void initJavaFxToolkit() {
        try {
            System.setProperty("prism.order", "sw");
            System.setProperty("prism.text", "t2k");
            javafx.application.Platform.startup(() -> {});
            javafx.application.Platform.setImplicitExit(false);
        } catch (Throwable ignored) {
            // Platform already initialized or running under headless stub
        }
    }



    @TempDir
    Path tempDir;

    private static VehicleDetectionService createReadyDemoDetector() {
        return new VehicleDetectionService() {
            @Override
            public DetectorStatus getStatus() {
                return DetectorStatus.READY;
            }

            @Override
            public String describeConfiguration() {
                return "verified-onnx-detector-v23";
            }

            @Override
            public DetectionBatchResult detectVehicles(BufferedImage image) {
                List<VehicleDetection> dets = List.of(
                    new VehicleDetection("V1", VehicleCategory.CAR, 15, 22, 70, 88, 0.94, "onnx-v23", 640, 640, "onnx"),
                    new VehicleDetection("V2", VehicleCategory.CAR, 105, 132, 70, 88, 0.91, "onnx-v23", 640, 640, "onnx"),
                    new VehicleDetection("V3", VehicleCategory.TRUCK, 195, 132, 70, 88, 0.89, "onnx-v23", 640, 640, "onnx"),
                    new VehicleDetection("V4", VehicleCategory.CAR, 285, 242, 70, 88, 0.92, "onnx-v23", 640, 640, "onnx")
                );
                return new DetectionBatchResult(DetectorStatus.READY, "verified-onnx-detector-v23", "Apache-2.0", "OK", dets, dets, 4L);
            }
        };
    }

    @Test
    void scenario1To4StartupLoadDetectReviewAndGeometryChangeConfirmationWorkflow() throws Exception {
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(createReadyDemoDetector());

        // 1. Initial startup with no image
        MainWindow window = new MainWindow(svc, false);
        Scene scene1366 = new Scene(window, 1366, 768);
        assertEquals(1366.0, scene1366.getWidth(), 1e-6);
        assertEquals(768.0, scene1366.getHeight(), 1e-6);

        assertEquals("ParkNChill", window.getLblAppTitle().getText());
        assertEquals("Smart Parking Occupancy Detection", window.getLblAppSubtitle().getText());

        AppState state = window.getAppState();
        assertEquals(AppState.WorkflowStage.LOAD_IMAGE, state.getCurrentStage());
        assertNull(state.getCurrentImage());
        assertEquals(AppState.StageVisualState.ACTIVE, state.getStageVisualState(AppState.WorkflowStage.LOAD_IMAGE));
        assertEquals(AppState.StageVisualState.BLOCKED, state.getStageVisualState(AppState.WorkflowStage.DETECT_VEHICLES_AND_SLOTS));
        assertEquals(AppState.StageVisualState.BLOCKED, state.getStageVisualState(AppState.WorkflowStage.REVIEW_OCCUPANCY));

        // Summary must not show fake 0 Occupied / 0 Available confirmed outcomes before analysis
        assertEquals("—", window.getSummaryView().getOccupiedText());
        assertEquals("—", window.getSummaryView().getAvailableText());

        // Advanced options must be collapsed by default
        assertFalse(window.getParkingView().isAdvancedOptionsExpanded(),
            "Advanced Image Options must be collapsed by default");
        assertFalse(window.getSlotTableView().getAdvancedSlotDetailsPane().isExpanded(),
            "Advanced Slot Details must be collapsed by default");
        assertFalse(window.getModelDetailsView().getAdvModelSettingsPane().isExpanded(),
            "Advanced Model Settings must be collapsed by default");
        assertFalse(window.getModelDetailsView().getAdvEvaluationDiagnosticsPane().isExpanded(),
            "Advanced Evaluation Diagnostics must be collapsed by default");

        // 2. Loading an image
        File demoFile = SampleDataGenerator.ensureSampleImageExists(SampleDataGenerator.DEFAULT_SAMPLE_IMAGE_PATH);
        window.loadImageFromFile(demoFile);
        assertNotNull(state.getCurrentImage());
        assertEquals(AppState.WorkflowStage.DETECT_VEHICLES_AND_SLOTS, state.getCurrentStage());
        assertEquals(AppState.StageVisualState.COMPLETED, state.getStageVisualState(AppState.WorkflowStage.LOAD_IMAGE));
        assertEquals(AppState.StageVisualState.ACTIVE, state.getStageVisualState(AppState.WorkflowStage.DETECT_VEHICLES_AND_SLOTS));
        assertEquals(12, state.getCurrentSlots().size());

        // 3. Running detection
        window.runDetectionAndAnalysis();
        assertNotNull(state.getCurrentSummary());
        assertEquals("12", window.getSummaryView().getTotalText());
        assertEquals("4", window.getSummaryView().getOccupiedText());
        assertEquals("8", window.getSummaryView().getAvailableText());
        assertTrue(window.getSummaryView().getSummaryNoteText().contains("Provisional"),
            "Summary must distinguish provisional detector suggestions from confirmed labels");

        // Navigate to Stage 3: Review Occupancy
        window.navigateToStage(AppState.WorkflowStage.REVIEW_OCCUPANCY);
        assertEquals(AppState.WorkflowStage.REVIEW_OCCUPANCY, state.getCurrentStage());
        assertTrue(window.getTxtReviewerId().isVisible(), "Reviewer ID input must be visible during slot review");
        window.getTxtReviewerId().setText("QA_REVIEWER_1");

        // Verify the 5 essential columns in SlotTableView and desktop width fit (1366x768 and 1600x900)
        List<String> colHeaders = window.getSlotTableView().getTable().getColumns().stream()
            .map(TableColumn::getText)
            .collect(Collectors.toList());
        assertEquals(List.of("Slot ID", "Current Status", "Verified Label", "Review Status", "Detector Suggestion"),
            colHeaders, "Normal review table must show only the 5 essential columns");
        double totalMinColWidth = window.getSlotTableView().getTotalMinimumColumnWidth();
        assertTrue(totalMinColWidth <= 620.0,
            "5-column review table must fit inside 1366x768 and 1600x900 split pane without horizontal scrolling; got " + totalMinColWidth);

        // When no slot is selected, show unobtrusive 1-line prompt and hide large details box
        assertTrue(window.getSlotTableView().isUnselectedPromptVisible(),
            "Unobtrusive 1-line prompt must be visible when no slot row is selected");
        assertFalse(window.getSlotTableView().isSelectedSlotActionBoxVisible(),
            "Selected-slot review card must remain hidden when no slot row is selected");

        // 4. Reviewing and correcting a parking slot + geometry-change re-confirmation gate
        ParkingSlot slot1 = state.getCurrentSlots().get(0);
        window.getSlotTableView().selectSlot(slot1);
        assertFalse(window.getSlotTableView().isUnselectedPromptVisible());
        assertTrue(window.getSlotTableView().isSelectedSlotActionBoxVisible());
        assertEquals(slot1.getId(), window.getParkingView().getSelectedSlot().getId(),
            "Selecting a slot in the table must highlight its boundary in ParkingView");

        // Mark Slot #1 as OCCUPIED and Slots #2, #3, #6 as AVAILABLE/OCCUPIED
        window.getSlotTableView().getBtnMarkOccupied().fire();
        assertEquals(SlotStatus.OCCUPIED, state.getVerifiedLabelForSlot(slot1.getId()));
        assertTrue(state.getReviewStatusForSlot(slot1.getId()).contains("Confirmed"));

        state.reviewSlot(2, SlotStatus.AVAILABLE);
        state.reviewSlot(3, SlotStatus.AVAILABLE);
        state.reviewSlot(6, SlotStatus.OCCUPIED);
        assertEquals(4, state.getEligibleReviewedRecords().size());
        assertTrue(state.canTrainDecisionTree());

        // Now modify Slot #1 geometry -> must invalidate training eligibility until re-confirmed!
        state.markSlotGeometryModified(slot1.getId());
        assertTrue(state.isSlotGeometryUnconfirmed(slot1.getId()));
        assertFalse(state.canTrainDecisionTree(),
            "Modified slot geometry must block training until label is explicitly re-confirmed");
        assertTrue(state.getTrainingDisabledReason().contains("unconfirmed boundary geometry"),
            "Disabled training reason must explain the unconfirmed geometry change");

        // Re-confirm Slot #1 label after geometry change -> training becomes eligible again
        state.reviewSlot(slot1.getId(), SlotStatus.OCCUPIED);
        assertFalse(state.isSlotGeometryUnconfirmed(slot1.getId()));
        assertTrue(state.canTrainDecisionTree());

        // Revisit Step 1 and Step 2 -> verify image, slots, and reviewed labels are preserved!
        window.navigateToStage(AppState.WorkflowStage.LOAD_IMAGE);
        window.navigateToStage(AppState.WorkflowStage.REVIEW_OCCUPANCY);
        assertNotNull(state.getCurrentImage());
        assertEquals(4, state.getEligibleReviewedRecords().size());
    }

    @Test
    void scenario5To9PkLotImportInspectTrainEvaluateAndSaveLoadModelWorkflow() throws Exception {
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(createReadyDemoDetector());
        MainWindow window = new MainWindow(svc, false);
        AppState state = window.getAppState();

        // Create a realistic multi-camera/date PKLot directory structure with real PNG images and XML annotations
        Path pkLotRoot = tempDir.resolve("PKLot_Sample");
        Path cam1Date1 = pkLotRoot.resolve("PUCPR/Sunny/2012-10-12");
        Path cam2Date2 = pkLotRoot.resolve("UFPR04/Cloudy/2012-12-07");
        Path cam3Date3 = pkLotRoot.resolve("UFPR05/Sunny/2012-12-15");
        Files.createDirectories(cam1Date1);
        Files.createDirectories(cam2Date2);
        Files.createDirectories(cam3Date3);

        File demoFile = SampleDataGenerator.ensureSampleImageExists(SampleDataGenerator.DEFAULT_SAMPLE_IMAGE_PATH);
        BufferedImage baseImg = ImageIO.read(demoFile);

        writePkLotImageAndXml(cam1Date1, "2012-10-12_08_00_00", baseImg, true);
        writePkLotImageAndXml(cam2Date2, "2012-12-07_10_15_00", baseImg, false);
        writePkLotImageAndXml(cam3Date3, "2012-12-15_14_30_00", baseImg, false);

        // 6. Inspecting PKLot annotations (Sample Annotations... must be read-only and never fabricate features!)
        int poolBeforeInspect = state.getEligibleReviewedRecords().size();
        PkLotDatasetImporter.AnnotationInspectionReport inspectRep =
            window.getModelDetailsView().inspectPkLotAnnotations(pkLotRoot);
        assertNotNull(inspectRep);
        assertEquals(3, inspectRep.annotationFilesFound());
        assertEquals(3, inspectRep.pairedImagesFound());
        assertTrue(inspectRep.occupiedAnnotations() >= 6);
        assertTrue(inspectRep.availableAnnotations() >= 6);
        assertEquals(1, inspectRep.invalidAnnotations(), "Malformed space entry must be counted as invalid");
        assertEquals(poolBeforeInspect, state.getEligibleReviewedRecords().size(),
            "Sample Annotations... inspection must NEVER fabricate training records");

        // 5. Importing PKLot asynchronously with genuine image features, ONNX detections, and camera/date partitions
        PkLotDatasetImporter.PkLotImportResult importRes =
            window.getModelDetailsView().importPkLotDirectoryAsync(pkLotRoot).get(5, TimeUnit.SECONDS);
        assertNotNull(importRes);
        assertEquals(3, importRes.validImageCount());
        assertEquals(12, importRes.validSlotCount());
        assertEquals(6, importRes.occupiedCount());
        assertEquals(6, importRes.availableCount());
        assertEquals(1, importRes.invalidRecordCount());
        assertEquals(0, importRes.detectorFailureCount());
        assertEquals(3, importRes.groupPartitionMap().size(), "Must preserve 3 distinct camera/date capture groups");

        // Verify all imported records have 10-feature schema v2.3-10feat and IMPORTED_GROUND_TRUTH provenance
        for (var rec : state.getEligibleReviewedRecords()) {
            assertEquals(FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION, rec.featureSchemaVersion());
            assertEquals(SlotTrainingRecord.PROVENANCE_IMPORTED_GROUND_TRUTH, rec.provenance());
            assertEquals("", rec.reviewerId(), "Imported records must not have a fabricated reviewerId");
            assertFalse(rec.isHumanVerified());
            assertTrue(rec.isImportedGroundTruth());
            assertFalse(rec.provisional());
        }

        // 7. Training the 10-feature Decision Tree using saved PKLot partitions
        window.navigateToStage(AppState.WorkflowStage.TRAIN_DECISION_TREE);
        assertTrue(state.canTrainDecisionTree());
        assertFalse(window.getModelDetailsView().getBtnTrainDecisionTree().isDisabled());

        window.getModelDetailsView().handleTrainDecisionTree();
        assertTrue(state.isCompatibleTenFeatureModelActive(),
            "Training on 10-feature PKLot records must activate a compatible 10-feature Decision Tree");
        assertEquals(10, svc.getPredictor().getModel().getFeatureCount());
        assertEquals("VEHICLE_DETECTION", svc.getAnalysisMode().name(),
            "Training a tree must not silently replace the default detector result mode");
        assertFalse(window.getModelDetailsView().getLblLegacyModelWarning().isVisible(),
            "Legacy 8-feature warning must hide once a compatible 10-feature model is active");

        // 8. Evaluating the trained model (Stage 5)
        window.navigateToStage(AppState.WorkflowStage.EVALUATE_MODEL);
        ModelEvaluator.EvaluationReport evalRep = state.getLastEvaluationReport();
        assertNotNull(evalRep);
        assertTrue(window.getModelDetailsView().getLblEvalPartitionBadge().getText().contains("Test set"),
            "Evaluation screen must clearly identify the Held-Out Test Partition");
        assertEquals(ModelDetailsView.FALSE_AVAILABLE_EXPLANATION,
            window.getModelDetailsView().getLblFalseAvailablePlainEnglish().getText(),
            "Evaluation screen must explain False-Available error in plain English");

        // 9. Saving and loading the model (Stage 6)
        window.navigateToStage(AppState.WorkflowStage.SAVE_AND_USE_MODEL);
        Path savedModelPath = tempDir.resolve("trained_10feat.pncmodel");
        ModelSerializer.LoadedModelBundle savedBundle =
            window.getModelDetailsView().saveActiveModelToFile(savedModelPath);
        assertTrue(Files.exists(savedModelPath));
        assertTrue(savedBundle.compatibleTenFeatureModel());
        assertEquals(10, savedBundle.featureCount());

        var summaryBeforeLoad = state.getCurrentSummary();
        var modeBeforeLoad = svc.getAnalysisMode();
        ModelSerializer.LoadedModelBundle loadedBundle =
            window.getModelDetailsView().loadModelFromFile(savedModelPath);
        assertTrue(loadedBundle.compatibleTenFeatureModel());
        assertEquals(FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION, loadedBundle.schemaVersion());
        assertEquals(savedBundle.model().toTreeString(), loadedBundle.model().toTreeString());
        assertEquals(modeBeforeLoad, svc.getAnalysisMode(),
            "Loading a saved tree must not silently switch the primary occupancy mode");
        assertSame(summaryBeforeLoad, state.getCurrentSummary(),
            "Loading a saved tree must not rerun analysis or replace the current detection summary");
        assertEquals("VEHICLE_DETECTION", window.getParkingView().getSelectedAnalysisMode().name(),
            "Loading a saved tree must not silently switch the displayed primary result mode");
        assertEquals("Run Tree Separately", window.getParkingView().getBtnRunLoadedTree().getText(),
            "A distinct action must apply tree predictions without overwriting detector results");
    }

    @Test
    void loadingSavedTreeDoesNotReplaceDetectorResultsAndSeparateRunPreservesThem() throws Exception {
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(createReadyDemoDetector());
        MainWindow window = new MainWindow(svc, false);

        // Establish a detector-mode result against a real bundled demo photograph.
        window.handleLoadDemo(true).get(10, TimeUnit.SECONDS);
        var detectorSummaryBefore = window.getAppState().getCurrentSummary();
        assertNotNull(detectorSummaryBefore, "Demo analysis should provide a baseline result");
        assertTrue(detectorSummaryBefore.getTotalSlots() > 0, "Demo analysis should have slots to evaluate");
        var statusesBefore = detectorSummaryBefore.getSlotResults().stream().map(r -> r.getStatus()).toList();
        assertEquals("VEHICLE_DETECTION", svc.getAnalysisMode().name());

        // A compatible saved ten-feature tree.
        double[][] x = {
            {35, .08, .80, .12, .05, 12, .02, .04, .00, 0},
            {70, .42, .25, .38, .35, 115, .55, .31, .85, 1},
            {42, .12, .70, .18, .08, 20, .05, .06, .03, 0},
            {78, .50, .20, .45, .44, 135, .68, .39, .92, 1}
        };
        DecisionTreeModel model = new DecisionTreeTrainer(3, 1).train(x,
            new SlotStatus[]{SlotStatus.AVAILABLE, SlotStatus.OCCUPIED, SlotStatus.AVAILABLE, SlotStatus.OCCUPIED});
        Path savedTree = tempDir.resolve("separate_tree.pncmodel");
        ModelSerializer.saveModel(model, savedTree, "Separate tree behavior test");

        window.getModelDetailsView().loadModelFromFile(savedTree);
        assertSame(detectorSummaryBefore, window.getAppState().getCurrentSummary(),
            "Loading a saved tree must not recompute or replace the detector summary");
        assertEquals(statusesBefore, detectorSummaryBefore.getSlotResults().stream().map(r -> r.getStatus()).toList(),
            "Loading a saved tree must leave the original result statuses unchanged");
        assertEquals("VEHICLE_DETECTION", svc.getAnalysisMode().name(),
            "Loading a saved tree must not silently change the primary occupancy mode");

        window.getParkingView().getBtnRunLoadedTree().fire();
        assertEquals(detectorSummaryBefore.getTotalSlots(), window.getSlotTableView().getSeparatelyAppliedTreePredictionCount(),
            "Separate application must create one tree result per analysed slot");
        assertSame(detectorSummaryBefore, window.getAppState().getCurrentSummary(),
            "Applying the tree separately must not replace the detector analysis summary");
        assertEquals(statusesBefore, detectorSummaryBefore.getSlotResults().stream().map(r -> r.getStatus()).toList(),
            "Applying the tree separately must not change the previous detector statuses");
        assertEquals("VEHICLE_DETECTION", window.getParkingView().getSelectedAnalysisMode().name());
    }

    @Test
    void scenario10HandlesMissingDetectorIncompatibleLegacyModelAndInsufficientTrainingData() throws Exception {
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(new OnnxVehicleDetectionService(tempDir.resolve("missing-detector-scenario10.onnx")));
        MainWindow window = new MainWindow(svc, true);
        AppState state = window.getAppState();

        // 1. Missing detector handling (MODEL_MISSING)
        assertEquals(VehicleDetectionService.DetectorStatus.MODEL_MISSING, state.getDetectorStatus());
        assertTrue(window.getLblDetectorStatusBadge().getText().contains("MODEL_MISSING"));
        assertEquals("Awaiting Review", window.getSummaryView().getOccupiedText());
        assertEquals("Awaiting Review", window.getSummaryView().getAvailableText());
        assertEquals("12", window.getSummaryView().getNeedsReviewText());

        // 2. Incompatible legacy 8-feature model explanation & ONNX first-run setup notice
        window.navigateToStage(AppState.WorkflowStage.TRAIN_DECISION_TREE);
        assertTrue(window.getModelDetailsView().getOnnxSetupBox().isVisible(),
            "ONNX setup box must be prominently visible before PKLot import when model is MODEL_MISSING");
        assertTrue(state.isLegacyEightFeatureModelPresent());
        assertEquals(ModelSerializer.LEGACY_EIGHT_FEATURE_MESSAGE, state.getLegacyModelExplanation());
        assertTrue(window.getModelDetailsView().getLblLegacyModelWarning().isVisible());

        DecisionTreeModel legacy8 = new DecisionTreeTrainer(3, 1).train(
            new double[][]{{60, 0.05, 0.85, 0.15, 0.02, 15, 0.0, 0.02}, {85, 0.45, 0.50, 0.48, 0.42, 120, 0.60, 0.38}},
            new SlotStatus[]{SlotStatus.AVAILABLE, SlotStatus.OCCUPIED}
        );
        Path legacyFile = tempDir.resolve("legacy_8feat.pncmodel");
        ModelSerializer.saveModel(legacy8, legacyFile, "Legacy 8-Feature CSV");
        ModelSerializer.LoadedModelBundle loadedLegacy = window.getModelDetailsView().loadModelFromFile(legacyFile);
        assertFalse(loadedLegacy.compatibleTenFeatureModel());
        assertEquals(8, loadedLegacy.featureCount());
        assertEquals(ModelSerializer.LEGACY_EIGHT_FEATURE_MESSAGE, loadedLegacy.compatibilityMessage());

        // 3. Insufficient training data explanation
        assertFalse(state.canTrainDecisionTree());
        assertTrue(window.getModelDetailsView().getBtnTrainDecisionTree().isDisabled());
        String disabledMsg = window.getModelDetailsView().getLblTrainingDisabledExplanation().getText();
        assertTrue(disabledMsg.contains("Training requires at least 4 confirmed 10-feature records"),
            "Disabled training button must explain why training is disabled; got: " + disabledMsg);

        // 4. Reviewer ID validation when blank
        window.getTxtReviewerId().setText(" ");
        assertFalse(state.isReviewerIdValid());
        assertTrue(window.getLblReviewerValidation().getText().contains("Required"));
    }

    @Test
    void desktopResolutions1366x768And1600x900FitStepperButtonsAndTableWithoutOverflow() throws Exception {
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(createReadyDemoDetector());
        MainWindow window = new MainWindow(svc, true);

        Scene scene1366 = new Scene(window, 1366, 768);
        MainWindow window1600 = new MainWindow(svc, true);
        Scene scene1600 = new Scene(window1600, 1600, 900);
        assertEquals(1366.0, scene1366.getWidth(), 1e-6);
        assertEquals(1600.0, scene1600.getWidth(), 1e-6);

        // Verify simplified user-facing navigation: four plain-language steps, not six internal stages.
        double stepperPrefWidth = window.getTotalStepperBarPreferredWidth();
        double stepperMinWidth = window.getTotalStepperBarMinimumWidth();
        assertTrue(stepperPrefWidth <= 1250.0,
            "Four user-facing steps plus navigation must fit inside 1366x768 (" + stepperPrefWidth + " px)");
        assertTrue(stepperMinWidth <= 1000.0,
            "Minimum stepper width must fit compact desktop windows (" + stepperMinWidth + " px)");
        assertEquals(4, window.getStageButtons().size(), "Only four simple workflow steps should be shown to users");
        assertTrue(window.getStageButtons().containsKey(AppState.WorkflowStage.LOAD_IMAGE));
        assertTrue(window.getStageButtons().containsKey(AppState.WorkflowStage.REVIEW_OCCUPANCY));
        assertTrue(window.getStageButtons().containsKey(AppState.WorkflowStage.TRAIN_DECISION_TREE));
        assertTrue(window.getStageButtons().containsKey(AppState.WorkflowStage.SAVE_AND_USE_MODEL));
        for (var entry : window.getStageButtons().entrySet()) {
            String btnText = entry.getValue().getText();
            assertTrue(btnText.contains("Load & Analyze") || btnText.contains("Review Slots")
                    || btnText.contains("Train & Check") || btnText.contains("Save & Use"),
                "Stepper button should use simple user-facing language: " + btnText);
            assertNotNull(entry.getValue().getTooltip(), "Each workflow step should explain its purpose");
        }

        double minWidthSum = window.getSlotTableView().getTotalMinimumColumnWidth();
        double rightPaneAt1366 = 1366.0 * (1.0 - 0.46) - 40.0;
        double rightPaneAt1600 = 1600.0 * (1.0 - 0.46) - 40.0;
        assertTrue(minWidthSum < rightPaneAt1366,
            "Table minimum width (" + minWidthSum + ") must fit inside 1366x768 right pane (" + rightPaneAt1366 + ")");
        assertTrue(minWidthSum < rightPaneAt1600,
            "Table minimum width (" + minWidthSum + ") must fit inside 1600x900 right pane (" + rightPaneAt1600 + ")");
    }

    @Test
    void pkLotImportWithMissingDetectorReportsDetectorFailuresAndNeverFabricatesTenFeatureRecords() throws Exception {
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();
        svc.setVehicleDetectionService(new OnnxVehicleDetectionService(tempDir.resolve("missing-detector-import.onnx")));
        assertEquals(VehicleDetectionService.DetectorStatus.MODEL_MISSING, svc.getVehicleDetectionService().getStatus());

        MainWindow window = new MainWindow(svc, false);
        Path pkLotRoot = tempDir.resolve("PKLot_MissingDet");
        Path camDir = pkLotRoot.resolve("PUCPR/Sunny/2012-10-12");
        Files.createDirectories(camDir);
        File demoFile = SampleDataGenerator.ensureSampleImageExists(SampleDataGenerator.DEFAULT_SAMPLE_IMAGE_PATH);
        BufferedImage baseImg = ImageIO.read(demoFile);
        writePkLotImageAndXml(camDir, "2012-10-12_09_00_00", baseImg, false);

        PkLotDatasetImporter.PkLotImportResult res =
            window.getModelDetailsView().importPkLotDirectoryAsync(pkLotRoot).get(5, TimeUnit.SECONDS);
        assertNotNull(res);
        assertEquals(0, res.validImageCount());
        assertEquals(0, res.validSlotCount(), "Must never fabricate 10-feature records when ONNX detector is MODEL_MISSING");
        assertEquals(1, res.detectorFailureCount(), "Must explicitly report detector failure count");
        assertTrue(window.getAppState().getEligibleReviewedRecords().isEmpty());
    }

    private static void writePkLotImageAndXml(Path dir,
                                              String baseName,
                                              BufferedImage baseImg,
                                              boolean includeMalformedEntry) throws Exception {
        BufferedImage copy = new BufferedImage(baseImg.getWidth(), baseImg.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = copy.createGraphics();
        g.drawImage(baseImg, 0, 0, null);
        g.setColor(new Color(20, 20, 20));
        g.drawRect(0, 0, 2, 2);
        g.dispose();

        Path imgPath = dir.resolve(baseName + ".png");
        ImageIO.write(copy, "png", imgPath.toFile());

        StringBuilder xml = new StringBuilder();
        xml.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.append("<parking id=\"").append(baseName).append("\">\n");
        xml.append("  <space id=\"1\" occupied=\"1\"><contour>")
           .append("<point x=\"10\" y=\"20\"/><point x=\"90\" y=\"20\"/><point x=\"90\" y=\"120\"/><point x=\"10\" y=\"120\"/>")
           .append("</contour></space>\n");
        xml.append("  <space id=\"6\" occupied=\"1\"><contour>")
           .append("<point x=\"100\" y=\"130\"/><point x=\"180\" y=\"130\"/><point x=\"180\" y=\"230\"/><point x=\"100\" y=\"230\"/>")
           .append("</contour></space>\n");
        xml.append("  <space id=\"2\" occupied=\"0\"><contour>")
           .append("<point x=\"100\" y=\"20\"/><point x=\"180\" y=\"20\"/><point x=\"180\" y=\"120\"/><point x=\"100\" y=\"120\"/>")
           .append("</contour></space>\n");
        xml.append("  <space id=\"3\" occupied=\"0\"><contour>")
           .append("<point x=\"190\" y=\"20\"/><point x=\"270\" y=\"20\"/><point x=\"270\" y=\"120\"/><point x=\"190\" y=\"120\"/>")
           .append("</contour></space>\n");
        if (includeMalformedEntry) {
            xml.append("  <space id=\"99\" occupied=\"\"><contour><point x=\"-5\" y=\"-5\"/></contour></space>\n");
        }
        xml.append("</parking>\n");
        Files.writeString(dir.resolve(baseName + ".xml"), xml.toString(), StandardCharsets.UTF_8);
    }

    @Test
    void testFinding01AsyncExecutionOnFxThreadDoesNotBlock() throws Exception {
        ParkingAnalysisService analysisService = ParkingAnalysisService.createWithDefaultModel();
        MainWindow window = new MainWindow(analysisService, false);
        File demoFile = SampleDataGenerator.ensureSampleImageExists(SampleDataGenerator.DEFAULT_SAMPLE_IMAGE_PATH);
        window.loadImageFromFile(demoFile);

        java.util.concurrent.atomic.AtomicReference<java.util.concurrent.CompletableFuture<com.parknchillai.app.AnalysisSummary>> futureRef =
            new java.util.concurrent.atomic.AtomicReference<>();

        javafx.application.Platform.runLater(() -> {
            futureRef.set(window.runDetectionAndAnalysis());
        });

        // Wait up to 10 seconds for the future to complete from the background worker thread
        long start = System.currentTimeMillis();
        while (futureRef.get() == null && System.currentTimeMillis() - start < 3000) {
            Thread.sleep(50);
        }
        assertNotNull(futureRef.get(), "CompletableFuture should have been created on FX thread");
        com.parknchillai.app.AnalysisSummary summary = futureRef.get().get(10, TimeUnit.SECONDS);
        assertNotNull(summary, "AnalysisSummary must be produced asynchronously");
        assertNotNull(window.getAppState().getCurrentSummary(), "AppState must be updated on completion");
    }

    @Test
    void testFinding02TrainDecisionTreeAsyncExecutionOnFxThread() throws Exception {
        ParkingAnalysisService analysisService = ParkingAnalysisService.createWithDefaultModel();
        MainWindow window = new MainWindow(analysisService, false);
        File demoFile = SampleDataGenerator.ensureSampleImageExists(SampleDataGenerator.DEFAULT_SAMPLE_IMAGE_PATH);
        window.loadImageFromFile(demoFile);
        window.runDetectionAndAnalysis(true).get(10, TimeUnit.SECONDS);

        // Populate valid human-verified slots
        window.getAppState().setReviewerId("TEST_REV");
        window.getAppState().reviewSlot(1, SlotStatus.OCCUPIED);
        window.getAppState().reviewSlot(2, SlotStatus.OCCUPIED);
        window.getAppState().reviewSlot(3, SlotStatus.AVAILABLE);
        window.getAppState().reviewSlot(4, SlotStatus.AVAILABLE);

        java.util.concurrent.atomic.AtomicReference<java.util.concurrent.CompletableFuture<ModelEvaluator.EvaluationReport>> futureRef =
            new java.util.concurrent.atomic.AtomicReference<>();

        javafx.application.Platform.runLater(() -> {
            futureRef.set(window.getModelDetailsView().handleTrainDecisionTree());
        });

        long start = System.currentTimeMillis();
        while (futureRef.get() == null && System.currentTimeMillis() - start < 3000) {
            Thread.sleep(50);
        }
        assertNotNull(futureRef.get(), "CompletableFuture should have been created on FX thread");
        ModelEvaluator.EvaluationReport report = futureRef.get().get(10, TimeUnit.SECONDS);
        assertNotNull(report, "EvaluationReport must be produced asynchronously without UI lock");
    }

    @Test
    void testFinding08MainAppCleanStop() {
        MainApp app = new MainApp();
        assertDoesNotThrow(app::stop, "MainApp stop must execute cleanly without NPE");
    }
}