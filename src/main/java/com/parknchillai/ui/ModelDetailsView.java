package com.parknchillai.ui;

import com.parknchillai.ai.ClassifierBenchmark;
import com.parknchillai.ai.DatasetLoader;
import com.parknchillai.ai.ModelEvaluator;
import com.parknchillai.ai.ModelSerializer;
import com.parknchillai.ai.PkLotDatasetImporter;
import com.parknchillai.app.AppState;
import com.parknchillai.app.ParkingAnalysisService;
import com.parknchillai.detection.OnnxVehicleDetectionService;
import com.parknchillai.detection.VehicleDetectionService;
import com.parknchillai.features.FeatureVector;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;

import java.io.File;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

/**
 * Simplified, Stage-Focused Decision Tree Training, Evaluation &amp; Model Persistence View
 * (ParkNChill v2.3 Requirements 1, 2, 3, 4, 5 &amp; 6):
 * <ul>
 *   <li><strong>Stage 4 ({@code Train Decision Tree})</strong>:
 *     <ul>
 *       <li>Prominent First-Run ONNX Model Setup notice &amp; {@code Setup ONNX Model...} action
 *           when {@code models/vehicle_detector.onnx} is missing (Requirement 1).</li>
 *       <li>Explicit separation between {@code Interactive User-Reviewed Slots} and
 *           {@code Imported PKLot Dataset} pools (Requirement 2).</li>
 *       <li>Provenance breakdown distinguishing {@code IMPORTED_GROUND_TRUTH} from
 *           {@code HUMAN_VERIFIED} records (Requirement 3).</li>
 *       <li>Genuinely asynchronous PKLot import on a background worker thread with progress bar,
 *           {@code Cancel Import} support, duplicate-action disabling, and non-destructive
 *           staging (Requirement 4).</li>
 *     </ul>
 *   </li>
 *   <li><strong>Stage 5 ({@code Evaluate Model})</strong>: Held-out test partition badge,
 *       5 standard metrics including {@code False-Available Rate} with plain-English explanation,
 *       provenance summary, Confusion Matrix, and collapsed diagnostics.</li>
 *   <li><strong>Stage 6 ({@code Save and Use Model})</strong>: Save/load `.pncmodel` with
 *       10-feature vs legacy 8-feature compatibility verification.</li>
 * </ul>
 */
public class ModelDetailsView extends VBox {

    public static final String FALSE_AVAILABLE_EXPLANATION =
        "False-Available Error: An occupied parking space incorrectly classified as available.";

    private final ParkingAnalysisService service;
    private AppState appState;
    private Runnable onActiveModelChanged;
    private Runnable onReturnToImageAnalysis;

    private final VBox stage4TrainingBox = new VBox(10);
    private final VBox stage5EvaluationBox = new VBox(10);
    private final VBox stage6SaveUseBox = new VBox(10);

    // Stage 4: First-Run ONNX Setup Notice & Controls (Requirement 1)
    private final HBox onnxSetupBox = new HBox(10);
    private final Label lblOnnxSetupBanner = new Label("");
    private final Button btnSetupOnnxModel = new Button("Setup ONNX Model...");

    // Stage 4: Dataset Mode & Summary Labels (Requirements 2 & 3)
    private final ComboBox<AppState.ActiveDatasetMode> datasetModeSelector = new ComboBox<>();
    private final Label lblDatasetSourceAndSchema = new Label("Dataset: My Reviewed Slots");
    private final Label lblEligibleRecordCount = new Label("0");
    private final Label lblOccupiedSampleCount = new Label("0");
    private final Label lblAvailableSampleCount = new Label("0");
    private final Label lblProvenanceBreakdown = new Label("Dataset provenance: —");
    private final Label lblStage4ModelStatus = new Label("No 10-feature tree");
    private final Label lblStage4EvalStatus = new Label("Not evaluated yet");
    private final Label lblLegacyModelWarning = new Label(ModelSerializer.LEGACY_EIGHT_FEATURE_MESSAGE);
    private final Label lblTrainingDisabledExplanation = new Label("");
    private final Label lblPkLotSummaryBanner = new Label("PKLot Dataset: Not imported yet. Click 'Load Dataset (PKLot)...' to import annotated parking images.");
    // Visible explanation of how loading data affects (and does not automatically train) CART.
    private final Label lblDatasetImpactStatus = new Label("No training dataset loaded yet.");
    private final Label lblDatasetImpactDetails = new Label(
        "Loading a dataset prepares labelled slot records and their 10 features. It does not train or update the Decision Tree. "
        + "Click Train Decision Tree to fit the rules, then inspect the held-out evaluation.");
    private final Button btnExplainDatasetFlow = new Button("Explain Dataset → Decision Tree");
    private final ProgressBar importProgressBar = new ProgressBar(0.0);

    // Stage 4 Primary & Secondary Actions
    private final Button btnTrainDecisionTree = new Button("Train Decision Tree");
    private final Button btnImportPkLot = new Button("Import PKLot...");
    private final Button btnCancelImport = new Button("Cancel Import");
    private final Button btnSampleAnnotations = new Button("Sample Annotations...");

    // Background Async Import State (Requirement 4)
    private volatile Task<PkLotDatasetImporter.PkLotImportResult> activeImportTask;
    private volatile Thread activeImportThread;
    private volatile boolean importInProgress = false;

    // Stage 4 Collapsed Advanced Settings & Diagnostics
    private final Spinner<Integer> spinMaxDepth = new Spinner<>(1, 10, 4);
    private final Spinner<Integer> spinMinSamples = new Spinner<>(1, 10, 2);
    private final TextArea treeTextArea = new TextArea();
    private final Label lblFeatureSchemaDiagnostics = new Label("");
    private final TitledPane advModelSettingsPane = new TitledPane();

    // Stage 5 Labels & Controls (Requirement 2.F)
    private final Label lblEvalPartitionBadge = new Label("Test set —");
    private final Label lblEvalProvenanceBanner = new Label("Provenance: —");
    private final Label lblAccuracy = new Label("—");
    private final Label lblPrecision = new Label("—");
    private final Label lblRecall = new Label("—");
    private final Label lblF1 = new Label("—");
    private final Label lblFalseAvailableRate = new Label("—");
    private final Label lblFalseAvailablePlainEnglish = new Label(FALSE_AVAILABLE_EXPLANATION);
    private final Label lblConfusionTp = new Label("Occupied correctly: —");
    private final Label lblConfusionFn = new Label("Occupied predicted free: —");
    private final Label lblConfusionFp = new Label("Free predicted occupied: —");
    private final Label lblConfusionTn = new Label("Available correctly: —");
    private final Label lblTechnicalEvalLog = new Label("No evaluation diagnostics generated yet.");
    private final TitledPane advEvaluationDiagnosticsPane = new TitledPane();

    // Stage 6 Labels & Controls (Requirement 2.G)
    private final Label lblStage6ActiveModelStatus = new Label("Model status: —");
    private final Label lblStage6ActiveSchema = new Label("Features: —");
    private final Label lblStage6SaveLoadFeedback = new Label("Save your trained 10-feature Decision Tree or load a compatible saved model.");
    private final Button btnSaveModelStage6 = new Button("Save Trained Decision Tree...");
    private final Button btnLoadModelStage6 = new Button("Load Saved Decision Tree...");
    private final Button btnReturnToAnalysis = new Button("Return to Image Analysis");

    public ModelDetailsView(ParkingAnalysisService service) {
        this(service, new AppState(service));
    }

    public ModelDetailsView(ParkingAnalysisService service, AppState appState) {
        this.service = service;
        this.appState = appState != null ? appState : new AppState(service);

        setSpacing(10);
        setPadding(new Insets(10));
        VBox.setVgrow(this, Priority.ALWAYS);

        buildStage4TrainingSection();
        buildStage5EvaluationSection();
        buildStage6SaveAndUseSection();

        showForStage(AppState.WorkflowStage.TRAIN_DECISION_TREE);
        refreshModelInfo();
    }

    public void setAppState(AppState state) {
        this.appState = state;
        refreshModelInfo();
    }

    public void setOnActiveModelChanged(Runnable callback) {
        this.onActiveModelChanged = callback;
    }

    public void setOnReturnToImageAnalysis(Runnable callback) {
        this.onReturnToImageAnalysis = callback;
    }

    public void showForStage(AppState.WorkflowStage stage) {
        getChildren().clear();
        if (stage == AppState.WorkflowStage.EVALUATE_MODEL) {
            getChildren().add(stage5EvaluationBox);
        } else if (stage == AppState.WorkflowStage.SAVE_AND_USE_MODEL) {
            getChildren().add(stage6SaveUseBox);
        } else {
            getChildren().add(stage4TrainingBox);
        }
        refreshModelInfo();
    }

    // -------------------------------------------------------------------------
    // Stage 4: Train Decision Tree (Requirements 1, 2, 3, 4 & 5)
    // -------------------------------------------------------------------------

    private void buildStage4TrainingSection() {
        VBox mainCard = new VBox(10);
        mainCard.getStyleClass().add("summary-card-container");
        mainCard.setPadding(new Insets(12));

        HBox topRow = new HBox(10);
        topRow.setAlignment(Pos.CENTER_LEFT);
        Label sectionTitle = new Label("Train Decision Tree");
        sectionTitle.getStyleClass().add("card-section-title");
        HBox.setHgrow(sectionTitle, Priority.ALWAYS);

        Label modeLbl = new Label("Dataset:");
        modeLbl.setStyle("-fx-text-fill: #cbd5e1; -fx-font-weight: bold; -fx-font-size: 12px;");
        datasetModeSelector.getItems().addAll(AppState.ActiveDatasetMode.values());
        datasetModeSelector.setValue(appState.getActiveDatasetMode());
        datasetModeSelector.setPrefWidth(220);
        datasetModeSelector.setOnAction(e -> {
            if (datasetModeSelector.getValue() != null && datasetModeSelector.getValue() != appState.getActiveDatasetMode()) {
                appState.setActiveDatasetMode(datasetModeSelector.getValue());
                refreshModelInfo();
            }
        });
        topRow.getChildren().addAll(sectionTitle, modeLbl, datasetModeSelector);

        // First-Run ONNX Model Setup Banner (Requirement 1)
        onnxSetupBox.setAlignment(Pos.CENTER_LEFT);
        onnxSetupBox.setPadding(new Insets(8, 12, 8, 12));
        onnxSetupBox.setStyle(
            "-fx-background-color: #3f2e1e; -fx-border-color: #f59e0b; " +
            "-fx-border-radius: 6px; -fx-background-radius: 6px;"
        );
        lblOnnxSetupBanner.setWrapText(true);
        lblOnnxSetupBanner.setStyle("-fx-text-fill: #fde68a; -fx-font-size: 12px; -fx-font-weight: bold;");
        HBox.setHgrow(lblOnnxSetupBanner, Priority.ALWAYS);

        btnSetupOnnxModel.getStyleClass().add("btn-secondary");
        btnSetupOnnxModel.setMinWidth(145);
        btnSetupOnnxModel.setOnAction(e -> handleSetupOnnxModelDialog());
        onnxSetupBox.getChildren().addAll(lblOnnxSetupBanner, btnSetupOnnxModel);

        lblLegacyModelWarning.setWrapText(true);
        lblLegacyModelWarning.setText("An older 8-feature model is available. Train or load a 10-feature model to continue.");
        lblLegacyModelWarning.setTooltip(new Tooltip(ModelSerializer.LEGACY_EIGHT_FEATURE_MESSAGE));
        lblLegacyModelWarning.setStyle(
            "-fx-background-color: #78350f; -fx-text-fill: #fde68a; -fx-padding: 8px 12px; " +
            "-fx-background-radius: 6px; -fx-font-size: 12px; -fx-font-weight: bold;"
        );

        lblDatasetSourceAndSchema.setWrapText(true);
        lblDatasetSourceAndSchema.setStyle("-fx-text-fill: #38bdf8; -fx-font-size: 12.5px; -fx-font-weight: bold;");

        GridPane summaryGrid = new GridPane();
        summaryGrid.setHgap(10);
        summaryGrid.setVgap(8);

        addMetricCell(summaryGrid, "Eligible Samples (all partitions)", lblEligibleRecordCount, 0);
        addMetricCell(summaryGrid, "Occupied / Available", lblOccupiedSampleCount, 1);
        addMetricCell(summaryGrid, "Decision Tree", lblStage4ModelStatus, 2);

        lblTrainingDisabledExplanation.setWrapText(true);
        lblTrainingDisabledExplanation.setStyle(
            "-fx-background-color: #1e293b; -fx-text-fill: #fbbf24; -fx-padding: 7px 10px; " +
            "-fx-border-color: #f59e0b; -fx-border-radius: 5px; -fx-background-radius: 5px; -fx-font-size: 12px;"
        );

        HBox actionsRow = new HBox(10);
        actionsRow.setAlignment(Pos.CENTER_LEFT);

        btnTrainDecisionTree.getStyleClass().add("btn-primary");
        btnTrainDecisionTree.setOnAction(e -> handleTrainDecisionTree());

        btnImportPkLot.getStyleClass().add("btn-secondary");
        btnImportPkLot.setText("Load Dataset (PKLot)...");
        btnImportPkLot.setTooltip(new Tooltip("Load annotated PKLot images, labels, slot geometry, and 10-feature records. Loading prepares data; it does not train the Decision Tree."));
        btnImportPkLot.setOnAction(e -> handleBrowseAndImportPkLot());

        btnCancelImport.getStyleClass().add("btn-tool");
        btnCancelImport.setStyle("-fx-background-color: #7f1d1d; -fx-text-fill: #fecaca; -fx-font-weight: bold;");
        btnCancelImport.setVisible(false);
        btnCancelImport.setManaged(false);
        btnCancelImport.setOnAction(e -> cancelActiveImport());

        btnSampleAnnotations.getStyleClass().add("btn-tool");
        btnSampleAnnotations.setTooltip(new Tooltip("Read-only inspection of PKLot XML/CSV annotations (does not fabricate training features)"));
        btnSampleAnnotations.setOnAction(e -> handleBrowseAndInspectPkLotAnnotations());

        btnExplainDatasetFlow.getStyleClass().add("btn-tool");
        btnExplainDatasetFlow.setOnAction(e -> showDatasetImpactDialog());
        actionsRow.getChildren().addAll(
            btnTrainDecisionTree, btnImportPkLot, btnCancelImport, btnSampleAnnotations
        );

        importProgressBar.setMaxWidth(Double.MAX_VALUE);
        importProgressBar.setPrefHeight(8);
        importProgressBar.setVisible(false);
        importProgressBar.setManaged(false);

        lblPkLotSummaryBanner.setWrapText(true);
        lblPkLotSummaryBanner.setStyle(
            "-fx-background-color: #0f172a; -fx-text-fill: #cbd5e1; -fx-padding: 8px 10px; " +
            "-fx-background-radius: 5px; -fx-font-size: 12px;"
        );

        lblDatasetImpactStatus.setWrapText(true);
        lblDatasetImpactStatus.setStyle("-fx-text-fill: #f8fafc; -fx-font-weight: bold; -fx-font-size: 13px;");
        lblDatasetImpactDetails.setWrapText(true);
        lblDatasetImpactDetails.setStyle("-fx-text-fill: #cbd5e1; -fx-font-size: 12px;");
        VBox datasetImpactCard = new VBox(5,
            new Label("What Loading This Dataset Does to the Decision Tree"),
            lblDatasetImpactStatus,
            lblDatasetImpactDetails,
            btnExplainDatasetFlow
        );
        datasetImpactCard.setPadding(new Insets(9, 11, 9, 11));
        datasetImpactCard.setStyle("-fx-background-color: #111c30; -fx-border-color: #38bdf8; -fx-border-radius: 6px; -fx-background-radius: 6px;");
        Label datasetImpactHeading = (Label) datasetImpactCard.getChildren().get(0);
        datasetImpactHeading.setStyle("-fx-text-fill: #38bdf8; -fx-font-weight: bold; -fx-font-size: 12.5px;");

        mainCard.getChildren().addAll(
            topRow,
            onnxSetupBox,
            lblLegacyModelWarning,
            lblDatasetSourceAndSchema,
            summaryGrid,
            lblTrainingDisabledExplanation,
            actionsRow,
            importProgressBar,
            lblPkLotSummaryBanner,
            datasetImpactCard
        );

        // Collapsed-by-default Advanced Model Settings & Diagnostics
        VBox advContent = new VBox(8);
        advContent.setPadding(new Insets(10));
        advContent.setStyle("-fx-background-color: #0f172a;");

        HBox hyperRow = new HBox(12);
        hyperRow.setAlignment(Pos.CENTER_LEFT);
        Label lblDepth = new Label("Max Tree Depth:");
        lblDepth.setStyle("-fx-text-fill: #cbd5e1; -fx-font-weight: bold;");
        spinMaxDepth.setPrefWidth(72);

        Label lblMinLeaf = new Label("Min Samples per Leaf:");
        lblMinLeaf.setStyle("-fx-text-fill: #cbd5e1; -fx-font-weight: bold;");
        spinMinSamples.setPrefWidth(72);
        hyperRow.getChildren().addAll(lblDepth, spinMaxDepth, lblMinLeaf, spinMinSamples);

        lblProvenanceBreakdown.setStyle("-fx-text-fill: #cbd5e1; -fx-font-size: 11.5px;");
        lblStage4EvalStatus.setStyle("-fx-text-fill: #cbd5e1; -fx-font-size: 11.5px;");
        lblFeatureSchemaDiagnostics.setWrapText(true);
        lblFeatureSchemaDiagnostics.setStyle("-fx-text-fill: #94a3b8; -fx-font-size: 11.5px; -fx-font-family: monospace;");
        lblFeatureSchemaDiagnostics.setText(
            String.format(Locale.ROOT,
                "10-Feature Schema (%s, Policy: %s): [0] meanIntensity, [1] foregroundRatio, [2] darkPixelRatio, "
                + "[3] edgeDensity, [4] objectArea, [5] meanSaturation, [6] saturationRatio, [7] largestBlobRatio, "
                + "[8] vehicleOverlapRatio (Polygon/Rect IoS), [9] vehicleMatchedFlag (0/1)",
                com.parknchillai.imageprocessing.ProcessingPolicy.CANONICAL_SCHEMA_ID,
                com.parknchillai.imageprocessing.ProcessingPolicy.CANONICAL_POLICY_ID)
        );

        treeTextArea.setEditable(false);
        treeTextArea.setPrefRowCount(7);
        treeTextArea.setStyle(
            "-fx-font-family: 'Consolas', 'Courier New', monospace; -fx-font-size: 12px; " +
            "-fx-control-inner-background: #0b1120; -fx-text-fill: #38bdf8;"
        );

        lblProvenanceBreakdown.setStyle("-fx-text-fill: #cbd5e1; -fx-font-size: 11.5px;");
        lblStage4EvalStatus.setStyle("-fx-text-fill: #cbd5e1; -fx-font-size: 11.5px;");
        advContent.getChildren().addAll(
            hyperRow,
            lblProvenanceBreakdown,
            lblStage4EvalStatus,
            new Label("Current active Decision Tree rules (these change only after training or loading a compatible model):"),
            treeTextArea
        );

        advModelSettingsPane.setText("Advanced Settings (optional)");
        advModelSettingsPane.setContent(advContent);
        advModelSettingsPane.setCollapsible(true);
        advModelSettingsPane.setExpanded(false);

        stage4TrainingBox.getChildren().addAll(mainCard, advModelSettingsPane);
    }

    // -------------------------------------------------------------------------
    // Stage 5: Evaluate Model (Requirement 2.F & 3)
    // -------------------------------------------------------------------------

    private void buildStage5EvaluationSection() {
        VBox evalCard = new VBox(10);
        evalCard.getStyleClass().add("summary-card-container");
        evalCard.setPadding(new Insets(14));

        HBox header = new HBox(10);
        header.setAlignment(Pos.CENTER_LEFT);
        Label title = new Label("Decision Tree Results");
        title.getStyleClass().add("card-section-title");
        HBox.setHgrow(title, Priority.ALWAYS);

        lblEvalPartitionBadge.setStyle(
            "-fx-background-color: #064e3b; -fx-text-fill: #a7f3d0; -fx-font-weight: bold; " +
            "-fx-font-size: 11.5px; -fx-padding: 4px 10px; -fx-background-radius: 12px;"
        );
        header.getChildren().addAll(title, lblEvalPartitionBadge);

        lblEvalProvenanceBanner.setWrapText(true);
        lblEvalProvenanceBanner.setStyle(
            "-fx-background-color: #0f172a; -fx-text-fill: #38bdf8; -fx-padding: 6px 10px; " +
            "-fx-background-radius: 5px; -fx-font-size: 12px; -fx-font-weight: bold;"
        );

        GridPane metricsGrid = new GridPane();
        metricsGrid.setHgap(10);
        metricsGrid.setVgap(8);

        addMetricCell(metricsGrid, "Accuracy", lblAccuracy, 0);
        addMetricCell(metricsGrid, "Precision", lblPrecision, 1);
        addMetricCell(metricsGrid, "Occupied Recall", lblRecall, 2);
        addMetricCell(metricsGrid, "F1 Score", lblF1, 3);
        addMetricCell(metricsGrid, "False-Available Rate", lblFalseAvailableRate, 4);

        lblFalseAvailablePlainEnglish.setWrapText(true);
        lblFalseAvailablePlainEnglish.setStyle(
            "-fx-background-color: #1e293b; -fx-text-fill: #fde68a; -fx-padding: 7px 10px; " +
            "-fx-background-radius: 5px; -fx-font-size: 12.5px; -fx-font-weight: bold;"
        );

        VBox matrixBox = new VBox(6);
        matrixBox.setPadding(new Insets(10));
        matrixBox.setStyle("-fx-background-color: #0f172a; -fx-border-color: #334155; -fx-border-radius: 6px; -fx-background-radius: 6px;");
        Label matrixTitle = new Label("Test Results Breakdown");
        matrixTitle.setStyle("-fx-font-weight: bold; -fx-text-fill: #f8fafc; -fx-font-size: 12.5px;");

        GridPane cmGrid = new GridPane();
        cmGrid.setHgap(16);
        cmGrid.setVgap(6);
        lblConfusionTp.setStyle("-fx-text-fill: #34d399; -fx-font-weight: bold; -fx-font-size: 13px;");
        lblConfusionFn.setStyle("-fx-text-fill: #f87171; -fx-font-weight: bold; -fx-font-size: 13px;");
        lblConfusionFp.setStyle("-fx-text-fill: #fbbf24; -fx-font-weight: bold; -fx-font-size: 13px;");
        lblConfusionTn.setStyle("-fx-text-fill: #38bdf8; -fx-font-weight: bold; -fx-font-size: 13px;");
        cmGrid.add(lblConfusionTp, 0, 0);
        cmGrid.add(lblConfusionFn, 1, 0);
        cmGrid.add(lblConfusionFp, 0, 1);
        cmGrid.add(lblConfusionTn, 1, 1);
        matrixBox.getChildren().addAll(matrixTitle, cmGrid);

        evalCard.getChildren().addAll(
            header, lblEvalProvenanceBanner, metricsGrid, lblFalseAvailablePlainEnglish, matrixBox
        );

        VBox diagContent = new VBox(8);
        diagContent.setPadding(new Insets(10));
        diagContent.setStyle("-fx-background-color: #0f172a;");

        HBox diagButtons = new HBox(10);
        Button btnEvalValidation = new Button("Inspect Validation Partition");
        btnEvalValidation.getStyleClass().add("btn-tool");
        btnEvalValidation.setOnAction(e -> {
            try {
                ModelEvaluator.EvaluationReport valRep = service.evaluateValidationPartition();
                lblTechnicalEvalLog.setText("Validation Partition Report:\n" + valRep.toString());
            } catch (Exception ex) {
                lblTechnicalEvalLog.setText("Validation error: " + ex.getMessage());
            }
        });

        Button btnRunBenchmark = new Button("Compare CART vs Random Forest Baseline");
        btnRunBenchmark.getStyleClass().add("btn-tool");
        btnRunBenchmark.setOnAction(e -> {
            try {
                ClassifierBenchmark.BenchmarkResult bench = service.runClassifierBenchmark();
                lblTechnicalEvalLog.setText(bench.formatSummaryTable());
            } catch (Exception ex) {
                lblTechnicalEvalLog.setText("Baseline comparison error: " + ex.getMessage());
            }
        });
        diagButtons.getChildren().addAll(btnEvalValidation, btnRunBenchmark);

        lblTechnicalEvalLog.setWrapText(true);
        lblTechnicalEvalLog.setStyle(
            "-fx-font-family: 'Consolas', 'Courier New', monospace; -fx-font-size: 11.5px; -fx-text-fill: #cbd5e1;"
        );

        diagContent.getChildren().addAll(diagButtons, lblTechnicalEvalLog);
        advEvaluationDiagnosticsPane.setText("Advanced Evaluation Diagnostics (Partition Details, Baseline Comparisons & Technical Logs)");
        advEvaluationDiagnosticsPane.setContent(diagContent);
        advEvaluationDiagnosticsPane.setCollapsible(true);
        advEvaluationDiagnosticsPane.setExpanded(false);

        stage5EvaluationBox.getChildren().addAll(evalCard, advEvaluationDiagnosticsPane);
    }

    // -------------------------------------------------------------------------
    // Stage 6: Save and Use Model (Requirement 2.G)
    // -------------------------------------------------------------------------

    private void buildStage6SaveAndUseSection() {
        VBox card = new VBox(12);
        card.getStyleClass().add("summary-card-container");
        card.setPadding(new Insets(16));

        Label title = new Label("Save and Use Decision Tree Model");
        title.getStyleClass().add("card-section-title");

        VBox statusBox = new VBox(6);
        statusBox.setPadding(new Insets(10));
        statusBox.setStyle("-fx-background-color: #0f172a; -fx-border-color: #334155; -fx-border-radius: 6px; -fx-background-radius: 6px;");

        lblStage6ActiveModelStatus.setWrapText(true);
        lblStage6ActiveModelStatus.setStyle("-fx-text-fill: #f8fafc; -fx-font-weight: bold; -fx-font-size: 13.5px;");

        lblStage6ActiveSchema.setWrapText(true);
        lblStage6ActiveSchema.setStyle("-fx-text-fill: #38bdf8; -fx-font-weight: bold; -fx-font-size: 12.5px;");

        lblStage6SaveLoadFeedback.setWrapText(true);
        lblStage6SaveLoadFeedback.setStyle("-fx-text-fill: #cbd5e1; -fx-font-size: 12.5px;");

        statusBox.getChildren().addAll(lblStage6ActiveModelStatus, lblStage6ActiveSchema, lblStage6SaveLoadFeedback);

        HBox buttonBar = new HBox(12);
        buttonBar.setAlignment(Pos.CENTER_LEFT);

        btnSaveModelStage6.getStyleClass().add("btn-primary");
        btnSaveModelStage6.setOnAction(e -> handleSaveModelDialog());

        btnLoadModelStage6.getStyleClass().add("btn-secondary");
        btnLoadModelStage6.setOnAction(e -> handleLoadModelDialog());

        btnReturnToAnalysis.getStyleClass().add("btn-success");
        btnReturnToAnalysis.setOnAction(e -> {
            if (onReturnToImageAnalysis != null) {
                onReturnToImageAnalysis.run();
            }
        });

        buttonBar.getChildren().addAll(btnSaveModelStage6, btnLoadModelStage6, btnReturnToAnalysis);
        card.getChildren().addAll(title, statusBox, buttonBar);
        stage6SaveUseBox.getChildren().add(card);
    }

    private void addMetricCell(GridPane grid, String name, Label label, int col) {
        VBox cell = new VBox(2);
        cell.setPadding(new Insets(6, 10, 6, 10));
        cell.getStyleClass().add("feature-cell");
        Label lblName = new Label(name);
        lblName.getStyleClass().add("feature-name");
        label.getStyleClass().add("feature-value");
        cell.getChildren().addAll(lblName, label);
        GridPane.setHgrow(cell, Priority.ALWAYS);
        grid.add(cell, col, 0);
    }

    private static String simplifyTrainingMessage(String detail) {
        String text = detail == null ? "" : detail.toLowerCase(Locale.ROOT);
        if (text.contains("onnx") || text.contains("detector")) {
            return "Install the vehicle detector before importing PKLot for training.";
        }
        if (text.contains("partition") || text.contains("capture group") || text.contains("group")) {
            return "Choose the top-level extracted PKLot folder—not one date/weather folder. Include multiple dates and camera folders (PUCPR, UFPR04, UFPR05), then click Import PKLot again. "
                + (detail == null ? "" : detail);
        }
        if (text.contains("occupied") && text.contains("available")) {
            return "Training requires at least 4 confirmed 10-feature records. "
                + "Add reviewed examples of both Occupied and Available spaces.";
        }
        if (text.contains("review") || text.contains("confirmed") || text.contains("eligible")) {
            return "Review more parking slots before training.";
        }
        if (detail == null || detail.isBlank()) return "The dataset is not ready for training.";
        return detail.length() > 120 ? detail.substring(0, 117) + "…" : detail;
    }

    // -------------------------------------------------------------------------
    // Actions: First-Run ONNX Setup, Async PKLot Import/Cancel, Train, Save & Load
    // -------------------------------------------------------------------------

    private void handleSetupOnnxModelDialog() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Select Pretrained ONNX Vehicle Detector Model (.onnx)");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("ONNX Model (*.onnx)", "*.onnx"));
        File file = fc.showOpenDialog(getScene() != null ? getScene().getWindow() : null);
        if (file != null) {
            installOnnxModelFromFile(file.toPath());
        }
    }

    public OnnxVehicleDetectionService.ModelSetupResult installOnnxModelFromFile(Path sourceOnnxFile) {
        try {
            OnnxVehicleDetectionService.ModelSetupResult res =
                OnnxVehicleDetectionService.installAndVerifyModel(
                    sourceOnnxFile, Path.of(OnnxVehicleDetectionService.DEFAULT_MODEL_PATH));
            if (res.validOnnxHeader()) {
                service.setVehicleDetectionService(new OnnxVehicleDetectionService(res.installedPath()));
            }
            lblPkLotSummaryBanner.setText(res.message());
            refreshModelInfo();
            return res;
        } catch (Exception ex) {
            lblPkLotSummaryBanner.setText("ONNX model setup failed: " + ex.getMessage());
            throw new IllegalStateException("ONNX model setup failed: " + ex.getMessage(), ex);
        }
    }

    public CompletableFuture<ModelEvaluator.EvaluationReport> handleTrainDecisionTree() {
        int depth = spinMaxDepth.getValue();
        int minLeaf = spinMinSamples.getValue();

        String preflightReason = appState.getTrainingDisabledReason();
        if (preflightReason != null) {
            lblDatasetImpactStatus.setText("TRAINING BLOCKED — dataset loaded, but it does not meet the training requirements.");
            lblDatasetImpactDetails.setText(simplifyTrainingMessage(preflightReason) + " Full reason: " + preflightReason);
            refreshModelInfo();
            java.util.concurrent.CompletableFuture<ModelEvaluator.EvaluationReport> blocked = new java.util.concurrent.CompletableFuture<>();
            blocked.completeExceptionally(new IllegalStateException(preflightReason));
            return blocked;
        }

        // A new training attempt invalidates metrics from the previously active model.
        appState.setLastEvaluationReport(null);
        lblDatasetImpactStatus.setText("TRAINING IN PROGRESS — CART is fitting candidate trees from the active dataset.");
        lblDatasetImpactDetails.setText("PKLot uses TRAIN to fit, VALIDATION to select settings, and frozen TEST for final evaluation. "
            + "The dataset is not itself the model; the learned decision rules update only when training completes successfully.");
        lblStage4EvalStatus.setText("Training in progress — previous evaluation is no longer current.");
        btnTrainDecisionTree.setDisable(true);
        CompletableFuture<ModelEvaluator.EvaluationReport> future = new CompletableFuture<>();

        if (!Platform.isFxApplicationThread()) {
            try {
                ModelEvaluator.EvaluationReport report = appState.trainTenFeatureDecisionTree(depth, minLeaf);
                displayReport(report);
                lblDatasetImpactStatus.setText(buildDatasetImpactStatus());
                lblDatasetImpactDetails.setText(buildDatasetImpactDetails());
                refreshModelInfo();
                advModelSettingsPane.setExpanded(true);
                if (onActiveModelChanged != null) {
                    onActiveModelChanged.run();
                }
                future.complete(report);
            } catch (Exception ex) {
                future.completeExceptionally(ex);
            } finally {
                btnTrainDecisionTree.setDisable(false);
            }
            return future;
        }

        Task<ModelEvaluator.EvaluationReport> task = new Task<>() {
            @Override
            protected ModelEvaluator.EvaluationReport call() throws Exception {
                return appState.trainTenFeatureDecisionTree(depth, minLeaf);
            }
        };

        task.setOnSucceeded(e -> {
            btnTrainDecisionTree.setDisable(false);
            ModelEvaluator.EvaluationReport report = task.getValue();
            displayReport(report);
            lblDatasetImpactStatus.setText(buildDatasetImpactStatus());
            lblDatasetImpactDetails.setText(buildDatasetImpactDetails());
            refreshModelInfo();
            advModelSettingsPane.setExpanded(true);
            if (onActiveModelChanged != null) {
                onActiveModelChanged.run();
            }
            future.complete(report);
        });

        task.setOnFailed(e -> {
            btnTrainDecisionTree.setDisable(false);
            Throwable ex = task.getException();
            refreshModelInfo();
            lblDatasetImpactStatus.setText("TRAINING FAILED — no new model was confirmed. The loaded dataset remains selected.");
            lblDatasetImpactDetails.setText((ex == null ? "Unknown training error." : ex.getMessage())
                + " Correct the prerequisite or dataset issue, then try Train Decision Tree again.");
            Alert alert = new Alert(Alert.AlertType.ERROR, ex.getMessage(), ButtonType.OK);
            alert.setTitle("Training Cannot Proceed");
            alert.setHeaderText("Decision Tree Training Blocked");
            alert.showAndWait();
            future.completeExceptionally(ex);
        });

        Thread thread = new Thread(task, "PNC-Tree-Training-Worker");
        thread.setDaemon(true);
        thread.start();

        return future;
    }

    private void handleBrowseAndImportPkLot() {
        DirectoryChooser dc = new DirectoryChooser();
        dc.setTitle("Select PKLot Dataset Directory");
        Path defaultDir = PkLotDatasetImporter.resolveDefaultPkLotDirectory();
        if (defaultDir != null) {
            dc.setInitialDirectory(defaultDir.toFile());
        }
        File dir = dc.showDialog(getScene() != null ? getScene().getWindow() : null);
        if (dir != null) {
            importPkLotDirectoryAsync(dir.toPath());
        }
    }

    /**
     * Launches PKLot dataset import genuinely asynchronously on a dedicated background daemon
     * thread ({@code ParkNChill-PKLot-Import-Worker}), updating UI state on the JavaFX Application
     * Thread, disabling duplicate import actions while running, supporting cancellation via
     * {@link #cancelActiveImport()}, and ensuring a failed or cancelled import never partially
     * replaces the previously valid dataset (Requirement 4).
     */
    public CompletableFuture<PkLotDatasetImporter.PkLotImportResult> importPkLotDirectoryAsync(Path rootDir) {
        if (importInProgress) {
            CompletableFuture<PkLotDatasetImporter.PkLotImportResult> rejected = new CompletableFuture<>();
            rejected.completeExceptionally(new IllegalStateException("A PKLot import is already in progress."));
            return rejected;
        }

        importInProgress = true;
        btnImportPkLot.setDisable(true);
        btnSampleAnnotations.setDisable(true);
        btnTrainDecisionTree.setDisable(true);
        btnCancelImport.setVisible(true);
        btnCancelImport.setManaged(true);
        btnCancelImport.setDisable(false);
        importProgressBar.setVisible(true);
        importProgressBar.setManaged(true);
        importProgressBar.setProgress(0.0);
        lblPkLotSummaryBanner.setText("Importing PKLot dataset asynchronously from " + rootDir.getFileName() + "...");
        lblDatasetImpactStatus.setText("LOADING DATASET — reading source images, XML slot labels, feature vectors, and capture groups.");
        lblDatasetImpactDetails.setText("The existing Decision Tree is not being trained or changed during import. When loading finishes, this panel will show eligible rows, class balance, partition counts, and the required next action.");

        CompletableFuture<PkLotDatasetImporter.PkLotImportResult> future = new CompletableFuture<>();

        Task<PkLotDatasetImporter.PkLotImportResult> task = new Task<>() {
            @Override
            protected PkLotDatasetImporter.PkLotImportResult call() throws Exception {
                return PkLotDatasetImporter.importDirectory(
                    rootDir,
                    service.getVehicleDetectionService(),
                    service.getOverlapPolicy(),
                    service.isUseAdaptiveThreshold(),
                    p -> Platform.runLater(() -> {
                        importProgressBar.setProgress(p);
                        updateProgress((long) (p * 100), 100);
                    }),
                    this::isCancelled
                );
            }
        };

        task.setOnSucceeded(e -> {
            restoreUiAfterImportFinished();
            PkLotDatasetImporter.PkLotImportResult res = task.getValue();
            appState.registerPkLotImportResult(res);
            datasetModeSelector.setValue(appState.getActiveDatasetMode());
            lblPkLotSummaryBanner.setText(res.conciseSummary()
                + "\nNext Action: " + res.recommendedNextAction()
                + "\nDecision Tree effect: dataset is loaded and selected; the active tree has NOT been retrained. "
                + "Click Train Decision Tree after the partition and class requirements are satisfied.");
            refreshModelInfo();
            future.complete(res);
        });

        task.setOnCancelled(e -> {
            restoreUiAfterImportFinished();
            lblPkLotSummaryBanner.setText("Dataset load cancelled. The previous valid dataset remains intact.");
            refreshModelInfo();
            lblDatasetImpactStatus.setText("DATASET LOAD CANCELLED — previous valid dataset preserved.");
            lblDatasetImpactDetails.setText(buildDatasetImpactDetails());
            future.completeExceptionally(new CancellationException("PKLot import cancelled."));
        });

        task.setOnFailed(e -> {
            restoreUiAfterImportFinished();
            Throwable ex = task.getException();
            if (ex instanceof CancellationException) {
                lblPkLotSummaryBanner.setText("Dataset load cancelled. The previous valid dataset remains intact.");
            } else {
                lblPkLotSummaryBanner.setText("Dataset load failed (previous dataset preserved): "
                    + (ex != null ? ex.getMessage() : "Unknown error"));
            }
            refreshModelInfo();
            lblDatasetImpactStatus.setText("DATASET LOAD FAILED — previous valid dataset preserved.");
            lblDatasetImpactDetails.setText((ex != null ? ex.getMessage() : "Unknown import error")
                + " Correct the folder/model issue and load the dataset again.");
            future.completeExceptionally(ex != null ? ex : new RuntimeException("PKLot import failed"));
        });

        this.activeImportTask = task;
        Thread worker = new Thread(task, "ParkNChill-PKLot-Import-Worker");
        worker.setDaemon(true);
        this.activeImportThread = worker;
        worker.start();

        return future;
    }

    /**
     * Cancels the currently running asynchronous PKLot import without modifying the previously
     * valid dataset in {@link AppState} (Requirement 4).
     */
    public void cancelActiveImport() {
        Task<PkLotDatasetImporter.PkLotImportResult> task = this.activeImportTask;
        if (task != null && importInProgress) {
            task.cancel(true);
        }
    }

    public boolean isImportInProgress() {
        return importInProgress;
    }

    private void restoreUiAfterImportFinished() {
        importInProgress = false;
        activeImportTask = null;
        activeImportThread = null;
        btnImportPkLot.setDisable(false);
        btnSampleAnnotations.setDisable(false);
        btnCancelImport.setVisible(false);
        btnCancelImport.setManaged(false);
        importProgressBar.setVisible(false);
        importProgressBar.setManaged(false);
    }

    private void handleBrowseAndInspectPkLotAnnotations() {
        DirectoryChooser dc = new DirectoryChooser();
        dc.setTitle("Select PKLot Directory to Inspect Annotations");
        Path defaultDir = PkLotDatasetImporter.resolveDefaultPkLotDirectory();
        if (defaultDir != null) {
            dc.setInitialDirectory(defaultDir.toFile());
        }
        File dir = dc.showDialog(getScene() != null ? getScene().getWindow() : null);
        if (dir != null) {
            inspectPkLotAnnotations(dir.toPath());
        }
    }

    public PkLotDatasetImporter.AnnotationInspectionReport inspectPkLotAnnotations(Path rootDir) {
        try {
            PkLotDatasetImporter.AnnotationInspectionReport report =
                PkLotDatasetImporter.inspectSampleAnnotations(rootDir, 8);
            appState.setLastPkLotInspectionReport(report);
            StringBuilder sb = new StringBuilder(report.conciseSummary());
            if (!report.sampleEntries().isEmpty()) {
                sb.append("\nSample Entries: ").append(String.join(" ; ", report.sampleEntries()));
            }
            lblPkLotSummaryBanner.setText(sb.toString());
            return report;
        } catch (Exception ex) {
            lblPkLotSummaryBanner.setText("Annotation inspection failed: " + ex.getMessage());
            return null;
        }
    }

    private void handleSaveModelDialog() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Save Trained Decision Tree Model");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("ParkNChill Model (*.pncmodel)", "*.pncmodel"));
        fc.setInitialFileName("parknchill_v23_cart_10feat.pncmodel");
        File file = fc.showSaveDialog(getScene() != null ? getScene().getWindow() : null);
        if (file != null) {
            saveActiveModelToFile(file.toPath());
        }
    }

    public ModelSerializer.LoadedModelBundle saveActiveModelToFile(Path targetFile) {
        try {
            ModelSerializer.LoadedModelBundle bundle = ModelSerializer.saveModel(
                service.getPredictor().getModel(),
                targetFile,
                appState.getDatasetSourceDescription()
            );
            lblStage6SaveLoadFeedback.setText("Saved model to " + targetFile.getFileName() + " (" + bundle.statusSummary() + ").");
            refreshModelInfo();
            return bundle;
        } catch (Exception ex) {
            lblStage6SaveLoadFeedback.setText("Save failed: " + ex.getMessage());
            throw new IllegalStateException("Failed to save model: " + ex.getMessage(), ex);
        }
    }

    private void handleLoadModelDialog() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Load Saved Decision Tree Model");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("ParkNChill Model (*.pncmodel)", "*.pncmodel"));
        File file = fc.showOpenDialog(getScene() != null ? getScene().getWindow() : null);
        if (file != null) {
            loadModelFromFile(file.toPath());
        }
    }

    public ModelSerializer.LoadedModelBundle loadModelFromFile(Path sourceFile) {
        try {
            ModelSerializer.LoadedModelBundle bundle = ModelSerializer.loadModel(sourceFile);
            appState.applyLoadedModelBundle(bundle);
            lblStage6SaveLoadFeedback.setText(bundle.compatibilityMessage()
                + " Loading activates existing rules only; it does not train or replace current detection results. "
                + "Return to Review Slots and click 'Run Tree Separately' to show its prediction in a separate column.");
            refreshModelInfo();
            advModelSettingsPane.setExpanded(true);
            if (onActiveModelChanged != null) {
                onActiveModelChanged.run();
            }
            return bundle;
        } catch (Exception ex) {
            lblStage6SaveLoadFeedback.setText("Load failed: " + ex.getMessage());
            throw new IllegalStateException("Failed to load model: " + ex.getMessage(), ex);
        }
    }

    /**
     * Builds a user-facing explanation of the active dataset's exact effect on CART.
     * Dataset loading is deliberately kept separate from training and model loading.
     */
    private String buildDatasetImpactDetails() {
        java.util.List<com.parknchillai.ai.SlotTrainingRecord> records = appState.getEligibleReviewedRecords();
        int occupied = appState.getEligibleOccupiedCount();
        int available = appState.getEligibleAvailableCount();
        java.util.Map<String, Integer> rowsByPartition = new java.util.LinkedHashMap<>();
        java.util.Map<String, java.util.Set<String>> groupsByPartition = new java.util.LinkedHashMap<>();
        for (String part : java.util.List.of("TRAIN", "VALIDATION", "TEST", "UNASSIGNED")) {
            rowsByPartition.put(part, 0);
            groupsByPartition.put(part, new java.util.LinkedHashSet<>());
        }
        for (com.parknchillai.ai.SlotTrainingRecord record : records) {
            String part = record.partition() == null ? "UNASSIGNED" : record.partition().trim().toUpperCase(Locale.ROOT);
            if ("VAL".equals(part)) part = "VALIDATION";
            if (!rowsByPartition.containsKey(part)) part = "UNASSIGNED";
            rowsByPartition.put(part, rowsByPartition.get(part) + 1);
            if (record.captureGroup() != null && !record.captureGroup().isBlank()) {
                groupsByPartition.get(part).add(record.captureGroup());
            }
        }

        StringBuilder sb = new StringBuilder();
        sb.append("Active source: ").append(appState.getDatasetSourceDescription()).append(". ");
        sb.append("Eligible current-schema records: ").append(records.size())
          .append(" (Occupied=").append(occupied).append(", Available=").append(available).append("). ");
        sb.append("Provenance: ").append(appState.getEligibleImportedGroundTruthCount())
          .append(" imported ground-truth labels; ").append(appState.getEligibleHumanVerifiedCount())
          .append(" human-reviewed labels. ");

        com.parknchillai.ai.PkLotDatasetImporter.PkLotImportResult imported = appState.getLastPkLotImportResult();
        if (appState.getActiveDatasetMode() == AppState.ActiveDatasetMode.PKLOT_IMPORTED_DATASET && imported != null) {
            sb.append("Import scan: ").append(imported.validImageCount()).append(" valid images, ")
              .append(imported.validSlotCount()).append(" valid annotated slots, ")
              .append(imported.invalidRecordCount()).append(" invalid records, ")
              .append(imported.detectorFailureCount()).append(" detector failures. ");
            sb.append("Saved partition counts: TRAIN ").append(rowsByPartition.get("TRAIN"))
              .append(" rows/").append(groupsByPartition.get("TRAIN").size()).append(" groups; VALIDATION ")
              .append(rowsByPartition.get("VALIDATION")).append(" rows/").append(groupsByPartition.get("VALIDATION").size())
              .append(" groups; TEST ").append(rowsByPartition.get("TEST")).append(" rows/")
              .append(groupsByPartition.get("TEST").size()).append(" groups; unassigned ")
              .append(rowsByPartition.get("UNASSIGNED")).append(" rows.");
        } else {
            java.util.Set<String> groups = new java.util.LinkedHashSet<>();
            for (com.parknchillai.ai.SlotTrainingRecord record : records) {
                if (record.captureGroup() != null && !record.captureGroup().isBlank()) groups.add(record.captureGroup());
            }
            sb.append("Independent capture groups currently available: ").append(groups.size())
              .append(". Partition assignment for reviewed examples is performed during the training workflow.");
        }
        sb.append("\nActive tree: ")
          .append(appState.isCompatibleTenFeatureModelActive() ? "compatible enhanced 10-feature model active" : "no compatible enhanced 10-feature model active")
          .append(". Model source: ").append(appState.getActiveModelSourceDescription()).append(". ");
        sb.append("Loading data only prepares examples; it does NOT alter the learned tree rules. Click Train Decision Tree to fit/update rules. Saving or loading a .pncmodel is a separate action.");
        return sb.toString();
    }

    private String buildDatasetImpactStatus() {
        int count = appState.getEligibleReviewedRecords().size();
        String blockedReason = appState.getTrainingDisabledReason();
        boolean treeActive = appState.isCompatibleTenFeatureModelActive();
        ModelEvaluator.EvaluationReport report = appState.getLastEvaluationReport();

        if (count == 0) {
            return "DATA NOT READY — no eligible 10-feature training records are loaded. Import PKLot or review slots first. Loading an image alone does not add a labelled training sample.";
        }
        if (blockedReason != null) {
            return "DATA LOADED — training is blocked. " + simplifyTrainingMessage(blockedReason);
        }
        if (report != null && treeActive) {
            return report.metricsAvailable
                ? "TRAINING / EVALUATION RESULT AVAILABLE — current compatible tree has a report for this dataset (test N=" + report.totalEvaluated() + "). See View Results for metrics and the confusion matrix."
                : "TREE TRAINED, BUT EVALUATION IS INSUFFICIENT — " + (report.unavailableReason == null ? "the held-out test set is too small or not independent." : report.unavailableReason);
        }
        if (treeActive) {
            return "DATA LOADED — a compatible tree is active, but loading this dataset did not retrain it. Click Train Decision Tree to fit/update the tree for this dataset.";
        }
        return "DATA LOADED — records are prepared, but no compatible enhanced 10-feature tree has been trained from this dataset yet. Click Train Decision Tree when enabled.";
    }

    private String buildDatasetImpactDialogText() {
        return buildDatasetImpactStatus() + "\n\n"
            + buildDatasetImpactDetails() + "\n\n"
            + "WHAT HAPPENS NEXT\n"
            + "1. Load Dataset: reads slot labels, geometry, feature vectors and provenance. It does NOT change learned tree rules.\n"
            + "2. Train Decision Tree: CART learns feature thresholds from TRAIN records; validation selects settings; the frozen TEST partition measures performance.\n"
            + "3. Save Trained Decision Tree: stores the learned nodes and compatible feature-schema metadata in a .pncmodel file.\n"
            + "4. Load Saved Decision Tree: activates existing rules. It does not train the tree or add new samples.\n"
            + "5. Analyse an image in Feature Decision Tree mode: computes the same 10-feature vector and follows the active tree's learned branches.";
    }

    private void showDatasetImpactDialog() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Dataset → Decision Tree");
        alert.setHeaderText("What has been loaded, and what has not happened yet");
        TextArea details = new TextArea(buildDatasetImpactDialogText());
        details.setEditable(false);
        details.setWrapText(true);
        details.setPrefColumnCount(78);
        details.setPrefRowCount(18);
        alert.getDialogPane().setContent(details);
        alert.getDialogPane().setPrefWidth(760);
        alert.showAndWait();
    }

    public void refreshModelInfo() {
        if (service == null) return;

        if (service.getPredictor() != null) {
            treeTextArea.setText(service.getPredictor().getModel().toTreeString());
        } else {
            treeTextArea.setText("No model object is currently available. Load a compatible model or train a Decision Tree.");
        }
        datasetModeSelector.setValue(appState.getActiveDatasetMode());

        // Check ONNX model readiness for first-run setup notice (Requirement 1)
        VehicleDetectionService.DetectorStatus detStatus = appState.getDetectorStatus();
        boolean modelMissing = detStatus != VehicleDetectionService.DetectorStatus.READY;
        onnxSetupBox.setVisible(modelMissing);
        onnxSetupBox.setManaged(modelMissing);
        if (modelMissing) {
            lblOnnxSetupBanner.setText(
                "Vehicle detector is not installed. Install it before importing PKLot for 10-feature training."
            );
            lblOnnxSetupBanner.setTooltip(new Tooltip(
                detStatus.name() + "\nInstall a compatible ONNX model using the setup button or "
                + "scripts/setup_onnx_model.ps1. Model reference: " + OnnxVehicleDetectionService.DOCUMENTED_MODEL_URL
            ));
        }

        int eligibleTotal = appState.getEligibleReviewedRecords().size();
        int occCount = appState.getEligibleOccupiedCount();
        int availCount = appState.getEligibleAvailableCount();
        int importedGtCount = appState.getEligibleImportedGroundTruthCount();
        int humanVerCount = appState.getEligibleHumanVerifiedCount();

        lblDatasetSourceAndSchema.setText("Dataset: " + appState.getActiveDatasetMode().getDisplayName());
        lblDatasetSourceAndSchema.setTooltip(new Tooltip(
            "Source: " + appState.getDatasetSourceDescription() + "\nFeature schema: "
            + FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION + " (10 features)"));
        lblEligibleRecordCount.setText(String.valueOf(eligibleTotal));
        lblOccupiedSampleCount.setText(occCount + " Occ / " + availCount + " Avail");
        lblProvenanceBreakdown.setText("Dataset provenance: " + importedGtCount + " PKLot labels / " + humanVerCount + " user-reviewed");

        boolean legacyPresent = appState.isLegacyEightFeatureModelPresent();
        lblLegacyModelWarning.setText("An older 8-feature model is available. Train or load a 10-feature model to continue.");
        lblLegacyModelWarning.setTooltip(new Tooltip(ModelSerializer.LEGACY_EIGHT_FEATURE_MESSAGE));
        lblLegacyModelWarning.setVisible(legacyPresent);
        lblLegacyModelWarning.setManaged(legacyPresent);

        String disabledReason = appState.getTrainingDisabledReason();
        if (disabledReason != null) {
            btnTrainDecisionTree.setDisable(true);
            lblTrainingDisabledExplanation.setText(simplifyTrainingMessage(disabledReason));
            lblTrainingDisabledExplanation.setTooltip(new Tooltip(disabledReason));
        } else {
            btnTrainDecisionTree.setDisable(importInProgress);
            lblTrainingDisabledExplanation.setText(String.format(Locale.ROOT,
                "Ready: %d training samples (%d occupied, %d available).",
                eligibleTotal, occCount, availCount));
            lblTrainingDisabledExplanation.setTooltip(new Tooltip(
                "Source: " + appState.getDatasetSourceDescription() + "\nImported labels: "
                + importedGtCount + "\nUser-reviewed labels: " + humanVerCount));
        }

        lblStage4ModelStatus.setText(appState.isCompatibleTenFeatureModelActive() ? "Ready (10 features)" :
            (legacyPresent ? "Legacy (8 features)" : "Needs training"));
        ModelEvaluator.EvaluationReport lastRep = appState.getLastEvaluationReport();
        if (lastRep != null) {
            if (!lastRep.metricsAvailable) {
                lblStage4EvalStatus.setText("Latest evaluation: Insufficient test data (N=" + lastRep.totalEvaluated() + ")");
            } else {
                lblStage4EvalStatus.setText(String.format(Locale.ROOT, "Latest evaluation: %.1f%% accuracy (test N=%d)",
                    lastRep.accuracy * 100.0, lastRep.totalEvaluated()));
            }
            displayReport(lastRep);
        } else {
            lblStage4EvalStatus.setText("Latest evaluation: not available for the current dataset/model");
            lblEvalPartitionBadge.setText("Not evaluated");
            lblEvalProvenanceBanner.setText("No current evaluation is associated with the active dataset/model. Loading data does not train or evaluate a Decision Tree.");
            lblAccuracy.setText("—");
            lblPrecision.setText("—");
            lblRecall.setText("—");
            lblF1.setText("—");
            lblFalseAvailableRate.setText("—");
            lblConfusionTp.setText("Occupied correctly: —");
            lblConfusionFn.setText("Occupied predicted free: —");
            lblConfusionFp.setText("Free predicted occupied: —");
            lblConfusionTn.setText("Available correctly: —");
            lblTechnicalEvalLog.setText("No evaluation diagnostics for the current dataset/model. Train the Decision Tree and inspect results after successful evaluation.");
        }

        int activeFeatCount = service.getPredictor() != null ? service.getPredictor().getModel().getFeatureCount() : 0;
        String activeSchema = activeFeatCount > 0 ? ModelSerializer.schemaForFeatureCount(activeFeatCount) : "No model loaded";
        lblStage6ActiveModelStatus.setText(appState.isCompatibleTenFeatureModelActive()
            ? "Model status: Ready" : "Model status: No compatible model loaded");
        lblStage6ActiveModelStatus.setTooltip(new Tooltip(appState.getActiveModelStatusSummary()));
        lblStage6ActiveSchema.setText("Features: " + activeFeatCount + " | Dataset: " + appState.getActiveDatasetMode().getDisplayName());
        lblStage6ActiveSchema.setTooltip(new Tooltip("Feature schema: " + activeSchema + "\nModel source: " + appState.getActiveModelSourceDescription()));
        btnSaveModelStage6.setDisable(!appState.isCompatibleTenFeatureModelActive());

        // Keep the dataset-to-tree effect visible after import, mode changes, training, and model load.
        lblDatasetImpactStatus.setText(buildDatasetImpactStatus());
        lblDatasetImpactDetails.setText(buildDatasetImpactDetails());
    }

    public void displayReport(ModelEvaluator.EvaluationReport report) {
        if (report == null) return;

        boolean isHeldOutTest = report.scope == ModelEvaluator.EvaluationScope.HELD_OUT_TEST_PARTITION;
        boolean isInsufficient = report.scope == ModelEvaluator.EvaluationScope.INSUFFICIENT_GROUPS_NON_INDEPENDENT;
        if (isInsufficient) {
            lblEvalPartitionBadge.setText("Non-Independent (N=" + report.totalEvaluated() + ")");
            lblEvalPartitionBadge.setTooltip(new Tooltip("Evaluation refused: fewer than 3 independent groups exist (INSUFFICIENT_GROUPS / NON_INDEPENDENT)."));
        } else {
            lblEvalPartitionBadge.setText(isHeldOutTest
                ? ("Test set · N=" + report.totalEvaluated())
                : ("Evaluation · N=" + report.totalEvaluated()));
            lblEvalPartitionBadge.setTooltip(new Tooltip(isHeldOutTest
                ? "These examples were held out from training and model selection."
                : "These results are not from a separate held-out test set."));
        }

        if (!report.metricsAvailable) {
            lblAccuracy.setText("Insufficient test data");
            lblAccuracy.setTooltip(new Tooltip(report.unavailableReason));
            lblPrecision.setText("—");
            lblRecall.setText("—");
            lblF1.setText("—");
            lblFalseAvailableRate.setText("—");
            lblEvalProvenanceBanner.setText(report.unavailableReason != null && !report.unavailableReason.isBlank()
                ? report.unavailableReason : "Insufficient test data for honest metrics.");
            lblEvalProvenanceBanner.setTooltip(new Tooltip(report.unavailableReason));
        } else {
            lblAccuracy.setText(String.format(Locale.ROOT, "%.1f%%", report.accuracy * 100.0));
            lblAccuracy.setTooltip(null);
            lblPrecision.setText(report.formatMetric(report.precision));
            lblRecall.setText(report.formatMetric(report.recall));
            lblF1.setText(report.formatMetric(report.f1));
            lblFalseAvailableRate.setText(String.format(Locale.ROOT, "%.1f%% (%d/%d)",
                report.falseAvailableRate() * 100.0, report.fn, report.tp + report.fn));
            lblEvalProvenanceBanner.setText(isHeldOutTest
                ? "Results are measured on examples not used to train the model."
                : "These results are not a final independent test.");
            lblEvalProvenanceBanner.setTooltip(new Tooltip(report.provenanceNote != null ? report.provenanceNote : ""));
        }

        lblConfusionTp.setText("Occupied correctly: " + report.tp);
        lblConfusionFn.setText("Occupied predicted free: " + report.fn);
        lblConfusionFp.setText("Free predicted occupied: " + report.fp);
        lblConfusionTn.setText("Available correctly: " + report.tn);

        ParkingAnalysisService.PkLotTrainingSummary pkSummary = service.getLastPkLotTrainingSummary();
        DatasetLoader.GroupedPartitions parts = service.getLastGroupedPartitions();
        String partInfo = pkSummary != null
            ? pkSummary.selectionRationale()
            : (parts != null
                ? (parts.hasSufficientIndependentGroups()
                    ? String.format(Locale.ROOT, "Train Groups=%s (N=%d) | Val Groups=%s (N=%d) | Held-Out Test Groups=%s (N=%d)",
                        parts.trainGroups(), parts.train().size(),
                        parts.validationGroups(), parts.validation().size(),
                        parts.testGroups(), parts.test().size())
                    : String.format(Locale.ROOT, "Non-Independent Evaluation (%s): %d groups in training. Held-out test refused.",
                        DatasetLoader.GroupedPartitions.INSUFFICIENT_GROUPS_LABEL,
                        parts.trainGroups().size() + parts.validationGroups().size() + parts.testGroups().size()))
                : "Partitions: N/A");
        lblTechnicalEvalLog.setText(partInfo + "\n" + report);
    }

    public HBox getOnnxSetupBox()                    { return onnxSetupBox; }
    public Label getLblOnnxSetupBanner()             { return lblOnnxSetupBanner; }
    public Button getBtnSetupOnnxModel()             { return btnSetupOnnxModel; }
    public Label getLblLegacyModelWarning()          { return lblLegacyModelWarning; }
    public Label getLblTrainingDisabledExplanation() { return lblTrainingDisabledExplanation; }
    public Label getLblPkLotSummaryBanner()          { return lblPkLotSummaryBanner; }
    public Label getLblProvenanceBreakdown()         { return lblProvenanceBreakdown; }
    public Label getLblEvalProvenanceBanner()        { return lblEvalProvenanceBanner; }
    public Label getLblFalseAvailablePlainEnglish()  { return lblFalseAvailablePlainEnglish; }
    public Label getLblEvalPartitionBadge()          { return lblEvalPartitionBadge; }
    public Button getBtnTrainDecisionTree()          { return btnTrainDecisionTree; }
    public Button getBtnImportPkLot()                { return btnImportPkLot; }
    public Label getLblDatasetImpactStatus()          { return lblDatasetImpactStatus; }
    public Label getLblDatasetImpactDetails()         { return lblDatasetImpactDetails; }
    public Button getBtnExplainDatasetFlow()          { return btnExplainDatasetFlow; }
    public Button getBtnCancelImport()               { return btnCancelImport; }
    public Button getBtnSampleAnnotations()          { return btnSampleAnnotations; }
    public TitledPane getAdvModelSettingsPane()      { return advModelSettingsPane; }
    public TitledPane getAdvEvaluationDiagnosticsPane() { return advEvaluationDiagnosticsPane; }
}
