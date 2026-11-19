package com.parknchillai.ui;

import com.parknchillai.ai.DatasetManifest;
import com.parknchillai.ai.PkLotDatasetImporter;
import com.parknchillai.ai.DecisionTreeModel;
import com.parknchillai.app.AnalysisSummary;
import com.parknchillai.app.AppState;
import com.parknchillai.app.ImageSessionRecord;
import com.parknchillai.app.ParkingAnalysisService;
import com.parknchillai.app.SlotAnalysisResult;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.detection.OccupancyDecisionEngine;
import com.parknchillai.detection.VehicleDetectionService;
import com.parknchillai.imageprocessing.ImageEnhancer;
import com.parknchillai.imageprocessing.PerspectiveTransformer;
import com.parknchillai.layout.LayoutDetectionResult;
import com.parknchillai.layout.SlotGenerator;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.ParkingSlotManager;
import com.parknchillai.parking.SlotStatus;
import com.parknchillai.utils.ImageLoader;
import com.parknchillai.utils.SampleDataGenerator;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * User-focused ParkNChill window. Four plain-language steps group the six internal workflow
 * states so the application remains easy to navigate without duplicating analysis logic.
 */
public class MainWindow extends BorderPane {

    public static final String APP_TITLE = "ParkNChill";
    public static final String APP_SUBTITLE = "Smart Parking Occupancy Detection";

    private final ParkingAnalysisService analysisService;
    private final AppState appState;
    private final ParkingSlotManager slotManager = new ParkingSlotManager();

    // Core UI Views
    private final ParkingView parkingView = new ParkingView();
    private final ResultSummaryView summaryView = new ResultSummaryView();
    private final PipelineInspectorView inspectorView = new PipelineInspectorView();
    private final SlotTableView slotTableView = new SlotTableView();
    private final ModelDetailsView modelDetailsView;

    // Compact Application Header Elements (Requirement 2.A)
    private final Label lblAppTitle = new Label(APP_TITLE);
    private final Label lblAppSubtitle = new Label(APP_SUBTITLE);
    private final Label lblWorkflowStatusBadge = new Label("Step 1 of 4: Load & Analyze");
    private final Label lblDetectorStatusBadge = new Label("Detector: MODEL_MISSING");
    private final HBox reviewerBox = new HBox(6);
    private final TextField txtReviewerId = new TextField();
    private final Label lblReviewerValidation = new Label("Reviewer ID Ready");

    // Four-step user-facing navigation; six internal states remain available to the code.
    private final HBox stepperBar = new HBox(8);
    // Four user-facing steps map onto the six internal states. Internal states remain
    // separate so the analysis and evaluation logic is not duplicated or lost.
    private final Map<AppState.WorkflowStage, Button> stageButtons = new EnumMap<>(AppState.WorkflowStage.class);
    private static final List<AppState.WorkflowStage> USER_STEPS = List.of(
        AppState.WorkflowStage.LOAD_IMAGE,
        AppState.WorkflowStage.REVIEW_OCCUPANCY,
        AppState.WorkflowStage.TRAIN_DECISION_TREE,
        AppState.WorkflowStage.SAVE_AND_USE_MODEL
    );
    private final Label lblPrerequisiteBanner = new Label("");
    private final Button btnPreviousStage = new Button("Back");
    private final Button btnNextStage = new Button("Continue");
    private final Button btnLoadDemoImage = new Button("Load Demo Image");
    private final Button btnExportAnnotatedImage = new Button("Export Annotated PNG");

    // Stage-Specific Main Working Area Container
    private final StackPane stageWorkspaceContainer = new StackPane();
    private final SplitPane imageAndReviewSplitPane = new SplitPane();
    private final VBox stage1LoadImagePane = new VBox(10);
    private final VBox stage2DetectPane = new VBox(10);

    // Bottom Status Bar
    private final Label lblStatus = new Label("Ready. Open a parking lot image or click 'Load Demo Image' to begin.");
    private final ProgressBar taskProgress = new ProgressBar(0);

    public MainWindow(ParkingAnalysisService analysisService) {
        this(analysisService, true);
    }

    public MainWindow(ParkingAnalysisService analysisService, boolean loadDefaultDemoOnStartup) {
        this.analysisService = analysisService;
        this.appState = new AppState(analysisService);
        this.appState.setOnSessionClosedCallback(() -> this.analysisService.clearDetectionCache());
        this.modelDetailsView = new ModelDetailsView(analysisService, appState);
        this.slotTableView.setAppState(appState);
        this.slotTableView.setEmbeddedInspector(inspectorView);
        this.inspectorView.setModelForDomainInspection(analysisService.getPredictor().getModel());

        setTop(createCompactHeaderAndWorkflowStepper());
        setCenter(createMainStageWorkspace());
        setBottom(createStatusBar());

        setupViewWiringAndCallbacks();

        if (loadDefaultDemoOnStartup) {
            loadInitialDefaultState();
        } else {
            refreshAllViewsFromAppState();
        }
    }

    // -------------------------------------------------------------------------
    // 1. Compact Header & Four-Step Navigation
    // -------------------------------------------------------------------------

    private VBox createCompactHeaderAndWorkflowStepper() {
        VBox topBox = new VBox(0);

        // A. Compact Application Header
        HBox header = new HBox(12);
        header.setAlignment(Pos.CENTER_LEFT);
        header.getStyleClass().add("main-header");

        Label logo = new Label("P&C");
        logo.getStyleClass().add("app-logo-badge");

        VBox titleBox = new VBox(1);
        lblAppTitle.getStyleClass().add("app-title");
        lblAppSubtitle.getStyleClass().add("app-subtitle");
        titleBox.getChildren().addAll(lblAppTitle, lblAppSubtitle);

        HBox spacer = new HBox();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        reviewerBox.setAlignment(Pos.CENTER_LEFT);
        Label lblReviewer = new Label("Reviewer ID:");
        lblReviewer.setStyle("-fx-text-fill: #f8fafc; -fx-font-weight: bold; -fx-font-size: 12px;");
        txtReviewerId.setPromptText("Your reviewer ID");
        txtReviewerId.setPrefWidth(150);
        txtReviewerId.setMinWidth(140);
        txtReviewerId.setStyle("-fx-text-fill: #f8fafc; -fx-control-inner-background: #0f172a; -fx-font-size: 12px;");
        txtReviewerId.textProperty().addListener((obs, oldVal, newVal) -> {
            appState.setReviewerId(newVal);
            inspectorView.setReviewerId(newVal);
            updateReviewerValidationUI();
        });

        lblReviewerValidation.setStyle("-fx-font-size: 11px; -fx-font-weight: bold; -fx-text-fill: #34d399;");
        reviewerBox.getChildren().addAll(lblReviewer, txtReviewerId, lblReviewerValidation);

        lblWorkflowStatusBadge.setStyle(
            "-fx-background-color: #1e3a8a; -fx-text-fill: #f8fafc; -fx-font-weight: bold; " +
            "-fx-font-size: 11.5px; -fx-padding: 4px 9px; -fx-background-radius: 5px;"
        );

        updateDetectorStatusBadge(appState.getDetectorStatus(), null);

        header.getChildren().addAll(
            logo, titleBox, spacer, reviewerBox, lblWorkflowStatusBadge, lblDetectorStatusBadge
        );

        // B. Four simple user-facing workflow steps. Detection is part of Image Analysis;
        // training and evaluation are part of one model step. Internal states remain unchanged.
        stepperBar.setAlignment(Pos.CENTER_LEFT);
        stepperBar.setPadding(new Insets(7, 12, 7, 12));
        stepperBar.setStyle("-fx-background-color: #0b1120; -fx-border-color: #334155; -fx-border-width: 0 0 1px 0;");

        for (AppState.WorkflowStage stage : USER_STEPS) {
            Button stepBtn = new Button(simpleStepTitle(stage));
            stepBtn.setMinWidth(150);
            stepBtn.setPrefWidth(210);
            stepBtn.setMaxWidth(230);
            stepBtn.setMaxHeight(34);
            stepBtn.setOnAction(e -> navigateToUserStep(stage));
            stageButtons.put(stage, stepBtn);
            stepperBar.getChildren().add(stepBtn);
        }

        HBox navSpacer = new HBox();
        HBox.setHgrow(navSpacer, Priority.ALWAYS);

        btnPreviousStage.getStyleClass().add("btn-tool");
        btnPreviousStage.setMinWidth(72);
        btnPreviousStage.setPrefWidth(84);
        btnPreviousStage.setOnAction(e -> navigateUserStep(-1));

        btnNextStage.getStyleClass().add("btn-primary");
        btnNextStage.setMinWidth(98);
        btnNextStage.setPrefWidth(118);
        btnNextStage.setOnAction(e -> navigateUserStep(1));

        stepperBar.getChildren().addAll(navSpacer, btnPreviousStage, btnNextStage);

        // C. Concise prerequisite / guidance bar
        HBox prereqBar = new HBox(10);
        prereqBar.setAlignment(Pos.CENTER_LEFT);
        prereqBar.setPadding(new Insets(5, 14, 5, 14));
        prereqBar.setStyle("-fx-background-color: #162032; -fx-border-color: #334155; -fx-border-width: 0 0 1px 0;");

        lblPrerequisiteBanner.setWrapText(true);
        lblPrerequisiteBanner.setStyle("-fx-text-fill: #e2e8f0; -fx-font-size: 12px; -fx-font-weight: 600;");
        HBox.setHgrow(lblPrerequisiteBanner, Priority.ALWAYS);

        btnLoadDemoImage.getStyleClass().add("btn-secondary");
        btnLoadDemoImage.setOnAction(e -> handleLoadDemo());
        btnExportAnnotatedImage.getStyleClass().add("btn-tool");
        btnExportAnnotatedImage.setDisable(true);
        btnExportAnnotatedImage.setOnAction(e -> handleExportAnnotatedImage());

        // Avoid crowding the guidance line. Optional demo/export actions appear only when relevant.
        prereqBar.getChildren().addAll(lblPrerequisiteBanner, btnLoadDemoImage, btnExportAnnotatedImage);

        topBox.getChildren().addAll(header, stepperBar, prereqBar);
        return topBox;
    }

    /** Computes the total preferred width of four user-facing steps and navigation controls. */
    public double getTotalStepperBarPreferredWidth() {
        double sum = 24.0; // left + right padding (12 + 12)
        int count = 0;
        for (Button b : stageButtons.values()) {
            sum += b.getPrefWidth();
            count++;
        }
        sum += btnPreviousStage.getPrefWidth() + btnNextStage.getPrefWidth();
        count += 3; // Previous, Next, and the expanding spacer
        sum += Math.max(0, count - 1) * stepperBar.getSpacing();
        return sum;
    }

    public double getTotalStepperBarMinimumWidth() {
        double sum = 24.0;
        int count = 0;
        for (Button b : stageButtons.values()) {
            sum += b.getMinWidth();
            count++;
        }
        sum += btnPreviousStage.getMinWidth() + btnNextStage.getMinWidth();
        count += 3; // Previous, Next, and the expanding spacer
        sum += Math.max(0, count - 1) * stepperBar.getSpacing();
        return sum;
    }

    private StackPane createMainStageWorkspace() {
        stageWorkspaceContainer.setPadding(new Insets(8));

        imageAndReviewSplitPane.setOrientation(Orientation.HORIZONTAL);
        imageAndReviewSplitPane.setDividerPositions(0.47);

        stageWorkspaceContainer.getChildren().add(stage1LoadImagePane);
        return stageWorkspaceContainer;
    }

    private HBox createStatusBar() {
        HBox bar = new HBox(14);
        bar.setAlignment(Pos.CENTER_LEFT);
        bar.getStyleClass().add("status-bar");

        lblStatus.getStyleClass().add("status-bar-text");
        HBox.setHgrow(lblStatus, Priority.ALWAYS);

        taskProgress.setPrefWidth(130);
        taskProgress.setVisible(false);

        bar.getChildren().addAll(lblStatus, taskProgress);
        return bar;
    }

    // -------------------------------------------------------------------------
    // 2. Stage Navigation & Active Workspace Switching (Requirement 1 & 6)
    // -------------------------------------------------------------------------

    public void navigateToStage(AppState.WorkflowStage targetStage) {
        if (targetStage == null) return;
        appState.setCurrentStage(targetStage);
        refreshAllViewsFromAppState();
    }

    private static int userStepNumber(AppState.WorkflowStage stage) {
        return switch (stage) {
            case LOAD_IMAGE, DETECT_VEHICLES_AND_SLOTS -> 1;
            case REVIEW_OCCUPANCY -> 2;
            case TRAIN_DECISION_TREE, EVALUATE_MODEL -> 3;
            case SAVE_AND_USE_MODEL -> 4;
        };
    }

    private static String userStepName(AppState.WorkflowStage stage) {
        return switch (userStepNumber(stage)) {
            case 1 -> "Load & Analyze";
            case 2 -> "Review Slots";
            case 3 -> "Train & Check";
            default -> "Save & Use";
        };
    }

    private static String simpleStepTitle(AppState.WorkflowStage representative) {
        return userStepNumber(representative) + ". " + userStepName(representative);
    }

    private AppState.StageVisualState userStepVisualState(AppState.WorkflowStage representative) {
        if (userStepNumber(appState.getCurrentStage()) == userStepNumber(representative)) {
            return AppState.StageVisualState.ACTIVE;
        }
        if (representative == AppState.WorkflowStage.LOAD_IMAGE) {
            return appState.getCurrentSummary() != null
                ? AppState.StageVisualState.COMPLETED : AppState.StageVisualState.PENDING;
        }
        if (representative == AppState.WorkflowStage.TRAIN_DECISION_TREE) {
            if (appState.getLastEvaluationReport() != null && appState.isStageCompleted(AppState.WorkflowStage.TRAIN_DECISION_TREE)) {
                return AppState.StageVisualState.COMPLETED;
            }
            return appState.isStageBlocked(AppState.WorkflowStage.TRAIN_DECISION_TREE)
                ? AppState.StageVisualState.BLOCKED : AppState.StageVisualState.PENDING;
        }
        return appState.getStageVisualState(representative);
    }

    private void navigateToUserStep(AppState.WorkflowStage representative) {
        switch (representative) {
            case LOAD_IMAGE -> navigateToStage(AppState.WorkflowStage.LOAD_IMAGE);
            case REVIEW_OCCUPANCY -> {
                if (appState.getCurrentSummary() == null) {
                    lblStatus.setText("First load an image and select Detect and Analyse.");
                    if (appState.getCurrentImage() != null) navigateToStage(AppState.WorkflowStage.DETECT_VEHICLES_AND_SLOTS);
                    return;
                }
                navigateToStage(AppState.WorkflowStage.REVIEW_OCCUPANCY);
            }
            case TRAIN_DECISION_TREE -> navigateToStage(AppState.WorkflowStage.TRAIN_DECISION_TREE);
            case SAVE_AND_USE_MODEL -> navigateToStage(AppState.WorkflowStage.SAVE_AND_USE_MODEL);
            default -> navigateToStage(representative);
        }
    }

    private void navigateUserStep(int direction) {
        AppState.WorkflowStage current = appState.getCurrentStage();
        int step = userStepNumber(current);
        if (direction > 0) {
            if (step == 1) {
                if (appState.getCurrentImage() == null) {
                    lblStatus.setText("Open a parking photo first, or use Load Demo Image.");
                } else if (appState.getCurrentSummary() == null) {
                    runDetectionAndAnalysis();
                } else {
                    navigateToStage(AppState.WorkflowStage.REVIEW_OCCUPANCY);
                }
            } else if (step == 2) {
                navigateToStage(AppState.WorkflowStage.TRAIN_DECISION_TREE);
            } else if (step == 3) {
                if (current == AppState.WorkflowStage.TRAIN_DECISION_TREE) {
                    if (appState.getLastEvaluationReport() == null) {
                        lblStatus.setText("Train the Decision Tree first to generate evaluation results.");
                    } else {
                        navigateToStage(AppState.WorkflowStage.EVALUATE_MODEL);
                    }
                } else {
                    navigateToStage(AppState.WorkflowStage.SAVE_AND_USE_MODEL);
                }
            } else {
                navigateToStage(AppState.WorkflowStage.LOAD_IMAGE);
            }
        } else {
            if (step == 1) {
                navigateToStage(AppState.WorkflowStage.LOAD_IMAGE);
            } else if (step == 2) {
                navigateToStage(AppState.WorkflowStage.LOAD_IMAGE);
            } else if (step == 3) {
                navigateToStage(current == AppState.WorkflowStage.EVALUATE_MODEL
                    ? AppState.WorkflowStage.TRAIN_DECISION_TREE : AppState.WorkflowStage.REVIEW_OCCUPANCY);
            } else {
                navigateToStage(AppState.WorkflowStage.EVALUATE_MODEL);
            }
        }
    }

    private String simpleGuidance(AppState.WorkflowStage stage) {
        return switch (stage) {
            case LOAD_IMAGE -> appState.getCurrentImage() == null
                ? "Start here: open a parking photo, or use the demo image to learn the controls."
                : "Image loaded. Select Continue / Detect Slots to analyse parking spaces.";
            case DETECT_VEHICLES_AND_SLOTS -> appState.getCurrentSummary() == null
                ? "Run Detect and Analyse. If the detector is missing, see the Detector status at the top."
                : "Analysis complete. Continue to Review Slots and confirm uncertain labels.";
            case REVIEW_OCCUPANCY -> appState.getCurrentSummary() == null
                ? "Run image analysis first, then review each parking space."
                : "Select each slot. Confirm Occupied or Available; leave unclear cases as Needs Review.";
            case TRAIN_DECISION_TREE -> appState.canTrainDecisionTree()
                ? "Your reviewed dataset is ready. Train the Decision Tree, then check its test results."
                : "To train, import PKLot or review enough slots to include both Occupied and Available examples.";
            case EVALUATE_MODEL -> appState.getLastEvaluationReport() == null
                ? "Train or load a compatible Decision Tree to see evaluation results."
                : "Review the held-out test results. False-Available means an occupied space was predicted as free.";
            case SAVE_AND_USE_MODEL -> appState.isCompatibleTenFeatureModelActive()
                ? "A compatible model is active. Save it to reuse it later."
                : "Train or load a compatible 10-feature Decision Tree before saving it for reuse.";
        };
    }

    private void rebuildActiveStageLayout(AppState.WorkflowStage stage) {
        stageWorkspaceContainer.getChildren().clear();
        parkingView.configurePrimaryActionForStage(stage);

        switch (stage) {
            case LOAD_IMAGE -> {
                stage1LoadImagePane.getChildren().clear();
                stage1LoadImagePane.getChildren().add(parkingView);
                VBox.setVgrow(parkingView, Priority.ALWAYS);
                stageWorkspaceContainer.getChildren().add(stage1LoadImagePane);
            }
            case DETECT_VEHICLES_AND_SLOTS -> {
                stage2DetectPane.getChildren().clear();
                stage2DetectPane.getChildren().addAll(summaryView, parkingView);
                VBox.setVgrow(parkingView, Priority.ALWAYS);
                stageWorkspaceContainer.getChildren().add(stage2DetectPane);
            }
            case REVIEW_OCCUPANCY -> {
                imageAndReviewSplitPane.getItems().clear();
                VBox leftImageBox = new VBox(6, parkingView);
                VBox.setVgrow(parkingView, Priority.ALWAYS);

                VBox rightReviewBox = new VBox(8, summaryView, slotTableView);
                VBox.setVgrow(slotTableView, Priority.ALWAYS);

                imageAndReviewSplitPane.getItems().addAll(leftImageBox, rightReviewBox);
                imageAndReviewSplitPane.setDividerPositions(0.46);
                stageWorkspaceContainer.getChildren().add(imageAndReviewSplitPane);
            }
            case TRAIN_DECISION_TREE, EVALUATE_MODEL, SAVE_AND_USE_MODEL -> {
                modelDetailsView.showForStage(stage);
                ScrollPane scroll = new ScrollPane(modelDetailsView);
                scroll.setFitToWidth(true);
                scroll.getStyleClass().add("clean-scroll-pane");
                stageWorkspaceContainer.getChildren().add(scroll);
            }
        }
    }

    public void refreshAllViewsFromAppState() {
        AppState.WorkflowStage activeStage = appState.getCurrentStage();
        AnalysisSummary visibleSummary = appState.getCurrentSummary();
        boolean treeCanRunSeparately = appState.isCompatibleTenFeatureModelActive()
            && visibleSummary != null && !visibleSummary.getSlotResults().isEmpty();
        parkingView.getBtnRunLoadedTree().setDisable(!treeCanRunSeparately);
        parkingView.getBtnRunLoadedTree().setTooltip(new Tooltip(treeCanRunSeparately
            ? "Run the active saved 10-feature tree against the current slot features. Shows a separate prediction column; does not replace the last analysis result, detector suggestions, summary, or overlays."
            : "Load a compatible 10-feature tree and analyse an image first. Then run its predictions separately without replacing the detector results."));
        String enhancementBadge = appState.getCurrentImage() != null
            ? (appState.isWorkingImageEnhanced() ? " · [ENHANCED WORKING IMAGE]" : " · [ORIGINAL IMAGE]")
            : "";
        lblWorkflowStatusBadge.setText(String.format(Locale.ROOT,
            "Step %d of 4: %s%s", userStepNumber(activeStage), userStepName(activeStage), enhancementBadge));

        boolean reviewerNeeded = activeStage == AppState.WorkflowStage.REVIEW_OCCUPANCY;
        reviewerBox.setVisible(reviewerNeeded);
        reviewerBox.setManaged(reviewerNeeded);
        updateReviewerValidationUI();

        for (Map.Entry<AppState.WorkflowStage, Button> entry : stageButtons.entrySet()) {
            AppState.WorkflowStage representative = entry.getKey();
            Button btn = entry.getValue();
            AppState.StageVisualState vis = userStepVisualState(representative);
            String prefix = switch (vis) {
                case ACTIVE -> "● ";
                case COMPLETED -> "✓ ";
                case BLOCKED -> "! ";
                case PENDING -> "";
            };
            btn.setText(prefix + simpleStepTitle(representative));
            btn.setStyle(styleForStepperButton(vis));
            btn.setTooltip(new Tooltip(simpleStepTitle(representative) + " — " + vis.getBadgeText()
                + "\n" + simpleGuidance(representative)));
            btn.setAccessibleText(simpleStepTitle(representative) + ", " + vis.getBadgeText());
        }

        int currentUiStep = userStepNumber(activeStage);
        btnPreviousStage.setText(currentUiStep == 4 ? "Back to Results" : "Back");
        btnPreviousStage.setDisable(currentUiStep == 1);
        if (currentUiStep == 1) {
            btnNextStage.setText(appState.getCurrentImage() == null ? "Continue" :
                (appState.getCurrentSummary() == null ? "Detect Slots" : "Review Slots →"));
            btnNextStage.setDisable(appState.getCurrentImage() == null);
        } else if (currentUiStep == 2) {
            btnNextStage.setText("Train Model →");
            btnNextStage.setDisable(false);
        } else if (currentUiStep == 3 && activeStage == AppState.WorkflowStage.TRAIN_DECISION_TREE) {
            btnNextStage.setText("View Results →");
            btnNextStage.setDisable(appState.getLastEvaluationReport() == null);
        } else if (currentUiStep == 3) {
            btnNextStage.setText("Save Model →");
            btnNextStage.setDisable(!appState.isCompatibleTenFeatureModelActive());
        } else {
            btnNextStage.setText("Load Another Image");
            btnNextStage.setDisable(false);
        }

        boolean showDemoBtn = currentUiStep == 1 && appState.getCurrentImage() == null;
        btnLoadDemoImage.setVisible(showDemoBtn);
        btnLoadDemoImage.setManaged(showDemoBtn);
        boolean showExportBtn = currentUiStep == 1 && appState.getCurrentSummary() != null;
        btnExportAnnotatedImage.setVisible(showExportBtn);
        btnExportAnnotatedImage.setManaged(showExportBtn);

        lblPrerequisiteBanner.setText(simpleGuidance(activeStage));
        updateDetectorStatusBadge(appState.getDetectorStatus(), null);
        summaryView.update(appState.getCurrentSummary(), appState);
        modelDetailsView.refreshModelInfo();

        rebuildActiveStageLayout(activeStage);
    }

    private void updateReviewerValidationUI() {
        if (appState.isReviewerIdValid()) {
            lblReviewerValidation.setText("Ready (" + appState.getReviewerId() + ")");
            lblReviewerValidation.setStyle("-fx-font-size: 11px; -fx-font-weight: bold; -fx-text-fill: #34d399;");
        } else {
            lblReviewerValidation.setText("Required (min 2 chars)");
            lblReviewerValidation.setStyle("-fx-font-size: 11px; -fx-font-weight: bold; -fx-text-fill: #fbbf24;");
        }
    }

    private static String styleForStepperButton(AppState.StageVisualState state) {
        return switch (state) {
            case ACTIVE ->
                "-fx-background-color: #2563eb; -fx-text-fill: #ffffff; -fx-font-weight: bold; " +
                "-fx-font-size: 11px; -fx-padding: 5px 7px; -fx-background-radius: 5px; -fx-border-color: #93c5fd; -fx-border-radius: 5px; -fx-border-width: 1.5px;";
            case COMPLETED ->
                "-fx-background-color: #064e3b; -fx-text-fill: #a7f3d0; -fx-font-weight: bold; " +
                "-fx-font-size: 11px; -fx-padding: 5px 7px; -fx-background-radius: 5px; -fx-border-color: #10b981; -fx-border-radius: 5px; -fx-border-width: 1px;";
            case BLOCKED ->
                "-fx-background-color: #3f2e1e; -fx-text-fill: #fde68a; -fx-font-weight: 600; " +
                "-fx-font-size: 11px; -fx-padding: 5px 7px; -fx-background-radius: 5px; -fx-border-color: #f59e0b; -fx-border-radius: 5px; -fx-border-width: 1px;";
            case PENDING ->
                "-fx-background-color: #1e293b; -fx-text-fill: #cbd5e1; -fx-font-weight: 600; " +
                "-fx-font-size: 11px; -fx-padding: 5px 7px; -fx-background-radius: 5px; -fx-border-color: #475569; -fx-border-radius: 5px; -fx-border-width: 1px;";
        };
    }

    // -------------------------------------------------------------------------
    // 3. Synchronized Selection, Slot Review & Boundary Editing Callbacks
    // -------------------------------------------------------------------------

    private void setupViewWiringAndCallbacks() {
        parkingView.setOnOpenImageAction(this::handleOpenImage);
        parkingView.setOnAddImageAction(this::handleOpenAdditionalImage);
        parkingView.setOnCloseSessionAction(this::handleCloseActiveSession);
        parkingView.setOnFreshDatasetAction(this::handleStartFreshDataset);
        parkingView.setOnSessionSelectedAction(this::handleSwitchImageSession);
        parkingView.setOnLoadSlotXmlAction(this::handleLoadSlotAnnotations);
        parkingView.setOnDetectAndAnalyseAction(this::runDetectionAndAnalysis);
        parkingView.setOnRunLoadedTreeAction(this::runLoadedTreeSeparatelyOnCurrentImage);
        parkingView.setOnViewIvpStagesAction(() -> IvpStagesDialog.show(
            getScene() != null ? getScene().getWindow() : null,
            appState.getOriginalImage(),
            appState.getActiveWorkingImage(),
            parkingView.getSelectedSlot(),
            analysisService.isUseAdaptiveThreshold()));
        parkingView.setOnEnhanceImageAction(() -> {
            ImageEnhancementDialog.show(
                getScene() != null ? getScene().getWindow() : null,
                appState,
                parkingView.getSelectedSlot());
            if (appState.getCurrentImage() != null) {
                parkingView.setImageAndSlots(appState.getCurrentImage(), appState.getCurrentSlots());
                if (appState.getCurrentSummary() != null) {
                    applyAnalysisSummary(appState.getCurrentSummary());
                }
            }
        });
        parkingView.setOnAutoDetectSlotsAction(this::handleAutoDetectLayout);
        parkingView.setOnLoadSlotsCsvAction(this::handleLoadSlotsCsv);
        parkingView.setOnSaveSlotsCsvAction(this::handleSaveSlotsCsv);
        parkingView.setOnClearSlotsAction(this::handleClearSlots);

        parkingView.setOnAdaptiveThresholdChanged(adaptive -> {
            analysisService.setUseAdaptiveThreshold(adaptive);
            if (appState.getCurrentImage() != null && appState.getCurrentSummary() != null) {
                runDetectionAndAnalysis();
            }
        });

        parkingView.setOnAnalysisModeChanged(mode -> {
            analysisService.setAnalysisMode(mode);
            if (appState.getCurrentImage() != null && appState.getCurrentSummary() != null) {
                runDetectionAndAnalysis();
            }
        });

        parkingView.setOnDetectionStrategyChanged(strategy -> {
            analysisService.setActiveDetectionStrategy(strategy);
            lblStatus.setText("Vehicle detection strategy set to: " + strategy.getDisplayName());
            if (appState.getCurrentImage() != null && appState.getCurrentSummary() != null) {
                runDetectionAndAnalysis();
            }
        });

        parkingView.setOnSlotSelected(slot -> {
            slotTableView.selectSlot(slot);
            updateInspectorForSlot(slot);
        });

        slotTableView.setOnSlotSelected(slot -> {
            parkingView.setSelectedSlot(slot);
            updateInspectorForSlot(slot);
        });

        inspectorView.setOnSlotNavigated(slot -> {
            parkingView.setSelectedSlot(slot);
            slotTableView.selectSlot(slot);
        });

        parkingView.setOnOpacityChanged(analysisService::setOverlayFillOpacity);

        slotTableView.setOnRequestBoundaryEdit(() -> {
            parkingView.setAdvancedOptionsExpanded(true);
            lblStatus.setText("Boundary editing tools expanded below the parking image. Remember to re-confirm slot labels after any geometry change.");
        });

        slotTableView.setOnSlotReviewed((slotResult, reviewedStatus) -> {
            try {
                appState.reviewSlot(slotResult.getId(), reviewedStatus);
                parkingView.renderOverlays();
                summaryView.update(appState.getCurrentSummary(), appState);
                refreshAllViewsFromAppState();
                lblStatus.setText(String.format(Locale.ROOT,
                    "Slot #%d marked as %s (%s). Eligible 10-feature training records: %d.",
                    slotResult.getId(), reviewedStatus,
                    appState.getReviewStatusForSlot(slotResult.getId()),
                    appState.getEligibleReviewedRecords().size()
                ));
            } catch (Exception ex) {
                showError("Review Action Blocked", ex.getMessage());
            }
        });

        inspectorView.setOnSampleLabeled((slotResult, groundTruth) -> {
            try {
                appState.reviewSlot(slotResult.getId(), groundTruth);
                parkingView.renderOverlays();
                summaryView.update(appState.getCurrentSummary(), appState);
                refreshAllViewsFromAppState();
            } catch (Exception ex) {
                showError("Review Action Blocked", ex.getMessage());
            }
        });

        parkingView.setOnBoxDrawn(box -> {
            try {
                int startId = slotManager.nextAvailableId();
                if (box.bayCount() <= 1) {
                    slotManager.addSlot(new ParkingSlot(
                        startId, box.x(), box.y(), box.width(), box.height(),
                        ParkingSlot.GeometryOrigin.MANUAL_DRAWN));
                    appState.setCurrentSlots(slotManager.getSlots());
                    runDetectionAndAnalysis();
                    appState.markSlotGeometryModified(startId);
                    lblStatus.setText(String.format(Locale.ROOT,
                        "Added Slot #%d (%d×%d px). Explicit label confirmation is required in Step 3 before this slot can enter training.",
                        startId, box.width(), box.height()));
                } else {
                    List<ParkingSlot> rowSlots = SlotGenerator.generateRowGrid(
                        startId, box.x(), box.y(), box.width(), box.height(), box.bayCount(), 4);
                    for (ParkingSlot s : rowSlots) {
                        slotManager.addSlot(s);
                    }
                    appState.setCurrentSlots(slotManager.getSlots());
                    runDetectionAndAnalysis();
                    for (ParkingSlot s : rowSlots) {
                        appState.markSlotGeometryModified(s.getId());
                    }
                    lblStatus.setText(String.format(Locale.ROOT,
                        "Generated %d-bay Row Grid (Slots #%d..#%d). Confirm labels in Step 3 before training.",
                        rowSlots.size(), startId, startId + rowSlots.size() - 1));
                }
                refreshAllViewsFromAppState();
            } catch (Exception ex) {
                showError("Could Not Add Slot(s)", ex.getMessage());
            }
        });

        parkingView.setOnPerspectiveQuadSelected(quad -> {
            if (appState.getCurrentImage() == null) return;
            try {
                BufferedImage warped = PerspectiveTransformer.warpPerspective(
                    appState.getCurrentImage(), quad,
                    appState.getCurrentImage().getWidth(), appState.getCurrentImage().getHeight()
                );
                ImageLoader.LoadedImageMetadata meta = appState.getCurrentImageMetadata() != null
                    ? appState.getCurrentImageMetadata().withPerspectiveWarp(warped)
                    : null;
                LayoutDetectionResult detected = analysisService.autoDetectLayout(ImageEnhancer.autoEnhance(warped));
                slotManager.setSlots(detected.getSlots());
                appState.setLoadedImageAndSlots(
                    meta, appState.getCurrentImageSha256(),
                    slotManager.getSlots(), detected.isLocalizationReliable(), detected.getLocalizationSummary()
                );
                parkingView.setImageAndSlots(warped, slotManager.getSlots());
                runDetectionAndAnalysis();
            } catch (Exception ex) {
                showError("Perspective Rectification Failed", ex.getMessage());
            }
        });

        parkingView.setOnSlotDeleted(slotId -> {
            if (slotManager.removeSlotById(slotId)) {
                appState.setCurrentSlots(slotManager.getSlots());
                runDetectionAndAnalysis();
                lblStatus.setText("Removed Slot #" + slotId + ". Remaining slots: " + slotManager.getTotalCount());
            }
        });

        modelDetailsView.setOnActiveModelChanged(() -> {
            // Model load/training only changes the active model. It must not change the selected
            // occupancy mode or rerun image analysis, because that would replace the visible
            // detector result with a tree-only result without the user's explicit request.
            if (analysisService.getPredictor() != null && analysisService.getPredictor().getModel() != null) {
                inspectorView.setModelForDomainInspection(analysisService.getPredictor().getModel());
            }
            slotTableView.clearSeparatelyAppliedTreePredictions();
            refreshAllViewsFromAppState();

            if (appState.isCompatibleTenFeatureModelActive()) {
                lblStatus.setText("Compatible 10-feature Decision Tree loaded/trained. Existing detector status and overlays were NOT changed. "
                    + "Go to Review Slots and click 'Run Tree Separately' to see a separate prediction column.");
            } else {
                lblStatus.setText("A legacy or incompatible tree is active for inspection only. It will not be applied to enhanced 10-feature slots.");
            }
        });

        modelDetailsView.setOnReturnToImageAnalysis(() -> {
            if (appState.getCurrentSummary() != null) {
                navigateToStage(AppState.WorkflowStage.REVIEW_OCCUPANCY);
            } else if (appState.getCurrentImage() != null) {
                navigateToStage(AppState.WorkflowStage.DETECT_VEHICLES_AND_SLOTS);
            } else {
                navigateToStage(AppState.WorkflowStage.LOAD_IMAGE);
            }
        });
    }

    private void updateInspectorForSlot(ParkingSlot slot) {
        AnalysisSummary summary = appState.getCurrentSummary();
        if (slot == null || summary == null) {
            inspectorView.updateSlot(null);
            return;
        }
        for (SlotAnalysisResult res : summary.getSlotResults()) {
            if (res.getId() == slot.getId()) {
                inspectorView.updateSlot(res);
                return;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 4. Image Loading, Detection & Analysis Execution
    // -------------------------------------------------------------------------

    private void loadInitialDefaultState() {
        try {
            File sampleFile = SampleDataGenerator.ensureSampleImageExists(SampleDataGenerator.DEFAULT_SAMPLE_IMAGE_PATH);
            loadImageFromFile(sampleFile);
            runDetectionAndAnalysis();
        } catch (Exception ex) {
            lblStatus.setText("Could not load default sample image: " + ex.getMessage());
            refreshAllViewsFromAppState();
        }
    }

    public void loadImageFromFile(File file) throws IOException {
        if (file == null) throw new IllegalArgumentException("Image file must not be null.");
        ImageLoader.LoadedImageMetadata meta = ImageLoader.loadWithMetadata(file);
        BufferedImage img = meta.image();
        String sha256 = DatasetManifest.computeFileSha256OrEmpty(file.toPath());

        // PKLot photographs are often overhead/perspective views where generic painted-line
        // localization is unreliable. If a matching XML sidecar exists, use its authoritative
        // parking-space polygons and labels rather than reporting zero spaces.
        List<ParkingSlot> annotatedSlots = loadAdjacentPkLotSlots(file.toPath(), meta, img);
        boolean loadedPkLotAnnotations = !annotatedSlots.isEmpty();
        LayoutDetectionResult detected = null;
        if (loadedPkLotAnnotations) {
            slotManager.setSlots(annotatedSlots);
        } else {
            detected = analysisService.autoDetectLayout(ImageEnhancer.autoEnhance(img));
            slotManager.setSlots(detected.getSlots());
        }

        boolean localizationReliable = loadedPkLotAnnotations || (detected != null && detected.isLocalizationReliable());
        String localizationMessage = loadedPkLotAnnotations
            ? String.format(Locale.ROOT,
                "Loaded %d parking-slot polygons and source occupancy labels from paired PKLot XML annotation '%s'. "
                + "These are dataset annotations; click Detect and Analyse to run live vehicle inference.",
                slotManager.getTotalCount(), findAdjacentPkLotXml(file.toPath()).getFileName())
            : (detected != null ? detected.getLocalizationSummary() : "Parking-slot localization unavailable.");

        appState.setLoadedImageAndSlots(
            meta, sha256, slotManager.getSlots(), localizationReliable, localizationMessage
        );

        parkingView.setImageAndSlots(img, slotManager.getSlots());
        summaryView.update(null, appState);
        slotTableView.setData(List.of());
        inspectorView.setFullImages(img, ImageEnhancer.autoEnhance(img));
        inspectorView.setAllResults(List.of());
        btnExportAnnotatedImage.setDisable(true);
        syncSessionSelectorFromAppState();

        if (appState.getCurrentStage() == AppState.WorkflowStage.LOAD_IMAGE) {
            appState.setCurrentStage(AppState.WorkflowStage.DETECT_VEHICLES_AND_SLOTS);
        }
        refreshAllViewsFromAppState();
        if (loadedPkLotAnnotations) {
            lblStatus.setText(String.format(Locale.ROOT,
                "Loaded '%s' with %d parking slots from paired PKLot XML. Click 'Detect and Analyse' to detect vehicles and classify the slots.",
                file.getName(), slotManager.getTotalCount()));
        } else {
            lblStatus.setText(String.format(Locale.ROOT,
                "Loaded '%s' (%d×%d px, %d slots from automatic image processing). %s",
                file.getName(), img.getWidth(), img.getHeight(), slotManager.getTotalCount(),
                slotManager.getTotalCount() > 0 ? "Click 'Detect and Analyse'." : "No matching XML was found; click Load Slot XML to select the paired PKLot annotation.") );
        }
    }

    private static String baseName(String name) {
        if (name == null) return "";
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    /** Finds a same-basename PKLot XML annotation next to the image, case-insensitively. */
    private static Path findAdjacentPkLotXml(Path imagePath) {
        if (imagePath == null || imagePath.getParent() == null || imagePath.getFileName() == null) return null;
        String name = imagePath.getFileName().toString();
        int dot = name.lastIndexOf('.');
        String base = dot > 0 ? name.substring(0, dot) : name;
        Path exact = imagePath.getParent().resolve(base + ".xml");
        if (Files.isRegularFile(exact)) return exact;
        try (var siblings = Files.list(imagePath.getParent())) {
            return siblings.filter(Files::isRegularFile)
                .filter(p -> p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".xml"))
                .filter(p -> {
                    String xml = p.getFileName().toString();
                    int xmlDot = xml.lastIndexOf('.');
                    String xmlBase = xmlDot > 0 ? xml.substring(0, xmlDot) : xml;
                    return xmlBase.equalsIgnoreCase(base);
                })
                .findFirst().orElse(null);
        } catch (IOException ignored) {
            return null;
        }
    }

    /**
     * Loads the original PKLot XML slot polygons and labels for a selected image when its
     * same-basename sidecar is present. Invalid/out-of-image annotations are rejected.
     */
    private static List<ParkingSlot> loadAdjacentPkLotSlots(Path imagePath,
                                                             ImageLoader.LoadedImageMetadata meta,
                                                             BufferedImage image) {
        Path xml = findAdjacentPkLotXml(imagePath);
        if (xml == null) return List.of();
        return loadPkLotSlotsFromXml(xml, meta, image);
    }

    private static List<ParkingSlot> loadPkLotSlotsFromXml(Path xml,
                                                            ImageLoader.LoadedImageMetadata meta,
                                                            BufferedImage image) {
        if (xml == null || !Files.isRegularFile(xml) || image == null) return List.of();
        List<ParkingSlot> slots = new java.util.ArrayList<>();
        for (PkLotDatasetImporter.ParsedSlotAnnotation annotation : PkLotDatasetImporter.parsePkLotXml(xml)) {
            if (annotation == null || !annotation.valid()
                    || (annotation.groundTruthLabel() != SlotStatus.OCCUPIED && annotation.groundTruthLabel() != SlotStatus.AVAILABLE)) {
                continue;
            }
            try {
                ParkingSlot slot = annotation.toParkingSlot();
                if (meta != null && meta.wasResized()) {
                    slot = slot.scaleToImage(meta.scaleX(), meta.scaleY(), image.getWidth(), image.getHeight());
                }
                if (slot.getX() < 0 || slot.getY() < 0
                        || slot.getX() + slot.getWidth() > image.getWidth()
                        || slot.getY() + slot.getHeight() > image.getHeight()) {
                    continue;
                }
                boolean polygonInside = slot.getPolygonVertices().stream().allMatch(pt ->
                    pt[0] >= 0 && pt[1] >= 0 && pt[0] <= image.getWidth() && pt[1] <= image.getHeight());
                if (!polygonInside) continue;
                // Before live analysis, show the annotation's reference label distinctly as a source label.
                slot.setStatus(annotation.groundTruthLabel());
                slots.add(slot);
            } catch (IllegalArgumentException ignored) {
                // One malformed annotation should not prevent valid slots in the image from loading.
            }
        }
        return List.copyOf(slots);
    }

    /** Loads official PKLot slot boundaries/labels when the XML file is stored separately from its image. */
    private void handleLoadSlotAnnotations() {
        ImageLoader.LoadedImageMetadata meta = appState.getCurrentImageMetadata();
        if (meta == null || appState.getCurrentImage() == null) {
            showError("No Image Loaded", "First click Open Image and select the parking photograph.");
            return;
        }
        FileChooser fc = new FileChooser();
        fc.setTitle("Select PKLot XML Annotation for Current Image");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("PKLot XML Annotations (*.xml)", "*.xml", "*.XML"));
        try {
            Path currentImagePath = Path.of(meta.filePath());
            Path adjacent = findAdjacentPkLotXml(currentImagePath);
            if (adjacent != null && adjacent.getParent() != null && Files.isDirectory(adjacent.getParent())) {
                fc.setInitialDirectory(adjacent.getParent().toFile());
                fc.setInitialFileName(adjacent.getFileName().toString());
            } else if (currentImagePath.getParent() != null && Files.isDirectory(currentImagePath.getParent())) {
                fc.setInitialDirectory(currentImagePath.getParent().toFile());
            }
        } catch (Exception ignored) { }
        File chosen = fc.showOpenDialog(getScene() != null ? getScene().getWindow() : null);
        if (chosen == null) return;
        try {
            String imageBase = baseName(Path.of(meta.filePath()).getFileName().toString());
            String xmlBase = baseName(chosen.getName());
            if (!imageBase.equalsIgnoreCase(xmlBase)) {
                showError("Annotation Does Not Match Image",
                    "Choose the XML whose filename matches the current image.\nImage: "
                    + imageBase + "\nXML: " + xmlBase);
                return;
            }
            List<ParkingSlot> annotated = loadPkLotSlotsFromXml(chosen.toPath(), meta, appState.getCurrentImage());
            if (annotated.isEmpty()) {
                showError("No Valid Parking Slots", "The selected XML did not contain valid OCCUPIED/AVAILABLE PKLot slot annotations for this image.");
                return;
            }
            slotManager.setSlots(annotated);
            String sha = DatasetManifest.computeFileSha256OrEmpty(Path.of(meta.filePath()));
            String message = String.format(Locale.ROOT,
                "Loaded %d parking slots from PKLot annotation %s. These are the source dataset labels; click Detect and Analyse to run live vehicle detection.",
                annotated.size(), chosen.getName());
            appState.setLoadedImageAndSlots(meta, sha, slotManager.getSlots(), true, message);
            parkingView.setImageAndSlots(appState.getCurrentImage(), slotManager.getSlots());
            summaryView.update(null, appState);
            slotTableView.setData(List.of());
            inspectorView.setAllResults(List.of());
            btnExportAnnotatedImage.setDisable(true);
            if (appState.getCurrentStage() == AppState.WorkflowStage.LOAD_IMAGE) {
                appState.setCurrentStage(AppState.WorkflowStage.DETECT_VEHICLES_AND_SLOTS);
            }
            refreshAllViewsFromAppState();
            lblStatus.setText(String.format(Locale.ROOT,
                "Loaded %d PKLot parking slots from %s. Click Detect and Analyse next.", annotated.size(), chosen.getName()));
        } catch (Exception ex) {
            showError("Slot Annotation Error", "Could not load the selected PKLot XML:\n" + ex.getMessage());
        }
    }

    private void handleOpenImage() {
        handleOpenImageInternal(false);
    }

    private void handleOpenAdditionalImage() {
        handleOpenImageInternal(true);
    }

    private void handleOpenImageInternal(boolean isAdditional) {
        FileChooser fc = new FileChooser();
        fc.setTitle(isAdditional ? "Add Additional Parking Image to Dataset" : "Open Parking Lot Image");
        fc.getExtensionFilters().add(
            new FileChooser.ExtensionFilter("Image Files (*.png, *.jpg, *.jpeg, *.bmp)", "*.png", "*.jpg", "*.jpeg", "*.bmp")
        );
        File file = fc.showOpenDialog(getScene() != null ? getScene().getWindow() : null);
        if (file == null) return;

        try {
            loadImageFromFile(file);
            runDetectionAndAnalysis();
        } catch (Exception ex) {
            showError("Image Load Error", "Failed to load image:\n" + ex.getMessage());
        }
    }

    private void handleCloseActiveSession() {
        String activeId = appState.getActiveSessionId();
        if (activeId != null) {
            appState.closeSession(activeId);
            syncSessionSelectorFromAppState();
            refreshAllViewsFromAppState();
            lblStatus.setText("Closed image session. Memory freed.");
        }
    }

    private void handleStartFreshDataset() {
        ButtonType btnClearSessionsOnly = new ButtonType("Clear Sessions (Keep Data)");
        ButtonType btnDiscardAll = new ButtonType("Discard Everything");
        ButtonType btnCancel = new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE);

        Alert confirm = new Alert(Alert.AlertType.CONFIRMATION,
            "Choose a reset option:\n\n"
            + "• Clear Sessions (Keep Data): Frees image memory while preserving the "
            + appState.getEligibleReviewedRecords().size() + " reviewed training records.\n\n"
            + "• Discard Everything: Clears all sessions and permanently wipes the training dataset.",
            btnClearSessionsOnly, btnDiscardAll, btnCancel);
        confirm.setTitle("Dataset and Session Reset");
        confirm.setHeaderText("Manage Sessions and Training Dataset");
        var res = confirm.showAndWait();
        if (res.isPresent()) {
            if (res.get() == btnClearSessionsOnly) {
                appState.clearAllSessions(true);
                parkingView.updateSessionList(List.of(), null);
                parkingView.setImageAndSlots(null, List.of());
                summaryView.update(null, appState);
                slotTableView.setData(List.of());
                inspectorView.setAllResults(List.of());
                lblStatus.setText("Image sessions cleared; " + appState.getEligibleReviewedRecords().size() + " reviewed records preserved.");
                refreshAllViewsFromAppState();
            } else if (res.get() == btnDiscardAll) {
                appState.clearAllSessions(false);
                parkingView.updateSessionList(List.of(), null);
                parkingView.setImageAndSlots(null, List.of());
                summaryView.update(null, appState);
                slotTableView.setData(List.of());
                inspectorView.setAllResults(List.of());
                lblStatus.setText("Dataset cleared. Open a new parking photograph to begin.");
                refreshAllViewsFromAppState();
            }
        }
    }

    private void handleSwitchImageSession(String selectedName) {
        if (selectedName == null || appState == null) return;
        for (ImageSessionRecord rec : appState.getAllImageSessions()) {
            if (selectedName.equals(formatSessionItem(rec))) {
                appState.switchToSession(rec.getSessionId());
                parkingView.setImageAndSlots(appState.getCurrentImage(), appState.getCurrentSlots());
                if (appState.getCurrentSummary() != null) {
                    applyAnalysisSummary(appState.getCurrentSummary());
                } else {
                    summaryView.update(null, appState);
                    slotTableView.setData(List.of());
                }
                lblStatus.setText("Switched to image session: " + rec.getFileName());
                break;
            }
        }
    }

    private static String formatSessionItem(ImageSessionRecord rec) {
        return rec.getFileName() + " (" + rec.getSlotCount() + " slots, " + rec.getOccupiedCount() + " occ)";
    }

    private void syncSessionSelectorFromAppState() {
        if (appState == null) return;
        List<ImageSessionRecord> sessions = appState.getAllImageSessions();
        List<String> items = new ArrayList<>();
        String activeItem = null;
        for (ImageSessionRecord s : sessions) {
            String item = formatSessionItem(s);
            items.add(item);
            if (s.getSessionId().equals(appState.getCurrentImageSha256())) {
                activeItem = item;
            }
        }
        parkingView.updateSessionList(items, activeItem);
    }

    public CompletableFuture<AnalysisSummary> handleLoadDemo() {
        return handleLoadDemo(false);
    }

    public CompletableFuture<AnalysisSummary> handleLoadDemo(boolean synchronous) {
        try {
            File sampleFile = SampleDataGenerator.ensureSampleImageExists(SampleDataGenerator.DEFAULT_SAMPLE_IMAGE_PATH);
            loadImageFromFile(sampleFile);
            return runDetectionAndAnalysis(synchronous);
        } catch (Exception ex) {
            showError("Demo Load Error", ex.getMessage());
            return CompletableFuture.failedFuture(ex);
        }
    }

    private void handleAutoDetectLayout() {
        if (appState.getCurrentImage() == null) {
            showError("No Image Loaded", "Please load a parking lot image first.");
            return;
        }
        try {
            LayoutDetectionResult detected = analysisService.autoDetectLayout(ImageEnhancer.autoEnhance(appState.getCurrentImage()));
            slotManager.setSlots(detected.getSlots());
            appState.setLoadedImageAndSlots(
                appState.getCurrentImageMetadata(),
                appState.getCurrentImageSha256(),
                slotManager.getSlots(),
                detected.isLocalizationReliable(),
                detected.getLocalizationSummary()
            );
            parkingView.setImageAndSlots(appState.getCurrentImage(), slotManager.getSlots());
            runDetectionAndAnalysis();
        } catch (Exception ex) {
            showError("Auto-Detection Error", ex.getMessage());
        }
    }

    private void handleLoadSlotsCsv() {
        if (appState.getCurrentImage() == null) {
            showError("No Image Loaded", "Please load a parking lot image before loading slot coordinates.");
            return;
        }
        FileChooser fc = new FileChooser();
        fc.setTitle("Load Slot Coordinates CSV");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files (*.csv)", "*.csv"));
        File file = fc.showOpenDialog(getScene() != null ? getScene().getWindow() : null);
        if (file == null) return;

        try {
            slotManager.loadFromCsv(file.getAbsolutePath());
            ImageLoader.LoadedImageMetadata meta = appState.getCurrentImageMetadata();
            BufferedImage img = appState.getCurrentImage();
            if (meta != null && meta.wasResized()) {
                slotManager.scaleAllSlots(meta.scaleX(), meta.scaleY(), img.getWidth(), img.getHeight());
            }
            appState.setLoadedImageAndSlots(
                meta, appState.getCurrentImageSha256(), slotManager.getSlots(),
                slotManager.getTotalCount() > 0,
                "Loaded " + slotManager.getTotalCount() + " slots from CSV '" + file.getName() + "'."
            );
            parkingView.setImageAndSlots(img, slotManager.getSlots());
            runDetectionAndAnalysis();
        } catch (Exception ex) {
            showError("CSV Configuration Error", "Failed to load slot configuration:\n" + ex.getMessage());
        }
    }

    private void handleSaveSlotsCsv() {
        if (slotManager.getTotalCount() == 0) {
            showError("No Slots to Save", "Draw or auto-detect slots first before saving.");
            return;
        }
        FileChooser fc = new FileChooser();
        fc.setTitle("Save Slot Coordinates CSV");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files (*.csv)", "*.csv"));
        fc.setInitialFileName("custom_slots.csv");
        File file = fc.showSaveDialog(getScene() != null ? getScene().getWindow() : null);
        if (file == null) return;

        try {
            slotManager.saveToCsv(file.getAbsolutePath());
            lblStatus.setText("Saved " + slotManager.getTotalCount() + " slots to " + file.getName());
        } catch (Exception ex) {
            showError("Save CSV Error", "Could not save slots CSV:\n" + ex.getMessage());
        }
    }

    private void handleClearSlots() {
        slotManager.clearSlots();
        appState.setCurrentSlots(List.of());
        if (appState.getCurrentImage() != null) {
            runDetectionAndAnalysis();
        }
    }

    public CompletableFuture<AnalysisSummary> runDetectionAndAnalysis() {
        return runDetectionAndAnalysis(false);
    }

    public CompletableFuture<AnalysisSummary> runDetectionAndAnalysis(boolean synchronous) {
        BufferedImage origSnapshot = appState.getOriginalImage();
        BufferedImage workingSnapshot = appState.getActiveWorkingImage();
        if (origSnapshot == null) {
            showError("No Image Loaded", "Please load a parking lot image in Step 1 before running detection.");
            return CompletableFuture.completedFuture(null);
        }

        final List<ParkingSlot> slotsSnapshot = new ArrayList<>(slotManager.getSlots());
        final boolean adaptiveMode = analysisService.isUseAdaptiveThreshold();
        final OccupancyDecisionEngine.AnalysisMode selectedMode = analysisService.getAnalysisMode();
        final boolean locReliable = appState.isSlotLocalizationReliable() && !slotsSnapshot.isEmpty();
        final String locMsg = appState.getSlotLocalizationMessage();

        parkingView.getBtnDetectAndAnalyse().setDisable(true);
        taskProgress.setVisible(true);
        taskProgress.setProgress(-1);
        lblStatus.setText("Running vehicle detection and slot analysis...");

        ImageSessionRecord activeSession = appState.getActiveSession();
        String identity = (activeSession != null) ? activeSession.resolveDetectionCacheIdentity() : null;

        CompletableFuture<AnalysisSummary> future = new CompletableFuture<>();

        if (synchronous || !Platform.isFxApplicationThread()) {
            try {
                AnalysisSummary summary = analysisService.analyze(origSnapshot, workingSnapshot, identity, slotsSnapshot, adaptiveMode, selectedMode, locReliable, locMsg);
                applyAnalysisSummary(summary);
                future.complete(summary);
            } catch (Throwable err) {
                handleAnalysisPipelineFailure(err, future);
            }
            return future;
        }

        Task<AnalysisSummary> task = new Task<>() {
            @Override
            protected AnalysisSummary call() throws Exception {
                return analysisService.analyze(origSnapshot, workingSnapshot, identity, slotsSnapshot, adaptiveMode, selectedMode, locReliable, locMsg);
            }
        };

        task.setOnSucceeded(e -> {
            AnalysisSummary summary = task.getValue();
            applyAnalysisSummary(summary);
            future.complete(summary);
        });

        task.setOnFailed(e -> {
            Throwable err = task.getException();
            handleAnalysisPipelineFailure(err, future);
        });

        Thread worker = new Thread(task, "PNC-Analysis-Worker");
        worker.setDaemon(true);
        worker.start();

        return future;
    }

    /**
     * Applies the active compatible Decision Tree to the feature vectors of the already analysed
     * current image and displays a separate prediction column. It deliberately does not mutate
     * slot statuses, detector suggestions, the occupancy summary, or image overlays.
     */
    private void runLoadedTreeSeparatelyOnCurrentImage() {
        AnalysisSummary summary = appState.getCurrentSummary();
        if (summary == null || summary.getSlotResults().isEmpty()) {
            showError("No Analysed Slots", "First load a parking image and click 'Detect and Analyse'. "
                + "Then load a compatible tree and click 'Run Tree Separately'.");
            return;
        }
        if (!appState.isCompatibleTenFeatureModelActive()
                || analysisService.getPredictor() == null
                || analysisService.getPredictor().getModel() == null
                || analysisService.getPredictor().getModel().getFeatureCount() != FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT) {
            showError("Compatible Decision Tree Required", "Load or train a compatible enhanced 10-feature Decision Tree first. "
                + "The detector's current results will remain unchanged.");
            return;
        }

        Map<Integer, DecisionTreeModel.DecisionTrace> traces = new java.util.LinkedHashMap<>();
        try {
            for (SlotAnalysisResult result : summary.getSlotResults()) {
                if (result.getFeatures() == null
                        || !FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION.equals(result.getFeatures().getSchemaVersion())) {
                    throw new IllegalStateException("Slot #" + result.getId() + " does not have the active enhanced 10-feature schema. "
                        + "Click Detect and Analyse again using the current enhancement policy before applying the tree.");
                }
                traces.put(result.getId(), analysisService.getPredictor().trace(result.getFeatures()));
            }
            slotTableView.setSeparatelyAppliedTreePredictions(traces, appState.getActiveModelSourceDescription());
            lblStatus.setText("Loaded tree evaluated separately on " + traces.size() + " slots. "
                + "Detector suggestions, Current Status, occupancy summary, and image overlays were NOT changed. "
                + "See the 'Loaded Tree Prediction' table column and select a slot for its tree rule path.");
        } catch (Exception ex) {
            slotTableView.clearSeparatelyAppliedTreePredictions();
            showError("Could Not Apply Decision Tree", ex.getMessage());
        }
    }

    private void handleAnalysisPipelineFailure(Throwable err, CompletableFuture<AnalysisSummary> future) {
        String exType = err != null ? err.getClass().getSimpleName() : "Error";
        String exMsg = (err != null && err.getMessage() != null && !err.getMessage().isBlank())
            ? err.getMessage()
            : "Unknown analysis pipeline exception";

        System.err.println("[PNC-Analysis-Worker ERROR] " + (err != null ? err.getClass().getName() : "Unknown") + ": " + exMsg);
        if (err != null) {
            err.printStackTrace(System.err);
        }

        String guidance = "\n\nTroubleshooting Suggestions:\n"
            + "• Ensure the loaded image is a valid RGB bitmap with positive dimensions.\n"
            + "• Verify that parking slot coordinates lie strictly within image boundaries.\n"
            + "• Verify that the ONNX vehicle detector model ('models/vehicle_detector.onnx') is present.\n"
            + "• Try resetting image enhancement or re-detecting bays.";
        String alertMsg = "Pipeline failure during detection and feature extraction (" + exType + "):\n\n" + exMsg + guidance;

        Runnable uiReset = () -> {
            parkingView.getBtnDetectAndAnalyse().setDisable(false);
            taskProgress.setVisible(false);
            lblStatus.setText("Analysis failed (" + exType + "): " + exMsg);
            showError("Analysis Pipeline Failed", alertMsg);
        };

        if (Platform.isFxApplicationThread()) {
            uiReset.run();
        } else {
            Platform.runLater(uiReset);
        }

        if (future != null) {
            future.completeExceptionally(err != null ? err : new RuntimeException("Analysis failed"));
        }
    }

    private void applyAnalysisSummary(AnalysisSummary summary) {
        parkingView.getBtnDetectAndAnalyse().setDisable(false);
        taskProgress.setVisible(false);
        appState.updateAnalysisSummary(summary);

        if (summary.getDetectionBatchResult() != null) {
            updateDetectorStatusBadge(
                summary.getDetectionBatchResult().status(),
                summary.getDetectionBatchResult().formatExecutionSummary()
            );
        }

        parkingView.updateAnalysisResults(
            summary.getAnnotatedImage(),
            summary.getSlotResults(),
            summary.getVehicleDetections()
        );
        slotTableView.setData(summary.getSlotResults());
        inspectorView.setFullImages(appState.getOriginalImage(), appState.getActiveWorkingImage());
        inspectorView.setAllResults(summary.getSlotResults());
        summaryView.update(summary, appState);
        btnExportAnnotatedImage.setDisable(false);
        syncSessionSelectorFromAppState();

        refreshAllViewsFromAppState();

        if (!summary.isSlotLocalizationReliable() || summary.getTotalSlots() == 0) {
            lblStatus.setText(String.format(Locale.ROOT,
                "Detection complete (%d ms): Parking-slot localization uncertain (0 spaces invented). Detected %d vehicle(s).",
                summary.getExecutionTimeMs(), summary.getVehicleDetections().size()
            ));
        } else {
            lblStatus.setText(String.format(Locale.ROOT,
                "Detection complete in %d ms: %d Occupied, %d Available, %d Needs Review (%d vehicle(s) detected). Proceed to Step 3 to review slots.",
                summary.getExecutionTimeMs(),
                summary.getOccupiedSlots(),
                summary.getAvailableSlots(),
                summary.getNeedsReviewSlots(),
                summary.getVehicleDetections().size()
            ));
        }
    }

    private void handleExportAnnotatedImage() {
        AnalysisSummary summary = appState.getCurrentSummary();
        if (summary == null || summary.getAnnotatedImage() == null) {
            showError("No Annotated Image", "Run detection first before exporting.");
            return;
        }
        FileChooser fc = new FileChooser();
        fc.setTitle("Export Annotated Parking Image");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("PNG Image (*.png)", "*.png"));
        fc.setInitialFileName("parknchill_v23_annotated_result.png");
        File file = fc.showSaveDialog(getScene() != null ? getScene().getWindow() : null);
        if (file == null) return;

        try {
            ImageIO.write(summary.getAnnotatedImage(), "png", file);
            lblStatus.setText("Exported annotated image to " + file.getAbsolutePath());
        } catch (IOException ex) {
            showError("Export Error", "Failed to save annotated image:\n" + ex.getMessage());
        }
    }

    private void updateDetectorStatusBadge(VehicleDetectionService.DetectorStatus status, String detailMsg) {
        String stName = status != null ? status.name() : "MODEL_MISSING";
        lblDetectorStatusBadge.setText("Detector: " + stName);
        String bg = status == VehicleDetectionService.DetectorStatus.READY ? "#064e3b" : "#78350f";
        String fg = status == VehicleDetectionService.DetectorStatus.READY ? "#a7f3d0" : "#fde68a";
        String border = status == VehicleDetectionService.DetectorStatus.READY ? "#10b981" : "#f59e0b";
        lblDetectorStatusBadge.setStyle(String.format(
            "-fx-background-color: %s; -fx-text-fill: %s; -fx-border-color: %s; " +
            "-fx-font-weight: bold; -fx-font-size: 11px; -fx-padding: 3px 8px; -fx-background-radius: 4px; -fx-border-radius: 4px;",
            bg, fg, border));
        if (detailMsg != null && !detailMsg.isBlank()) {
            lblDetectorStatusBadge.setTooltip(new Tooltip(detailMsg));
        }
    }

    private void showError(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR, message, ButtonType.OK);
        alert.setTitle(title);
        alert.setHeaderText(title);
        alert.showAndWait();
    }

    public AppState getAppState()                               { return appState; }
    public ParkingView getParkingView()                         { return parkingView; }
    public ResultSummaryView getSummaryView()                   { return summaryView; }
    public SlotTableView getSlotTableView()                     { return slotTableView; }
    public PipelineInspectorView getInspectorView()             { return inspectorView; }
    public ModelDetailsView getModelDetailsView()               { return modelDetailsView; }
    public ParkingSlotManager getSlotManager()                  { return slotManager; }
    public Label getLblAppTitle()                               { return lblAppTitle; }
    public Label getLblAppSubtitle()                            { return lblAppSubtitle; }
    public Label getLblWorkflowStatusBadge()                    { return lblWorkflowStatusBadge; }
    public Label getLblDetectorStatusBadge()                    { return lblDetectorStatusBadge; }
    public Label getLblPrerequisiteBanner()                     { return lblPrerequisiteBanner; }
    public TextField getTxtReviewerId()                         { return txtReviewerId; }
    public Label getLblReviewerValidation()                     { return lblReviewerValidation; }
    public Map<AppState.WorkflowStage, Button> getStageButtons() { return stageButtons; }
    public StackPane getStageWorkspaceContainer()               { return stageWorkspaceContainer; }
}