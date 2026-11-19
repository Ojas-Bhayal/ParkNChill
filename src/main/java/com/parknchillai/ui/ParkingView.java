package com.parknchillai.ui;

import com.parknchillai.app.AppState;
import com.parknchillai.app.SlotAnalysisResult;
import com.parknchillai.imageprocessing.ImageEnhancer;
import com.parknchillai.detection.DetectionStrategy;
import com.parknchillai.detection.OccupancyDecisionEngine;
import com.parknchillai.detection.VehicleDetection;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Cursor;
import javafx.scene.Group;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Spinner;
import javafx.scene.control.TitledPane;
import javafx.scene.control.Tooltip;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseButton;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Polygon;
import javafx.scene.shape.Rectangle;
import javafx.scene.shape.Shape;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.function.IntConsumer;

/**
 * Simplified, image-focused Parking Workspace (ParkNChill v2.3 Requirements 2.B, 5 &amp; 6):
 * <ul>
 *   <li>Adapts its primary action button to the active workflow stage ({@code Open Image} in Step 1,
 *       {@code Detect and Analyse} in Step 2, and clean zoom/reset controls in Step 3) so there is
 *       one concise main action per stage without competing controls.</li>
 *   <li>Renders custom angled/trapezoidal PKLot polygon contours ({@link Polygon}) when
 *       {@link ParkingSlot#hasCustomPolygon()} is {@code true} (Requirement 5).</li>
 *   <li>Keeps all advanced image-processing settings, layer toggles, and boundary editing tools
 *       inside a collapsed-by-default {@code Advanced Image Options} {@link TitledPane}.</li>
 * </ul>
 */
public class ParkingView extends VBox {

    public static final double DEFAULT_FILL_OPACITY = 0.12;
    public static final double SELECTED_FILL_OPACITY = 0.08;

    public enum InteractionMode {
        SELECT_INSPECT("Inspect / Select Slot"),
        DRAW_SINGLE_SLOT("+ Draw Single Slot"),
        DRAW_ROW_GRID("+ Draw Row Grid (N Bays)"),
        RECTIFY_4_CORNERS("4-Corner Perspective Warp");

        private final String displayName;
        InteractionMode(String displayName) { this.displayName = displayName; }
        @Override public String toString()  { return displayName; }
    }

    public record DrawnBox(int x, int y, int width, int height, int bayCount) {}

    private final ImageView imageView = new ImageView();
    private final Pane overlayPane = new Pane();
    private final Group contentGroup = new Group();
    private final StackPane centeredContainer = new StackPane();
    private final ScrollPane scrollPane = new ScrollPane();

    private final Button btnOpenImage = new Button("Open Image");
    private final Button btnAddImage = new Button("Add Image...");
    private final Button btnFreshDataset = new Button("Fresh Dataset...");
    private final ComboBox<String> sessionSelector = new ComboBox<>();
    private final Button btnCloseSession = new Button("Close");
    private final Button btnLoadSlotXml = new Button("Load Slot XML");
    private final Button btnDetectAndAnalyse = new Button("Detect and Analyse");
    private final Button btnRunLoadedTree = new Button("Run Tree Separately");
    private final Button btnResetView = new Button("Reset View");
    private final Button btnToggleEnhancement = new Button("Show Original");
    private final Button btnViewIvpStages = new Button("View IVP Stages");
    private final Button btnEnhanceImage = new Button("Enhance Image");
    private final Label lblEnhancementPreview = new Label("Auto enhancement will be applied when an image is loaded.");

    private final DoubleProperty zoomLevel = new SimpleDoubleProperty(1.2);
    private final CheckBox showBoxesCheck = new CheckBox("Slot Boundaries");
    private final CheckBox showDetectionsCheck = new CheckBox("Vehicle Boxes");
    private final CheckBox showBadgesCheck = new CheckBox("Slot Labels");
    private final CheckBox showFeatureOverlaysCheck = new CheckBox("Interior Tint");
    private final CheckBox rawImageOnlyCheck = new CheckBox("Original Photo Only");
    private final ComboBox<String> opacitySelector = new ComboBox<>();
    private final ComboBox<String> thresholdSelector = new ComboBox<>();
    private final ComboBox<OccupancyDecisionEngine.AnalysisMode> analysisModeSelector = new ComboBox<>();
    private final ComboBox<DetectionStrategy> detectionStrategySelector = new ComboBox<>();
    private final ComboBox<InteractionMode> modeSelector = new ComboBox<>();
    private final Spinner<Integer> bayCountSpinner = new Spinner<>(2, 24, 4);

    private final Button btnAutoDetectSlots = new Button("Re-Detect Slots");
    private final Button btnLoadSlotsCsv = new Button("Load Slots CSV");
    private final Button btnSaveSlotsCsv = new Button("Save Slots CSV");
    private final Button btnClearSlots = new Button("Clear Slots");
    private final TitledPane advancedImageOptionsPane = new TitledPane();

    private double fillOpacityFraction = DEFAULT_FILL_OPACITY;
    private BufferedImage originalAwtImage;
    private BufferedImage enhancedAwtImage;
    private boolean enhancedPreviewActive;
    private BufferedImage currentAwtImage;
    private BufferedImage currentAnnotatedAwtImage;
    private List<ParkingSlot> currentSlots;
    private List<VehicleDetection> currentVehicleDetections = new ArrayList<>();
    private final Map<Integer, SlotAnalysisResult> analysisMap = new HashMap<>();

    private Runnable onOpenImageAction;
    private Runnable onAddImageAction;
    private Runnable onFreshDatasetAction;
    private Runnable onCloseSessionAction;
    private Consumer<String> onSessionSelectedAction;
    private Runnable onLoadSlotXmlAction;
    private Runnable onDetectAndAnalyseAction;
    private Runnable onRunLoadedTreeAction;
    private Runnable onViewIvpStagesAction;
    private Runnable onEnhanceImageAction;
    private Runnable onAutoDetectSlotsAction;
    private Runnable onLoadSlotsCsvAction;
    private Runnable onSaveSlotsCsvAction;
    private Runnable onClearSlotsAction;
    private Consumer<Boolean> onAdaptiveThresholdChanged;
    private Consumer<OccupancyDecisionEngine.AnalysisMode> onAnalysisModeChanged;
    private boolean suppressAnalysisModeCallback;
    private Consumer<DetectionStrategy> onDetectionStrategyChanged;

    private Consumer<ParkingSlot> slotSelectionListener;
    private Consumer<DrawnBox> onBoxDrawnListener;
    private Consumer<double[][]> onPerspectiveQuadListener;
    private DoubleConsumer onOpacityChangedListener;
    private final List<double[]> cornerPoints = new ArrayList<>(4);
    private IntConsumer onSlotDeletedListener;
    private ParkingSlot selectedSlot;

    private double dragStartX;
    private double dragStartY;
    private final Rectangle dragPreviewRect = new Rectangle();

    public ParkingView() {
        setSpacing(6);
        setPadding(new Insets(8));
        VBox.setVgrow(this, Priority.ALWAYS);

        HBox primaryToolbar = createPrimaryWorkspaceBar();
        HBox enhancementBar = createEnhancementPreviewBar();
        HBox legendBar = createCompactLegendBar();
        TitledPane advPane = createCollapsedAdvancedOptionsPane();

        imageView.setPreserveRatio(true);
        imageView.setSmooth(true);

        overlayPane.setPickOnBounds(true);
        overlayPane.prefWidthProperty().bind(imageView.fitWidthProperty());
        overlayPane.prefHeightProperty().bind(imageView.fitHeightProperty());

        dragPreviewRect.setFill(Color.rgb(56, 189, 248, 0.15));
        dragPreviewRect.setStroke(Color.rgb(56, 189, 248, 0.95));
        dragPreviewRect.setStrokeWidth(2.0);
        dragPreviewRect.getStrokeDashArray().addAll(6.0, 4.0);
        dragPreviewRect.setVisible(false);
        dragPreviewRect.setMouseTransparent(true);

        contentGroup.getChildren().addAll(imageView, overlayPane);
        contentGroup.scaleXProperty().bind(zoomLevel);
        contentGroup.scaleYProperty().bind(zoomLevel);

        centeredContainer.setAlignment(Pos.CENTER);
        centeredContainer.setPadding(new Insets(16));
        centeredContainer.getChildren().add(contentGroup);
        centeredContainer.setStyle("-fx-background-color: #0b0f19;");

        scrollPane.setContent(centeredContainer);
        scrollPane.setPannable(true);
        scrollPane.setFitToWidth(true);
        scrollPane.setFitToHeight(true);
        scrollPane.getStyleClass().add("parking-scroll-pane");
        VBox.setVgrow(scrollPane, Priority.ALWAYS);

        getChildren().addAll(primaryToolbar, enhancementBar, legendBar, scrollPane, advPane);

        showBoxesCheck.setSelected(true);
        showDetectionsCheck.setSelected(true);
        showBadgesCheck.setSelected(true);
        showFeatureOverlaysCheck.setSelected(true);
        rawImageOnlyCheck.setSelected(false);

        showBoxesCheck.selectedProperty().addListener((obs, oldVal, newVal) -> renderOverlays());
        showDetectionsCheck.selectedProperty().addListener((obs, oldVal, newVal) -> renderOverlays());
        showBadgesCheck.selectedProperty().addListener((obs, oldVal, newVal) -> renderOverlays());
        showFeatureOverlaysCheck.selectedProperty().addListener((obs, oldVal, newVal) -> renderOverlays());
        rawImageOnlyCheck.selectedProperty().addListener((obs, oldVal, newVal) -> renderOverlays());

        setupCanvasDrawingHandlers();
    }

    /**
     * Highlights the single concise primary action for {@code stage} (Requirement 6).
     */
    public void configurePrimaryActionForStage(AppState.WorkflowStage stage) {
        if (stage == AppState.WorkflowStage.LOAD_IMAGE) {
            btnOpenImage.getStyleClass().setAll("btn-primary");
            btnDetectAndAnalyse.getStyleClass().setAll("btn-secondary");
        } else {
            btnOpenImage.getStyleClass().setAll("btn-secondary");
            btnDetectAndAnalyse.getStyleClass().setAll("btn-primary");
        }
    }

    private HBox createPrimaryWorkspaceBar() {
        HBox bar = new HBox(8);
        bar.setAlignment(Pos.CENTER_LEFT);
        bar.setPadding(new Insets(6, 10, 6, 10));
        bar.getStyleClass().add("viewport-controls-bar");

        btnOpenImage.getStyleClass().add("btn-secondary");
        btnOpenImage.setMinWidth(96);
        btnOpenImage.setOnAction(e -> {
            if (onOpenImageAction != null) onOpenImageAction.run();
        });

        btnAddImage.getStyleClass().add("btn-secondary");
        btnAddImage.setMinWidth(96);
        btnAddImage.setTooltip(new Tooltip("Analyse an additional parking image and add its slots to the active dataset"));
        btnAddImage.setOnAction(e -> {
            if (onAddImageAction != null) onAddImageAction.run();
        });

        sessionSelector.setStyle("-fx-background-color: #1e293b; -fx-text-fill: #38bdf8; -fx-border-color: #475569;");
        sessionSelector.setPromptText("Active Image");
        sessionSelector.setVisible(false);
        sessionSelector.setOnAction(e -> {
            String sel = sessionSelector.getValue();
            if (sel != null && onSessionSelectedAction != null) onSessionSelectedAction.accept(sel);
        });

        btnCloseSession.getStyleClass().add("btn-tool");
        btnCloseSession.setMinWidth(64);
        btnCloseSession.setTooltip(new Tooltip("Close the active image session to free memory"));
        btnCloseSession.setVisible(false);
        btnCloseSession.setOnAction(e -> {
            if (onCloseSessionAction != null) onCloseSessionAction.run();
        });

        btnLoadSlotXml.getStyleClass().add("btn-tool");
        btnLoadSlotXml.setMinWidth(104);
        btnLoadSlotXml.setTooltip(new Tooltip("Load PKLot parking-space boundaries and labels from an XML annotation file for the current image"));
        btnLoadSlotXml.setOnAction(e -> {
            if (onLoadSlotXmlAction != null) onLoadSlotXmlAction.run();
        });

        btnDetectAndAnalyse.getStyleClass().add("btn-primary");
        btnDetectAndAnalyse.setMinWidth(136);
        btnDetectAndAnalyse.setOnAction(e -> {
            if (onDetectAndAnalyseAction != null) onDetectAndAnalyseAction.run();
        });

        btnRunLoadedTree.getStyleClass().add("btn-tool");
        btnRunLoadedTree.setMinWidth(132);
        btnRunLoadedTree.setTooltip(new Tooltip(
            "Apply the active compatible 10-feature Decision Tree to the current slot features as a separate result. " +
            "It does not replace detector suggestions, current status, or image overlays."));
        btnRunLoadedTree.setOnAction(e -> {
            if (onRunLoadedTreeAction != null) onRunLoadedTreeAction.run();
        });

        btnResetView.getStyleClass().add("btn-tool");
        btnEnhanceImage.getStyleClass().add("btn-tool");
        btnEnhanceImage.setMinWidth(104);
        btnEnhanceImage.setDisable(true);
        btnEnhanceImage.setTooltip(new Tooltip("Open enhancement controls for the full image or the selected parking slot"));
        btnEnhanceImage.setOnAction(e -> { if (onEnhanceImageAction != null) onEnhanceImageAction.run(); });

        btnFreshDataset.getStyleClass().add("btn-tool");
        btnFreshDataset.setMinWidth(96);
        btnFreshDataset.setTooltip(new Tooltip("Discard current dataset and start a fresh collection session"));
        btnFreshDataset.setOnAction(e -> { if (onFreshDatasetAction != null) onFreshDatasetAction.run(); });

        btnResetView.setMinWidth(82);
        btnResetView.setTooltip(new Tooltip("Reset zoom to fit and return to standard slot inspection mode"));
        btnResetView.setOnAction(e -> resetView());

        HBox spacer = new HBox();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        Label zoomLabel = new Label("Zoom:");
        zoomLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: #cbd5e1;");

        Button btnZoomOut = new Button("Zoom Out");
        btnZoomOut.getStyleClass().add("btn-tool");
        btnZoomOut.setOnAction(e -> setZoom(zoomLevel.get() - 0.2));

        Label zoomVal = new Label("120%");
        zoomVal.setStyle("-fx-text-fill: #38bdf8; -fx-font-weight: bold; -fx-min-width: 44px; -fx-alignment: center;");
        zoomLevel.addListener((obs, o, n) -> zoomVal.setText((int) Math.round(n.doubleValue() * 100) + "%"));

        Button btnZoomIn = new Button("Zoom In");
        btnZoomIn.getStyleClass().add("btn-tool");
        btnZoomIn.setOnAction(e -> setZoom(zoomLevel.get() + 0.2));

        // Keep the everyday toolbar short: open, analyze, fit, and basic zoom.
        // 100% and Fit Image duplicated the Reset View action and crowded small screens.
        bar.getChildren().addAll(
            btnOpenImage, btnAddImage, sessionSelector, btnCloseSession, btnLoadSlotXml, btnDetectAndAnalyse, btnRunLoadedTree,
            btnEnhanceImage, btnViewIvpStages, btnResetView, btnFreshDataset,
            spacer,
            zoomLabel, btnZoomOut, zoomVal, btnZoomIn
        );
        return bar;
    }

    private HBox createEnhancementPreviewBar() {
        HBox bar = new HBox(8);
        bar.setAlignment(Pos.CENTER_LEFT);
        bar.setPadding(new Insets(5, 10, 5, 10));
        bar.setStyle("-fx-background-color: #101a2c; -fx-border-color: #334155; -fx-border-radius: 5px; -fx-background-radius: 5px;");

        lblEnhancementPreview.setWrapText(true);
        lblEnhancementPreview.setStyle("-fx-text-fill: #cbd5e1; -fx-font-size: 11.5px;");
        lblEnhancementPreview.setText("Image enhancement preview: Auto");
        lblEnhancementPreview.setMaxWidth(420);
        HBox.setHgrow(lblEnhancementPreview, Priority.ALWAYS);

        btnToggleEnhancement.getStyleClass().add("btn-tool");
        btnToggleEnhancement.setDisable(true);
        btnToggleEnhancement.setTooltip(new Tooltip(
            "Toggle between the canonical enhanced image (ivp-enhance-v1) and the original photograph. Vehicle detection runs on natural pixels; IVP features are computed from enhanced pixels."));
        btnToggleEnhancement.setOnAction(e -> setEnhancedPreviewActive(!enhancedPreviewActive));

        btnViewIvpStages.getStyleClass().add("btn-secondary");
        btnViewIvpStages.setDisable(true);
        btnViewIvpStages.setTooltip(new Tooltip("Show actual Original, Enhanced, Grayscale, Saturation, Blur, Threshold, Morphology, Connected Components, and Sobel images."));
        btnViewIvpStages.setOnAction(e -> {
            if (onViewIvpStagesAction != null) onViewIvpStagesAction.run();
        });

        bar.getChildren().addAll(lblEnhancementPreview, btnToggleEnhancement);
        return bar;
    }

    private HBox createCompactLegendBar() {
        HBox legend = new HBox(14);
        legend.setAlignment(Pos.CENTER_LEFT);
        legend.setPadding(new Insets(4, 10, 4, 10));
        legend.setStyle("-fx-background-color: #0f172a; -fx-border-color: #334155; -fx-border-radius: 5px; -fx-background-radius: 5px;");

        Label lblLegendTitle = new Label("Legend:");
        lblLegendTitle.setStyle("-fx-font-weight: bold; -fx-text-fill: #cbd5e1; -fx-font-size: 11.5px;");

        Label legVehicle = new Label("[--] Detected Vehicle (Sky-Blue Dashed)");
        legVehicle.setStyle("-fx-text-fill: #38bdf8; -fx-font-weight: bold; -fx-font-size: 11.5px;");

        Label legOccupied = new Label("[Solid Red] Occupied Slot");
        legOccupied.setStyle("-fx-text-fill: #fca5a5; -fx-font-weight: bold; -fx-font-size: 11.5px;");

        Label legAvailable = new Label("[Solid Green] Available Slot");
        legAvailable.setStyle("-fx-text-fill: #6ee7b7; -fx-font-weight: bold; -fx-font-size: 11.5px;");

        Label legReview = new Label("[-- Amber] Needs Review");
        legReview.setStyle("-fx-text-fill: #fde68a; -fx-font-weight: bold; -fx-font-size: 11.5px;");

        legend.getChildren().addAll(lblLegendTitle, legVehicle, legOccupied, legAvailable, legReview);
        return legend;
    }

    private TitledPane createCollapsedAdvancedOptionsPane() {
        VBox content = new VBox(8);
        content.setPadding(new Insets(8, 10, 8, 10));
        content.setStyle("-fx-background-color: #0f172a;");

        HBox rowLayers = new HBox(12);
        rowLayers.setAlignment(Pos.CENTER_LEFT);
        Label lblLayers = new Label("Overlay Layers:");
        lblLayers.setStyle("-fx-font-weight: bold; -fx-text-fill: #cbd5e1;");
        showBoxesCheck.setStyle("-fx-text-fill: #f8fafc;");
        showDetectionsCheck.setStyle("-fx-text-fill: #38bdf8;");
        showBadgesCheck.setStyle("-fx-text-fill: #f8fafc;");
        showFeatureOverlaysCheck.setStyle("-fx-text-fill: #a7f3d0;");
        rawImageOnlyCheck.setStyle("-fx-text-fill: #fbbf24;");

        Label fillLbl = new Label("Interior Fill:");
        fillLbl.setStyle("-fx-text-fill: #cbd5e1; -fx-font-weight: bold;");
        opacitySelector.getItems().addAll("0% (Outline)", "8%", "12% (Default)", "15%", "25%");
        opacitySelector.setValue("12% (Default)");
        opacitySelector.setPrefWidth(120);
        opacitySelector.setOnAction(e -> {
            String sel = opacitySelector.getValue();
            if (sel != null) {
                if (sel.startsWith("0%")) setFillOpacityFraction(0.0);
                else if (sel.startsWith("8%")) setFillOpacityFraction(0.08);
                else if (sel.startsWith("12%")) setFillOpacityFraction(0.12);
                else if (sel.startsWith("15%")) setFillOpacityFraction(0.15);
                else if (sel.startsWith("25%")) setFillOpacityFraction(0.25);
            }
        });
        rowLayers.getChildren().addAll(
            lblLayers, showBoxesCheck, showDetectionsCheck, showBadgesCheck,
            showFeatureOverlaysCheck, rawImageOnlyCheck, fillLbl, opacitySelector
        );

        HBox rowTools = new HBox(10);
        rowTools.setAlignment(Pos.CENTER_LEFT);
        Label modeLabel = new Label("Boundary Tool:");
        modeLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: #38bdf8;");
        modeSelector.getItems().addAll(InteractionMode.values());
        modeSelector.setValue(InteractionMode.SELECT_INSPECT);
        modeSelector.setPrefWidth(185);

        Label baysLabel = new Label("Bays:");
        baysLabel.setStyle("-fx-text-fill: #cbd5e1;");
        bayCountSpinner.setPrefWidth(64);
        baysLabel.setVisible(false);
        baysLabel.setManaged(false);
        bayCountSpinner.setVisible(false);
        bayCountSpinner.setManaged(false);

        modeSelector.valueProperty().addListener((obs, oldMode, newMode) -> {
            cornerPoints.clear();
            boolean isRowMode = (newMode == InteractionMode.DRAW_ROW_GRID);
            baysLabel.setVisible(isRowMode);
            baysLabel.setManaged(isRowMode);
            bayCountSpinner.setVisible(isRowMode);
            bayCountSpinner.setManaged(isRowMode);
            boolean isDrawing = (newMode != InteractionMode.SELECT_INSPECT);
            scrollPane.setPannable(!isDrawing);
            overlayPane.setCursor(isDrawing ? Cursor.CROSSHAIR : Cursor.DEFAULT);
            renderOverlays();
        });

        btnAutoDetectSlots.getStyleClass().add("btn-tool");
        btnAutoDetectSlots.setOnAction(e -> { if (onAutoDetectSlotsAction != null) onAutoDetectSlotsAction.run(); });

        btnLoadSlotsCsv.getStyleClass().add("btn-tool");
        btnLoadSlotsCsv.setOnAction(e -> { if (onLoadSlotsCsvAction != null) onLoadSlotsCsvAction.run(); });

        btnSaveSlotsCsv.getStyleClass().add("btn-tool");
        btnSaveSlotsCsv.setOnAction(e -> { if (onSaveSlotsCsvAction != null) onSaveSlotsCsvAction.run(); });

        btnClearSlots.getStyleClass().add("btn-tool");
        btnClearSlots.setOnAction(e -> { if (onClearSlotsAction != null) onClearSlotsAction.run(); });

        Label threshLbl = new Label("Segmentation:");
        threshLbl.setStyle("-fx-text-fill: #cbd5e1; -fx-font-weight: bold;");
        thresholdSelector.getItems().addAll("Otsu (Global)", "Adaptive (Shadow-Free)");
        thresholdSelector.setValue("Otsu (Global)");
        thresholdSelector.setPrefWidth(155);
        thresholdSelector.setOnAction(e -> {
            if (onAdaptiveThresholdChanged != null) {
                onAdaptiveThresholdChanged.accept(thresholdSelector.getSelectionModel().getSelectedIndex() == 1);
            }
        });

        Label modeClassifierLbl = new Label("Classifier Mode:");
        modeClassifierLbl.setStyle("-fx-text-fill: #cbd5e1; -fx-font-weight: bold;");
        analysisModeSelector.getItems().addAll(
            OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION,
            OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE,
            OccupancyDecisionEngine.AnalysisMode.COMPARISON
        );
        analysisModeSelector.setValue(OccupancyDecisionEngine.AnalysisMode.VEHICLE_DETECTION);
        analysisModeSelector.setPrefWidth(195);
        analysisModeSelector.setOnAction(e -> {
            if (!suppressAnalysisModeCallback
                    && onAnalysisModeChanged != null
                    && analysisModeSelector.getValue() != null) {
                onAnalysisModeChanged.accept(analysisModeSelector.getValue());
            }
        });

        Label stratLbl = new Label("Vehicle Strategy:");
        stratLbl.setStyle("-fx-text-fill: #cbd5e1; -fx-font-weight: bold;");
        detectionStrategySelector.getItems().addAll(
            DetectionStrategy.ACCURATE,
            DetectionStrategy.FAST
        );
        detectionStrategySelector.setValue(DetectionStrategy.ACCURATE);
        detectionStrategySelector.setPrefWidth(160);
        detectionStrategySelector.setOnAction(e -> {
            if (onDetectionStrategyChanged != null && detectionStrategySelector.getValue() != null) {
                onDetectionStrategyChanged.accept(detectionStrategySelector.getValue());
            }
        });

        rowTools.getChildren().addAll(
            modeLabel, modeSelector, baysLabel, bayCountSpinner,
            btnAutoDetectSlots, btnLoadSlotsCsv, btnSaveSlotsCsv, btnClearSlots,
            threshLbl, thresholdSelector, modeClassifierLbl, analysisModeSelector,
            stratLbl, detectionStrategySelector
        );

        content.getChildren().addAll(rowLayers, rowTools);

        advancedImageOptionsPane.setText("Advanced Image Options (Overlay Layers, Boundary Editing & Threshold Settings)");
        advancedImageOptionsPane.setContent(content);
        advancedImageOptionsPane.setCollapsible(true);
        advancedImageOptionsPane.setExpanded(false);
        return advancedImageOptionsPane;
    }

    public void resetView() {
        modeSelector.setValue(InteractionMode.SELECT_INSPECT);
        rawImageOnlyCheck.setSelected(false);
        showBoxesCheck.setSelected(true);
        showDetectionsCheck.setSelected(true);
        showBadgesCheck.setSelected(true);
        autoFitView();
        renderOverlays();
    }

    public void setAdvancedOptionsExpanded(boolean expanded) {
        advancedImageOptionsPane.setExpanded(expanded);
    }

    public boolean isAdvancedOptionsExpanded() {
        return advancedImageOptionsPane.isExpanded();
    }

    public TitledPane getAdvancedImageOptionsPane() {
        return advancedImageOptionsPane;
    }

    public Button getBtnOpenImage()        { return btnOpenImage; }
    public Button getBtnAddImage()         { return btnAddImage; }
    public Button getBtnFreshDataset()     { return btnFreshDataset; }
    public ComboBox<String> getSessionSelector() { return sessionSelector; }
    public Button getBtnLoadSlotXml()      { return btnLoadSlotXml; }
    public Button getBtnDetectAndAnalyse() { return btnDetectAndAnalyse; }
    public Button getBtnRunLoadedTree()    { return btnRunLoadedTree; }
    public Button getBtnResetView()        { return btnResetView; }

    public void setOnOpenImageAction(Runnable r)                         { this.onOpenImageAction = r; }
    public void setOnAddImageAction(Runnable r)                          { this.onAddImageAction = r; }
    public void setOnFreshDatasetAction(Runnable r)                      { this.onFreshDatasetAction = r; }
    public void setOnSessionSelectedAction(Consumer<String> c)           { this.onSessionSelectedAction = c; }
    public void setOnLoadSlotXmlAction(Runnable r)                       { this.onLoadSlotXmlAction = r; }
    public void setOnDetectAndAnalyseAction(Runnable r)                  { this.onDetectAndAnalyseAction = r; }
    public void setOnRunLoadedTreeAction(Runnable r)                     { this.onRunLoadedTreeAction = r; }
    public void setOnViewIvpStagesAction(Runnable r)                      { this.onViewIvpStagesAction = r; }
    public void setOnEnhanceImageAction(Runnable r)                       { this.onEnhanceImageAction = r; }

    public void updateSessionList(List<String> sessions, String activeSession) {
        if (sessions == null || sessions.size() <= 1) {
            sessionSelector.setVisible(false);
            sessionSelector.setManaged(false);
            btnCloseSession.setVisible(false);
            btnCloseSession.setManaged(false);
        } else {
            sessionSelector.setVisible(true);
            sessionSelector.setManaged(true);
            sessionSelector.getItems().setAll(sessions);
            if (activeSession != null) {
                sessionSelector.setValue(activeSession);
            }
            btnCloseSession.setVisible(true);
            btnCloseSession.setManaged(true);
        }
    }
    public void setOnCloseSessionAction(Runnable r)                      { this.onCloseSessionAction = r; }
    public void setOnAutoDetectSlotsAction(Runnable r)                   { this.onAutoDetectSlotsAction = r; }
    public void setOnLoadSlotsCsvAction(Runnable r)                      { this.onLoadSlotsCsvAction = r; }
    public void setOnSaveSlotsCsvAction(Runnable r)                      { this.onSaveSlotsCsvAction = r; }
    public void setOnClearSlotsAction(Runnable r)                        { this.onClearSlotsAction = r; }
    public void setOnAdaptiveThresholdChanged(Consumer<Boolean> c)       { this.onAdaptiveThresholdChanged = c; }
    public void setOnAnalysisModeChanged(Consumer<OccupancyDecisionEngine.AnalysisMode> c) { this.onAnalysisModeChanged = c; }

    /**
     * Synchronizes the displayed classifier mode without triggering an intermediate analysis.
     * Used after loading/training a compatible Decision Tree so the caller can update the service
     * and rerun the current photograph exactly once.
     */
    public void setAnalysisModeSelectionSilently(OccupancyDecisionEngine.AnalysisMode mode) {
        if (mode == null) return;
        suppressAnalysisModeCallback = true;
        try {
            analysisModeSelector.setValue(mode);
        } finally {
            suppressAnalysisModeCallback = false;
        }
    }

    public OccupancyDecisionEngine.AnalysisMode getSelectedAnalysisMode() {
        return analysisModeSelector.getValue();
    }

    public void setOnDetectionStrategyChanged(Consumer<DetectionStrategy> c) { this.onDetectionStrategyChanged = c; }

    private void setupCanvasDrawingHandlers() {
        overlayPane.setOnMousePressed(e -> {
            if (currentAwtImage == null || e.getButton() != MouseButton.PRIMARY) return;
            InteractionMode mode = modeSelector.getValue();
            if (mode == InteractionMode.SELECT_INSPECT) return;
            if (mode == InteractionMode.RECTIFY_4_CORNERS) {
                double cx = clamp(e.getX(), 0, currentAwtImage.getWidth() - 1);
                double cy = clamp(e.getY(), 0, currentAwtImage.getHeight() - 1);
                cornerPoints.add(new double[]{cx, cy});
                if (cornerPoints.size() == 4) {
                    double[][] quad = cornerPoints.toArray(new double[4][]);
                    cornerPoints.clear();
                    modeSelector.setValue(InteractionMode.SELECT_INSPECT);
                    if (onPerspectiveQuadListener != null) {
                        onPerspectiveQuadListener.accept(quad);
                    }
                } else {
                    renderOverlays();
                }
                e.consume();
                return;
            }

            dragStartX = clamp(e.getX(), 0, currentAwtImage.getWidth() - 1);
            dragStartY = clamp(e.getY(), 0, currentAwtImage.getHeight() - 1);
            dragPreviewRect.setX(dragStartX);
            dragPreviewRect.setY(dragStartY);
            dragPreviewRect.setWidth(1);
            dragPreviewRect.setHeight(1);
            dragPreviewRect.setVisible(true);
            if (!overlayPane.getChildren().contains(dragPreviewRect)) {
                overlayPane.getChildren().add(dragPreviewRect);
            }
            e.consume();
        });

        overlayPane.setOnMouseDragged(e -> {
            if (!dragPreviewRect.isVisible() || currentAwtImage == null) return;
            double curX = clamp(e.getX(), 0, currentAwtImage.getWidth());
            double curY = clamp(e.getY(), 0, currentAwtImage.getHeight());
            double x = Math.min(dragStartX, curX);
            double y = Math.min(dragStartY, curY);
            double w = Math.abs(curX - dragStartX);
            double h = Math.abs(curY - dragStartY);
            dragPreviewRect.setX(x);
            dragPreviewRect.setY(y);
            dragPreviewRect.setWidth(w);
            dragPreviewRect.setHeight(h);
            e.consume();
        });

        overlayPane.setOnMouseReleased(e -> {
            if (!dragPreviewRect.isVisible() || currentAwtImage == null) return;
            dragPreviewRect.setVisible(false);

            int rx = (int) Math.round(dragPreviewRect.getX());
            int ry = (int) Math.round(dragPreviewRect.getY());
            int rw = (int) Math.round(dragPreviewRect.getWidth());
            int rh = (int) Math.round(dragPreviewRect.getHeight());

            int imgW = currentAwtImage.getWidth();
            int imgH = currentAwtImage.getHeight();
            rx = Math.max(0, Math.min(imgW - 10, rx));
            ry = Math.max(0, Math.min(imgH - 10, ry));
            rw = Math.min(rw, imgW - rx);
            rh = Math.min(rh, imgH - ry);

            if (rw >= 12 && rh >= 12 && onBoxDrawnListener != null) {
                int bays = (modeSelector.getValue() == InteractionMode.DRAW_ROW_GRID)
                    ? bayCountSpinner.getValue()
                    : 1;
                onBoxDrawnListener.accept(new DrawnBox(rx, ry, rw, rh, bays));
            }
            e.consume();
        });
    }

    private static double clamp(double val, double min, double max) {
        return Math.max(min, Math.min(max, val));
    }

    public void setZoom(double z) {
        double clamped = Math.max(0.4, Math.min(3.5, z));
        zoomLevel.set(clamped);
    }

    public double getZoom() {
        return zoomLevel.get();
    }

    public void setFillOpacityFraction(double opacity) {
        if (!Double.isFinite(opacity) || opacity < 0.0 || opacity > 0.50) {
            throw new IllegalArgumentException("Fill opacity must be in [0.0, 0.50]; got " + opacity);
        }
        this.fillOpacityFraction = opacity;
        if (onOpacityChangedListener != null) {
            onOpacityChangedListener.accept(opacity);
        }
        renderOverlays();
    }

    public double getFillOpacityFraction() {
        return fillOpacityFraction;
    }

    public void setOnOpacityChanged(DoubleConsumer listener) {
        this.onOpacityChangedListener = listener;
    }

    public void setShowSlotBoxes(boolean show)          { showBoxesCheck.setSelected(show); }
    public boolean isShowSlotBoxes()                    { return showBoxesCheck.isSelected(); }
    public void setShowVehicleDetections(boolean show)  { showDetectionsCheck.setSelected(show); }
    public boolean isShowVehicleDetections()            { return showDetectionsCheck.isSelected(); }
    public void setShowSlotIds(boolean show)            { showBadgesCheck.setSelected(show); }
    public boolean isShowSlotIds()                      { return showBadgesCheck.isSelected(); }
    public void setShowFeatureOverlays(boolean show)    { showFeatureOverlaysCheck.setSelected(show); }
    public boolean isShowFeatureOverlays()              { return showFeatureOverlaysCheck.isSelected(); }
    public void setRawImageOnly(boolean rawOnly)        { rawImageOnlyCheck.setSelected(rawOnly); }
    public boolean isRawImageOnly()                     { return rawImageOnlyCheck.isSelected(); }
    public Pane getOverlayPane()                        { return overlayPane; }

    public void autoFitView() {
        if (currentAwtImage == null) return;
        double viewportW = scrollPane.getViewportBounds().getWidth();
        double viewportH = scrollPane.getViewportBounds().getHeight();
        if (viewportW > 50 && viewportH > 50) {
            double scaleX = (viewportW - 40) / currentAwtImage.getWidth();
            double scaleY = (viewportH - 40) / currentAwtImage.getHeight();
            double bestScale = Math.min(scaleX, scaleY);
            setZoom(Math.max(0.5, Math.min(2.2, bestScale)));
        } else {
            setZoom(1.2);
        }
    }

    public void setImageAndSlots(BufferedImage awtImage, List<ParkingSlot> slots) {
        this.originalAwtImage = awtImage;
        this.enhancedAwtImage = awtImage != null ? ImageEnhancer.autoEnhance(awtImage) : null;
        this.enhancedPreviewActive = awtImage != null;
        this.currentAwtImage = enhancedPreviewActive ? enhancedAwtImage : originalAwtImage;
        this.currentAnnotatedAwtImage = null;
        this.currentSlots = slots;
        this.currentVehicleDetections = new ArrayList<>();
        this.analysisMap.clear();
        this.selectedSlot = null;
        this.cornerPoints.clear();

        BufferedImage display = currentAwtImage;
        Image fxImage = ImageFxUtils.toFxImage(display);
        imageView.setImage(fxImage);
        if (fxImage != null) {
            imageView.setFitWidth(fxImage.getWidth());
            imageView.setFitHeight(fxImage.getHeight());
        }
        btnToggleEnhancement.setDisable(awtImage == null);
        btnViewIvpStages.setDisable(awtImage == null);
        btnEnhanceImage.setDisable(awtImage == null);
        refreshEnhancementStatus();
        if (awtImage != null && (awtImage.getWidth() > 550 || awtImage.getHeight() > 500)) {
            setZoom(0.95);
        } else {
            setZoom(1.25);
        }
        renderOverlays();
    }

    /** Show either the automatic enhanced preview or the untouched original without changing source data. */
    public void setEnhancedPreviewActive(boolean active) {
        if (originalAwtImage == null) return;
        this.enhancedPreviewActive = active && enhancedAwtImage != null;
        this.currentAwtImage = enhancedPreviewActive ? enhancedAwtImage : originalAwtImage;
        Image fxImage = ImageFxUtils.toFxImage(currentAwtImage);
        imageView.setImage(fxImage);
        if (fxImage != null) {
            imageView.setFitWidth(fxImage.getWidth());
            imageView.setFitHeight(fxImage.getHeight());
        }
        refreshEnhancementStatus();
        renderOverlays();
    }

    private void refreshEnhancementStatus() {
        if (originalAwtImage == null) {
            lblEnhancementPreview.setText("Auto enhancement will be applied when an image is loaded.");
            btnToggleEnhancement.setText("Show Original");
            return;
        }
        if (enhancedPreviewActive) {
            lblEnhancementPreview.setText("Enhanced preview ON • Canonical policy ivp-enhance-v1 active • Original retained for vehicle detection; IVP features computed from enhanced pixels.");
            btnToggleEnhancement.setText("Show Original");
        } else {
            lblEnhancementPreview.setText("Original preview ON • Displaying raw photograph • Vehicle detection uses natural pixels; IVP features computed from enhanced pixels.");
            btnToggleEnhancement.setText("Show Enhanced");
        }
    }

    public boolean isEnhancedPreviewActive() { return enhancedPreviewActive; }
    public BufferedImage getOriginalPreviewImage() { return originalAwtImage; }
    public BufferedImage getEnhancedPreviewImage() { return enhancedAwtImage; }

    public void updateAnalysisResults(BufferedImage annotatedImage, List<SlotAnalysisResult> results) {
        updateAnalysisResults(annotatedImage, results, List.of());
    }

    public void updateAnalysisResults(BufferedImage annotatedImage,
                                      List<SlotAnalysisResult> results,
                                      List<VehicleDetection> vehicleDetections) {
        this.currentAnnotatedAwtImage = annotatedImage;
        this.currentVehicleDetections = vehicleDetections != null
            ? new ArrayList<>(vehicleDetections)
            : new ArrayList<>();
        this.analysisMap.clear();
        for (SlotAnalysisResult res : results) {
            this.analysisMap.put(res.getId(), res);
        }
        renderOverlays();
    }

    public void setSelectedSlot(ParkingSlot slot) {
        this.selectedSlot = slot;
        renderOverlays();
    }

    public ParkingSlot getSelectedSlot() {
        return selectedSlot;
    }

    public void setOnSlotSelected(Consumer<ParkingSlot> listener)           { this.slotSelectionListener = listener; }
    public void setOnBoxDrawn(Consumer<DrawnBox> listener)                  { this.onBoxDrawnListener = listener; }
    public void setOnPerspectiveQuadSelected(Consumer<double[][]> listener) { this.onPerspectiveQuadListener = listener; }
    public void setOnSlotDeleted(IntConsumer listener)                      { this.onSlotDeletedListener = listener; }

    public void renderOverlays() {
        overlayPane.getChildren().clear();
        if (currentAwtImage == null) return;
        if (rawImageOnlyCheck.isSelected()) {
            return;
        }

        boolean showBoxes = showBoxesCheck.isSelected();
        boolean showDetections = showDetectionsCheck.isSelected();
        boolean showBadges = showBadgesCheck.isSelected();
        boolean showFeatureFill = showFeatureOverlaysCheck.isSelected();
        double effectiveFill = showFeatureFill ? fillOpacityFraction : 0.0;

        if (currentSlots != null && (showBoxes || showBadges)) {
            for (ParkingSlot slot : currentSlots) {
                SlotAnalysisResult analysis = analysisMap.get(slot.getId());
                SlotStatus status = slot.getStatus() != null
                    ? slot.getStatus()
                    : (analysis != null ? analysis.getStatus() : null);
                boolean isAnalyzed = analysis != null || status != null;

                Color strokeColor;
                Color fillColor;
                String statusText;
                String shortTag;
                List<Double> strokeDash;

                if (status == SlotStatus.OCCUPIED) {
                    strokeColor = Color.rgb(239, 68, 68, 0.96);
                    fillColor   = Color.rgb(239, 68, 68, effectiveFill);
                    statusText  = "OCCUPIED";
                    shortTag    = "OCC";
                    strokeDash  = List.of();
                } else if (status == SlotStatus.AVAILABLE) {
                    strokeColor = Color.rgb(16, 185, 129, 0.96);
                    fillColor   = Color.rgb(16, 185, 129, effectiveFill);
                    statusText  = "AVAILABLE";
                    shortTag    = "AVAIL";
                    strokeDash  = List.of();
                } else if (status == SlotStatus.NEEDS_REVIEW) {
                    strokeColor = Color.rgb(245, 158, 11, 0.98);
                    fillColor   = Color.rgb(245, 158, 11, showFeatureFill ? Math.min(0.20, effectiveFill + 0.02) : 0.0);
                    statusText  = "NEEDS REVIEW";
                    shortTag    = "REV";
                    strokeDash  = List.of(7.0, 4.0);
                } else {
                    strokeColor = Color.rgb(148, 163, 184, 0.88);
                    fillColor   = Color.rgb(148, 163, 184, showFeatureFill ? Math.min(0.08, effectiveFill) : 0.0);
                    statusText  = isAnalyzed ? "UNCLASSIFIED" : "UNANALYZED";
                    shortTag    = "SLOT";
                    strokeDash  = List.of(3.0, 3.0);
                }

                boolean isSelected = selectedSlot != null && selectedSlot.getId() == slot.getId();
                if (isSelected) {
                    fillColor = Color.rgb(56, 189, 248, Math.min(SELECTED_FILL_OPACITY, effectiveFill));
                }

                Shape slotShape;
                if (slot.hasCustomPolygon()) {
                    Polygon poly = new Polygon();
                    for (double[] pt : slot.getPolygonVertices()) {
                        poly.getPoints().addAll(pt[0], pt[1]);
                    }
                    slotShape = poly;
                } else {
                    Rectangle rect = new Rectangle(slot.getX(), slot.getY(), slot.getWidth(), slot.getHeight());
                    rect.setArcWidth(4);
                    rect.setArcHeight(4);
                    slotShape = rect;
                }

                slotShape.setFill(fillColor);
                slotShape.setStroke(strokeColor);
                slotShape.setStrokeWidth(isSelected ? 2.8 : 2.0);
                if (!strokeDash.isEmpty()) {
                    slotShape.getStrokeDashArray().addAll(strokeDash);
                }
                slotShape.setVisible(showBoxes);
                slotShape.setCursor(Cursor.HAND);
                slotShape.setOnMouseClicked(e -> {
                    if (e.getButton() == MouseButton.SECONDARY) {
                        if (onSlotDeletedListener != null) {
                            onSlotDeletedListener.accept(slot.getId());
                        }
                        e.consume();
                        return;
                    }
                    if (modeSelector.getValue() == InteractionMode.SELECT_INSPECT) {
                        this.selectedSlot = slot;
                        renderOverlays();
                        if (slotSelectionListener != null) {
                            slotSelectionListener.accept(slot);
                        }
                        e.consume();
                    }
                });

                String tooltipText = String.format(Locale.ROOT,
                    "Slot #%d: %s\nBounds: (%d, %d, %d×%d px) | Polygon: %s | Origin: %s",
                    slot.getId(), statusText, slot.getX(), slot.getY(), slot.getWidth(), slot.getHeight(),
                    slot.hasCustomPolygon() ? slot.formatPolygonCoordinates() : "Axis-Aligned",
                    slot.getOrigin().getDisplayName());
                Tooltip.install(slotShape, new Tooltip(tooltipText));
                overlayPane.getChildren().add(slotShape);

                if (isSelected && showBoxes) {
                    Rectangle selRing = new Rectangle(
                        slot.getX() - 2.5, slot.getY() - 2.5,
                        slot.getWidth() + 5.0, slot.getHeight() + 5.0
                    );
                    selRing.setFill(Color.rgb(0, 0, 0, 0.0));
                    selRing.setStroke(Color.rgb(56, 189, 248, 1.0));
                    selRing.setStrokeWidth(2.5);
                    selRing.setArcWidth(6);
                    selRing.setArcHeight(6);
                    selRing.setMouseTransparent(true);
                    overlayPane.getChildren().add(selRing);
                }

                if (showBadges) {
                    String badgeLabel = showFeatureFill
                        ? String.format("#%d %s", slot.getId(), shortTag)
                        : String.format("#%d", slot.getId());
                    Label badge = new Label(badgeLabel.trim());
                    badge.setLayoutX(slot.getX() + 4);
                    badge.setLayoutY(Math.max(slot.getY() + 3, slot.getY() + slot.getHeight() - 19));
                    String borderHex = isSelected ? "#38BDF8" : toHex(strokeColor);
                    badge.setStyle(String.format(
                        "-fx-background-color: #0b1120dc; -fx-text-fill: #f8fafc; -fx-font-weight: bold; " +
                        "-fx-font-size: 10px; -fx-padding: 1px 4px; -fx-background-radius: 3px; " +
                        "-fx-border-color: %s; -fx-border-radius: 3px; -fx-border-width: 1px;",
                        borderHex
                    ));
                    badge.setCursor(Cursor.HAND);
                    badge.setOnMouseClicked(slotShape.getOnMouseClicked());
                    overlayPane.getChildren().add(badge);
                }
            }
        }

        if (showDetections && currentVehicleDetections != null && !currentVehicleDetections.isEmpty()) {
            for (VehicleDetection det : currentVehicleDetections) {
                Rectangle detRect = new Rectangle(det.x(), det.y(), det.width(), det.height());
                detRect.setFill(Color.rgb(56, 189, 248, 0.05));
                detRect.setStroke(Color.rgb(56, 189, 248, 0.96));
                detRect.setStrokeWidth(1.9);
                detRect.getStrokeDashArray().addAll(5.0, 3.0);

                String detTooltip = String.format(Locale.ROOT,
                    "Vehicle Detection %s (%s, %.0f%%)\nBox: (%.1f, %.1f, %.1f×%.1f px)",
                    det.detectionId(), det.category().getDisplayName(), det.detectionScore() * 100.0,
                    det.x(), det.y(), det.width(), det.height());
                Tooltip.install(detRect, new Tooltip(detTooltip));

                Label detLbl = new Label(String.format(Locale.ROOT, "%s %.0f%%",
                    det.detectionId(), det.detectionScore() * 100.0));
                detLbl.setLayoutX(det.x() + 2);
                detLbl.setLayoutY(Math.max(2, det.y() + 2));
                detLbl.setStyle(
                    "-fx-background-color: #0284c7dd; -fx-text-fill: #ffffff; -fx-font-weight: bold; " +
                    "-fx-font-size: 9.5px; -fx-padding: 1px 4px; -fx-background-radius: 3px;"
                );
                detLbl.setMouseTransparent(true);
                overlayPane.getChildren().addAll(detRect, detLbl);
            }
        }

        for (int i = 0; i < cornerPoints.size(); i++) {
            double[] pt = cornerPoints.get(i);
            Circle dot = new Circle(pt[0], pt[1], 5.5, Color.rgb(56, 189, 248, 0.95));
            dot.setStroke(Color.WHITE);
            dot.setStrokeWidth(1.5);
            dot.setMouseTransparent(true);

            Label cornerTag = new Label("C" + (i + 1));
            cornerTag.setLayoutX(pt[0] + 7);
            cornerTag.setLayoutY(pt[1] - 9);
            cornerTag.setStyle(
                "-fx-background-color: #0284c7; -fx-text-fill: white; -fx-font-weight: bold; " +
                "-fx-font-size: 10px; -fx-padding: 1px 4px; -fx-background-radius: 3px;"
            );
            cornerTag.setMouseTransparent(true);
            overlayPane.getChildren().addAll(dot, cornerTag);
        }
    }

    private static String toHex(Color c) {
        return String.format("#%02X%02X%02X",
            (int) (c.getRed() * 255),
            (int) (c.getGreen() * 255),
            (int) (c.getBlue() * 255));
    }
}
