package com.parknchillai.ui;

import com.parknchillai.app.AppState;
import com.parknchillai.app.SlotAnalysisResult;
import com.parknchillai.app.SlotDiagnosticRecord;
import com.parknchillai.ai.DecisionTreeModel;
import com.parknchillai.detection.VehicleDetectionService;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.collections.transformation.FilteredList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * Streamlined Parking-Slot Review Table (ParkNChill v2.3 Requirement 2.D, 3 &amp; 4):
 * <ul>
 *   <li>Fits standard desktop resolutions ({@code 1366 × 768} and {@code 1600 × 900}) without
 *       horizontal scrolling by showing only the 5 essential columns in the normal view:
 *       {@code Slot ID}, {@code Current Status}, {@code Verified Label}, {@code Review Status},
 *       and {@code Detector Suggestion}.</li>
 *   <li>When no slot is selected, displays a single unobtrusive instruction line rather than a
 *       large empty details panel.</li>
 *   <li>When a slot is selected, highlights its boundary in {@link ParkingView}, shows its
 *       suggestion and verification state, provides {@code Mark Occupied}, {@code Mark Available},
 *       and {@code Needs Review} actions, reveals boundary-editing controls only on demand, and
 *       requires explicit label confirmation after any boundary geometry modification before the
 *       record can enter training.</li>
 *   <li>Moves detailed vehicle-overlap measurements, 10-feature values, Decision Tree rule paths,
 *       provenance, and the 8-stage IVP gallery into a collapsed-by-default advanced section.</li>
 * </ul>
 */
public class SlotTableView extends VBox {

    public static final String CELL_TEXT_HEX       = "#f8fafc";
    public static final String EVEN_ROW_BG_HEX     = "#0f172a";
    public static final String ODD_ROW_BG_HEX      = "#162032";
    public static final String SELECTED_ROW_BG_HEX = "#1e3a8a";
    public static final String HEADER_TEXT_HEX     = "#f1f5f9";
    public static final String HEADER_BG_HEX       = "#0b1120";

    public static final String OCCUPIED_TEXT_HEX   = "#fecaca";
    public static final String OCCUPIED_BG_HEX     = "#7f1d1d";
    public static final String AVAILABLE_TEXT_HEX  = "#a7f3d0";
    public static final String AVAILABLE_BG_HEX    = "#064e3b";
    public static final String REVIEW_TEXT_HEX     = "#fde68a";
    public static final String REVIEW_BG_HEX       = "#78350f";

    public static final String BASE_CELL_STYLE =
        "-fx-text-fill: #f8fafc; -fx-font-size: 13.5px; -fx-padding: 6px 8px;";
    public static final String NUMERIC_CELL_STYLE =
        "-fx-text-fill: #f8fafc; -fx-font-family: 'Consolas', 'Courier New', monospace; -fx-font-size: 13.5px; -fx-font-weight: bold; -fx-padding: 6px 8px;";

    private final TableView<SlotAnalysisResult> table = new TableView<>();
    private final ObservableList<SlotAnalysisResult> masterData = FXCollections.observableArrayList();
    private final FilteredList<SlotAnalysisResult> filteredData = new FilteredList<>(masterData, p -> true);

    private AppState appState;
    private PipelineInspectorView embeddedInspector;

    // Predictions from an explicitly applied saved tree are kept separate from the analysis
    // result so they never replace detector-based current statuses or overlays.
    private final Map<Integer, DecisionTreeModel.DecisionTrace> separatelyAppliedTreeTraces = new LinkedHashMap<>();
    private String separatelyAppliedTreeModelLabel = "";

    private final TextField searchField = new TextField();
    private final ComboBox<String> statusFilter = new ComboBox<>();

    // Unobtrusive 1-line prompt when no slot is selected (Requirement 2.D)
    private final Label lblUnselectedPrompt = new Label(
        "Select any row in the table above (or click a parking space in the image) to confirm its occupancy label or inspect details."
    );

    // Selected-slot action card (hidden when no row is selected)
    private final VBox selectedSlotActionBox = new VBox(8);
    private final Label lblSelectedDetailHeader = new Label("Selected Slot Review");
    private final Label lblGeometryWarningBanner = new Label("");
    private final Button btnMarkOccupied = new Button("Mark Occupied");
    private final Button btnMarkAvailable = new Button("Mark Available");
    private final Button btnMarkNeedsReview = new Button("Needs Review");
    private final Button btnToggleBoundaryEdit = new Button("Edit Slot Boundary...");

    // Collapsed-by-default Advanced Slot Details & Diagnostics section (Requirement 2.D & 4)
    private final TitledPane advancedSlotDetailsPane = new TitledPane();
    private final Label lblSelectedDetailBody = new Label("");
    private final VBox advancedContentBox = new VBox(8);

    private Consumer<ParkingSlot> selectionCallback;
    private BiConsumer<SlotAnalysisResult, SlotStatus> onSlotReviewedCallback;
    private Runnable onRequestBoundaryEditCallback;

    public SlotTableView() {
        setSpacing(8);
        setPadding(new Insets(10));
        getStyleClass().add("summary-card-container");
        VBox.setVgrow(this, Priority.ALWAYS);

        HBox header = createHeaderBar();
        configureEssentialFiveColumns();

        // Constrained resize policy so the 5 essential columns fit 1366x768 & 1600x900 without horizontal overflow
        table.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY_FLEX_LAST_COLUMN);
        table.setItems(filteredData);
        Label placeholder = new Label("Load an image and click 'Detect and Analyse' to populate parking slots.");
        placeholder.setStyle("-fx-text-fill: #cbd5e1; -fx-font-size: 13.5px;");
        table.setPlaceholder(placeholder);
        VBox.setVgrow(table, Priority.ALWAYS);

        lblUnselectedPrompt.setWrapText(true);
        lblUnselectedPrompt.setStyle(
            "-fx-text-fill: #cbd5e1; -fx-font-size: 12.5px; -fx-padding: 6px 10px; " +
            "-fx-background-color: #0f172a; -fx-background-radius: 5px;"
        );

        buildSelectedSlotReviewAndAdvancedPanes();

        table.getSelectionModel().selectedItemProperty().addListener((obs, oldVal, newVal) -> {
            updateSelectedRowDetailPanel(newVal);
            if (newVal != null && selectionCallback != null) {
                selectionCallback.accept(newVal.getSlot());
            }
        });

        getChildren().addAll(
            header,
            table,
            lblUnselectedPrompt,
            selectedSlotActionBox,
            advancedSlotDetailsPane
        );

        updateSelectedRowDetailPanel(null);
    }

    public void setAppState(AppState state) {
        this.appState = state;
        table.refresh();
    }

    public void setEmbeddedInspector(PipelineInspectorView inspector) {
        this.embeddedInspector = inspector;
        if (inspector != null && !advancedContentBox.getChildren().contains(inspector)) {
            advancedContentBox.getChildren().add(inspector);
        }
    }

    private HBox createHeaderBar() {
        HBox bar = new HBox(10);
        bar.setAlignment(Pos.CENTER_LEFT);

        Label title = new Label("Parking-Slot Review Table");
        title.getStyleClass().add("card-section-title");
        HBox.setHgrow(title, Priority.ALWAYS);

        searchField.setPromptText("Filter Slot #...");
        searchField.setPrefWidth(120);
        searchField.setStyle("-fx-text-fill: #f8fafc; -fx-control-inner-background: #0f172a; -fx-font-size: 12.5px;");
        searchField.textProperty().addListener((obs, oldVal, newVal) -> applyFilters());

        statusFilter.getItems().addAll("All Statuses", "OCCUPIED", "AVAILABLE", "NEEDS_REVIEW");
        statusFilter.setValue("All Statuses");
        statusFilter.setPrefWidth(145);
        statusFilter.setOnAction(e -> applyFilters());

        Button btnExportCsv = new Button("Export CSV");
        btnExportCsv.getStyleClass().add("btn-secondary");
        btnExportCsv.setOnAction(e -> exportToCsv());

        bar.getChildren().addAll(title, searchField, statusFilter, btnExportCsv);
        return bar;
    }

    private void buildSelectedSlotReviewAndAdvancedPanes() {
        selectedSlotActionBox.setPadding(new Insets(10, 12, 10, 12));
        selectedSlotActionBox.setStyle(
            "-fx-background-color: #0f172a; -fx-border-color: #0284c7; " +
            "-fx-border-radius: 6px; -fx-background-radius: 6px;"
        );

        lblSelectedDetailHeader.setWrapText(true);
        lblSelectedDetailHeader.setStyle("-fx-text-fill: #f8fafc; -fx-font-weight: bold; -fx-font-size: 13px;");

        lblGeometryWarningBanner.setWrapText(true);
        lblGeometryWarningBanner.setStyle(
            "-fx-background-color: #78350f; -fx-text-fill: #fde68a; -fx-font-weight: bold; " +
            "-fx-font-size: 12px; -fx-padding: 6px 10px; -fx-background-radius: 4px;"
        );
        lblGeometryWarningBanner.setVisible(false);
        lblGeometryWarningBanner.setManaged(false);

        HBox actionButtonsBar = new HBox(10);
        actionButtonsBar.setAlignment(Pos.CENTER_LEFT);

        btnMarkOccupied.getStyleClass().add("btn-primary");
        btnMarkOccupied.setStyle("-fx-background-color: #dc2626; -fx-text-fill: #ffffff; -fx-font-weight: bold;");
        btnMarkOccupied.setOnAction(e -> triggerSlotReview(SlotStatus.OCCUPIED));

        btnMarkAvailable.getStyleClass().add("btn-success");
        btnMarkAvailable.setOnAction(e -> triggerSlotReview(SlotStatus.AVAILABLE));

        btnMarkNeedsReview.getStyleClass().add("btn-secondary");
        btnMarkNeedsReview.setOnAction(e -> triggerSlotReview(SlotStatus.NEEDS_REVIEW));

        HBox spacer = new HBox();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        btnToggleBoundaryEdit.getStyleClass().add("btn-tool");
        btnToggleBoundaryEdit.setTooltip(new Tooltip("Open boundary-editing tools only when a slot box needs manual adjustment"));
        btnToggleBoundaryEdit.setOnAction(e -> {
            if (onRequestBoundaryEditCallback != null) {
                onRequestBoundaryEditCallback.run();
            }
        });

        actionButtonsBar.getChildren().addAll(
            btnMarkOccupied, btnMarkAvailable, btnMarkNeedsReview, spacer, btnToggleBoundaryEdit
        );

        selectedSlotActionBox.getChildren().addAll(
            lblSelectedDetailHeader,
            lblGeometryWarningBanner,
            actionButtonsBar
        );

        lblSelectedDetailBody.setWrapText(true);
        lblSelectedDetailBody.setStyle(
            "-fx-text-fill: #f8fafc; -fx-font-family: 'Consolas', 'Courier New', monospace; -fx-font-size: 12px;"
        );

        advancedContentBox.setPadding(new Insets(8));
        advancedContentBox.setStyle("-fx-background-color: #0f172a;");
        advancedContentBox.getChildren().add(lblSelectedDetailBody);

        advancedSlotDetailsPane.setText("Advanced Slot Details (Vehicle Overlap, 10 Feature Values, CART Rule Path & Provenance)");
        advancedSlotDetailsPane.setContent(advancedContentBox);
        advancedSlotDetailsPane.setCollapsible(true);
        advancedSlotDetailsPane.setExpanded(false);
    }

    private void triggerSlotReview(SlotStatus targetStatus) {
        SlotAnalysisResult selected = table.getSelectionModel().getSelectedItem();
        if (selected != null && onSlotReviewedCallback != null) {
            onSlotReviewedCallback.accept(selected, targetStatus);
            table.refresh();
            updateSelectedRowDetailPanel(selected);
        }
    }

    /**
     * Updates the selected-slot action card and collapsed diagnostics section.
     * When {@code item == null}, hides the details panels and shows only the unobtrusive 1-line instruction
     * (Requirement 2.D).
     */
    public void updateSelectedRowDetailPanel(SlotAnalysisResult item) {
        if (item == null) {
            lblUnselectedPrompt.setVisible(true);
            lblUnselectedPrompt.setManaged(true);
            selectedSlotActionBox.setVisible(false);
            selectedSlotActionBox.setManaged(false);
            advancedSlotDetailsPane.setVisible(false);
            advancedSlotDetailsPane.setManaged(false);
            lblSelectedDetailHeader.setText("No slot selected.");
            lblSelectedDetailBody.setText("Select any row to inspect full details.");
            return;
        }

        lblUnselectedPrompt.setVisible(false);
        lblUnselectedPrompt.setManaged(false);
        selectedSlotActionBox.setVisible(true);
        selectedSlotActionBox.setManaged(true);
        advancedSlotDetailsPane.setVisible(true);
        advancedSlotDetailsPane.setManaged(true);

        int slotId = item.getId();
        SlotStatus currentStatus = item.getSlot().getStatus() != null ? item.getSlot().getStatus() : item.getStatus();
        SlotStatus verifiedLabel = appState != null ? appState.getVerifiedLabelForSlot(slotId) : null;
        String reviewState = appState != null ? appState.getReviewStatusForSlot(slotId) : "Provisional (Awaiting Review)";
        boolean geometryUnconfirmed = appState != null && appState.isSlotGeometryUnconfirmed(slotId);

        String detectorSuggestion = formatDetectorSuggestion(item);

        String originTag = item.getSlot() != null ? item.getSlot().getOrigin().getDisplayName() : "Unknown";
        lblSelectedDetailHeader.setText(String.format(Locale.ROOT,
            "Slot #%d — Status: %s | Origin: %s | Verified Label: %s | Review State: %s | Detector Suggestion: %s",
            slotId,
            currentStatus,
            originTag,
            verifiedLabel != null ? verifiedLabel.name() : "Unverified",
            reviewState,
            detectorSuggestion
        ));

        if (geometryUnconfirmed) {
            lblGeometryWarningBanner.setText(
                "Geometry Modified: Explicit label confirmation ('Mark Occupied' or 'Mark Available') is required after a boundary change before Slot #"
                + slotId + " can enter Decision Tree training."
            );
            lblGeometryWarningBanner.setVisible(true);
            lblGeometryWarningBanner.setManaged(true);
        } else {
            lblGeometryWarningBanner.setVisible(false);
            lblGeometryWarningBanner.setManaged(false);
        }

        SlotDiagnosticRecord diag = item.getDiagnosticRecord();
        String assignedVehId = (diag != null && diag.overlapEvidence() != null)
            ? diag.overlapEvidence().assignedVehicleId()
            : "NONE";
        String treePredStr = diag != null && diag.treePrediction() != null
            ? diag.treePrediction().name()
            : item.getDecisionTrace().finalStatus().name();
        String vehLine = diag != null
            ? ("Vehicle Detections: " + diag.formatVehicleDetectionsSummary() + " | Overlap: " + diag.formatOverlapEvidenceSummary())
            : "Vehicle Detections: UNAVAILABLE";
        String detFeatLine = String.format(Locale.ROOT,
            "10-Feature Schema (%s): meanInt=%.1f, fg=%.3f, dark=%.3f, edge=%.3f, area=%.3f, sat=%.1f, satR=%.3f, blob=%.3f, vehicleOverlapRatio=%.4f, vehicleMatchedFlag=%.1f",
            item.getFeatures().getSchemaVersion(),
            item.getFeatures().getMeanIntensity(),
            item.getFeatures().getForegroundRatio(),
            item.getFeatures().getDarkPixelRatio(),
            item.getFeatures().getEdgeDensity(),
            item.getFeatures().getObjectArea(),
            item.getFeatures().getMeanSaturation(),
            item.getFeatures().getSaturationRatio(),
            item.getFeatures().getLargestBlobRatio(),
            item.getFeatures().getVehicleOverlapRatio(),
            item.getFeatures().getVehicleMatchedFlag()
        );
        DecisionTreeModel.DecisionTrace appliedTrace = separatelyAppliedTreeTraces.get(slotId);
        String separateTreeLine;
        if (appliedTrace == null) {
            separateTreeLine = "SEPARATE LOADED TREE RESULT: NOT RUN. Use 'Run Tree Separately' to calculate a side-by-side result without changing Current Status.";
        } else {
            separateTreeLine = String.format(Locale.ROOT,
                "SEPARATE LOADED TREE RESULT (%s): %s | Purity %.1f%% | Path: %s",
                separatelyAppliedTreeModelLabel, appliedTrace.finalStatus(), appliedTrace.leaf().leafPurity() * 100.0,
                appliedTrace.compactPath());
        }

        lblSelectedDetailBody.setText(String.format(Locale.ROOT,
            "Last analysis result : %s | Tree trace from that analysis: %s | Vehicle: %s | Crop: (%d,%d %dx%d px)\n" +
            "Separate loaded-tree prediction: %s\n" +
            "Last-analysis rule path   : %s\n" +
            "Last-analysis leaf support: Purity %.1f%% (occ=%d, avail=%d, n=%d)\n" +
            "%s\n%s",
            currentStatus, treePredStr, assignedVehId,
            item.getSlot().getX(), item.getSlot().getY(), item.getSlot().getWidth(), item.getSlot().getHeight(),
            separateTreeLine,
            item.getDecisionPath(),
            item.getLeafPurity() * 100.0,
            item.getDecisionTrace().leaf().occupiedCount(),
            item.getDecisionTrace().leaf().availableCount(),
            item.getDecisionTrace().leaf().totalSamples(),
            vehLine,
            detFeatLine
        ));

        if (embeddedInspector != null) {
            embeddedInspector.updateSlot(item);
        }
    }

    private String formatDetectorSuggestion(SlotAnalysisResult r) {
        if (r == null) return "—";
        SlotDiagnosticRecord d = r.getDiagnosticRecord();
        if (d == null) {
            return r.getStatus().name();
        }
        if (d.detectorStatus() != VehicleDetectionService.DetectorStatus.READY) {
            return "NEEDS_REVIEW (" + d.detectorStatus().name() + ")";
        }
        SlotStatus detPred = d.vehicleDetectorPrediction() != null ? d.vehicleDetectorPrediction() : r.getStatus();
        String vid = (d.overlapEvidence() != null) ? d.overlapEvidence().assignedVehicleId() : "NONE";
        double ios = (d.overlapEvidence() != null) ? d.overlapEvidence().bestIoS() : 0.0;
        if (!"NONE".equals(vid)) {
            return String.format(Locale.ROOT, "%s (%s, IoS=%.2f)", detPred.name(), vid, ios);
        }
        return String.format(Locale.ROOT, "%s (IoS=%.2f)", detPred.name(), ios);
    }

    private void applyFilters() {
        String query = searchField.getText() == null ? "" : searchField.getText().trim().toLowerCase();
        String status = statusFilter.getValue();

        filteredData.setPredicate(item -> {
            SlotStatus effectiveStatus = item.getSlot().getStatus() != null ? item.getSlot().getStatus() : item.getStatus();
            boolean matchesQuery = query.isEmpty() || String.valueOf(item.getId()).contains(query);
            boolean matchesStatus = "All Statuses".equals(status) || effectiveStatus.name().equals(status);
            return matchesQuery && matchesStatus;
        });
    }

    /**
     * Requirement 2.D: Configures only the 5 essential columns in the normal view so that
     * desktop sizes (1366×768 and 1600×900) display all columns clearly without horizontal scrolling:
     * 1. Slot ID
     * 2. Current Status
     * 3. Verified Label
     * 4. Review Status
     * 5. Detector Suggestion
     */
    @SuppressWarnings("unchecked")
    private void configureEssentialFiveColumns() {
        TableColumn<SlotAnalysisResult, Number> colId = new TableColumn<>("Slot ID");
        colId.setCellValueFactory(c -> new SimpleIntegerProperty(c.getValue().getId()));
        colId.setCellFactory(col -> new TableCell<>() {
            @Override
            protected void updateItem(Number item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                    setTooltip(null);
                    setStyle("");
                } else {
                    String txt = "#" + item.intValue();
                    setText(txt);
                    setStyle(NUMERIC_CELL_STYLE);
                    setTooltip(new Tooltip("Parking Space " + txt));
                }
            }
        });
        colId.setMinWidth(65);
        colId.setPrefWidth(75);

        TableColumn<SlotAnalysisResult, String> colCurrentStatus = new TableColumn<>("Last Analysis Result");
        colCurrentStatus.setCellValueFactory(c -> {
            SlotAnalysisResult r = c.getValue();
            SlotStatus st = r.getSlot().getStatus() != null ? r.getSlot().getStatus() : r.getStatus();
            return new SimpleStringProperty(st.name());
        });
        colCurrentStatus.setCellFactory(col -> new TableCell<>() {
            @Override
            protected void updateItem(String item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                    setGraphic(null);
                    setTooltip(null);
                    setStyle("");
                } else {
                    setText(item);
                    setStyle(styleForStatus(item));
                    setTooltip(new Tooltip("Result from the most recent Detect and Analyse run: " + item + ". Loading or separately applying a saved tree does not replace this result."));
                }
            }
        });
        colCurrentStatus.setMinWidth(110);
        colCurrentStatus.setPrefWidth(125);

        TableColumn<SlotAnalysisResult, String> colVerifiedLabel = new TableColumn<>("Verified Label");
        colVerifiedLabel.setCellValueFactory(c -> {
            int id = c.getValue().getId();
            SlotStatus ver = appState != null ? appState.getVerifiedLabelForSlot(id) : null;
            return new SimpleStringProperty(ver != null ? ver.name() : "Unverified");
        });
        colVerifiedLabel.setCellFactory(col -> formatStringCell("Human-confirmed occupancy label for training"));
        colVerifiedLabel.setMinWidth(110);
        colVerifiedLabel.setPrefWidth(125);

        TableColumn<SlotAnalysisResult, String> colReviewStatus = new TableColumn<>("Review Status");
        colReviewStatus.setCellValueFactory(c -> {
            int id = c.getValue().getId();
            String rev = appState != null ? appState.getReviewStatusForSlot(id) : "Provisional (Awaiting Review)";
            return new SimpleStringProperty(rev);
        });
        colReviewStatus.setCellFactory(col -> formatStringCell("Provisional vs Reviewer-Confirmed training state"));
        colReviewStatus.setMinWidth(135);
        colReviewStatus.setPrefWidth(155);

        TableColumn<SlotAnalysisResult, String> colDetectorSuggestion = new TableColumn<>("Detector Suggestion");
        colDetectorSuggestion.setCellValueFactory(c -> new SimpleStringProperty(formatDetectorSuggestion(c.getValue())));
        colDetectorSuggestion.setCellFactory(col -> formatStringCell("Original analysis result from ONNX vehicle detections and slot overlap"));
        colDetectorSuggestion.setMinWidth(135);
        colDetectorSuggestion.setPrefWidth(160);

        TableColumn<SlotAnalysisResult, String> colLoadedTreePrediction = new TableColumn<>("Loaded Tree Prediction");
        colLoadedTreePrediction.setCellValueFactory(c -> {
            SlotAnalysisResult row = c.getValue();
            DecisionTreeModel.DecisionTrace trace = separatelyAppliedTreeTraces.get(row.getId());
            if (trace == null) return new SimpleStringProperty("Not applied");
            String value = String.format(Locale.ROOT, "%s (purity %.0f%%)",
                trace.finalStatus(), trace.leaf().leafPurity() * 100.0);
            if (row.getDetectorStatus() == VehicleDetectionService.DetectorStatus.READY) {
                SlotStatus detectorStatus = row.getVehicleDetectorStatus();
                if (detectorStatus != null && detectorStatus != SlotStatus.NEEDS_REVIEW
                        && detectorStatus != SlotStatus.UNCLASSIFIED && detectorStatus != trace.finalStatus()) {
                    value += " — DIFFERS FROM DETECTOR";
                }
            }
            return new SimpleStringProperty(value);
        });
        colLoadedTreePrediction.setCellFactory(col -> formatStringCell(
            "Separate prediction from the explicitly loaded Decision Tree; this never changes Current Status"));
        colLoadedTreePrediction.setMinWidth(150);
        colLoadedTreePrediction.setPrefWidth(175);

        TableColumn<SlotAnalysisResult, String> colOrigin = new TableColumn<>("Boundary Origin");
        colOrigin.setCellValueFactory(c -> {
            ParkingSlot s = c.getValue().getSlot();
            return new SimpleStringProperty(s != null ? s.getOrigin().getDisplayName() : "Unknown");
        });
        colOrigin.setCellFactory(col -> formatStringCell("Origin source of the slot boundary polygon"));
        colOrigin.setMinWidth(110);
        colOrigin.setPrefWidth(130);

        table.getColumns().setAll(
            colId, colCurrentStatus, colVerifiedLabel, colReviewStatus, colDetectorSuggestion, colLoadedTreePrediction
        );
    }

    public static String styleForStatus(String statusName) {
        if ("OCCUPIED".equals(statusName)) {
            return "-fx-background-color: #7f1d1d; -fx-text-fill: #fecaca; -fx-font-weight: bold; -fx-font-size: 13px; -fx-padding: 5px 8px;";
        } else if ("AVAILABLE".equals(statusName)) {
            return "-fx-background-color: #064e3b; -fx-text-fill: #a7f3d0; -fx-font-weight: bold; -fx-font-size: 13px; -fx-padding: 5px 8px;";
        } else if ("UNCLASSIFIED".equals(statusName)) {
            return "-fx-background-color: #1e293b; -fx-text-fill: #e2e8f0; -fx-font-weight: bold; -fx-font-size: 13px; -fx-padding: 5px 8px;";
        } else {
            return "-fx-background-color: #78350f; -fx-text-fill: #fde68a; -fx-font-weight: bold; -fx-font-size: 13px; -fx-padding: 5px 8px;";
        }
    }

    private TableCell<SlotAnalysisResult, String> formatStringCell(String descriptionPrefix) {
        return new TableCell<>() {
            @Override
            protected void updateItem(String item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                    setTooltip(null);
                    setStyle("");
                } else {
                    setText(item);
                    setStyle(BASE_CELL_STYLE);
                    setTooltip(new Tooltip(descriptionPrefix + ":\n" + item));
                }
            }
        };
    }

    /**
     * Populates the table without automatically selecting a row so that when no slot has been
     * selected by the user, only the unobtrusive instruction prompt is shown (Requirement 2.D).
     */
    public void setData(List<SlotAnalysisResult> results) {
        // A new detection/feature run invalidates separately applied predictions because its
        // feature vectors may have changed. The loaded model itself remains active.
        separatelyAppliedTreeTraces.clear();
        separatelyAppliedTreeModelLabel = "";
        masterData.setAll(results);
        table.refresh();
        table.getSelectionModel().clearSelection();
        updateSelectedRowDetailPanel(null);
    }

    /** Shows predictions from a separately applied saved tree without changing the analysis status. */
    public void setSeparatelyAppliedTreePredictions(Map<Integer, DecisionTreeModel.DecisionTrace> traces,
                                                    String modelLabel) {
        separatelyAppliedTreeTraces.clear();
        if (traces != null) separatelyAppliedTreeTraces.putAll(traces);
        separatelyAppliedTreeModelLabel = modelLabel != null ? modelLabel : "Active Decision Tree";
        table.refresh();
        SlotAnalysisResult selected = table.getSelectionModel().getSelectedItem();
        if (selected != null) updateSelectedRowDetailPanel(selected);
    }

    /** Clears side-by-side predictions when the active model changes; analysis output is retained. */
    public void clearSeparatelyAppliedTreePredictions() {
        separatelyAppliedTreeTraces.clear();
        separatelyAppliedTreeModelLabel = "";
        table.refresh();
        SlotAnalysisResult selected = table.getSelectionModel().getSelectedItem();
        if (selected != null) updateSelectedRowDetailPanel(selected);
    }

    public int getSeparatelyAppliedTreePredictionCount() {
        return separatelyAppliedTreeTraces.size();
    }

    public void selectSlot(ParkingSlot slot) {
        if (slot == null) {
            table.getSelectionModel().clearSelection();
            updateSelectedRowDetailPanel(null);
            return;
        }
        for (SlotAnalysisResult r : filteredData) {
            if (r.getId() == slot.getId()) {
                table.getSelectionModel().select(r);
                table.scrollTo(r);
                updateSelectedRowDetailPanel(r);
                break;
            }
        }
    }

    public void setOnSlotSelected(Consumer<ParkingSlot> callback) {
        this.selectionCallback = callback;
    }

    public void setOnSlotReviewed(BiConsumer<SlotAnalysisResult, SlotStatus> callback) {
        this.onSlotReviewedCallback = callback;
    }

    public void setOnRequestBoundaryEdit(Runnable callback) {
        this.onRequestBoundaryEditCallback = callback;
    }

    public TableView<SlotAnalysisResult> getTable()            { return table; }
    public String getSelectedDetailBodyText()                  { return lblSelectedDetailBody.getText(); }
    public String getSelectedDetailHeaderText()                { return lblSelectedDetailHeader.getText(); }
    public boolean isUnselectedPromptVisible()                 { return lblUnselectedPrompt.isVisible(); }
    public boolean isSelectedSlotActionBoxVisible()            { return selectedSlotActionBox.isVisible(); }
    public TitledPane getAdvancedSlotDetailsPane()             { return advancedSlotDetailsPane; }
    public Button getBtnMarkOccupied()                         { return btnMarkOccupied; }
    public Button getBtnMarkAvailable()                        { return btnMarkAvailable; }
    public Button getBtnMarkNeedsReview()                      { return btnMarkNeedsReview; }
    public Label getGeometryWarningBanner()                    { return lblGeometryWarningBanner; }

    /**
     * Computes total minimum width across all visible normal columns to verify that the table fits
     * within standard desktop resolutions (1366×768 and 1600×900) without horizontal overflow.
     */
    public double getTotalMinimumColumnWidth() {
        double sum = 0.0;
        for (TableColumn<SlotAnalysisResult, ?> col : table.getColumns()) {
            sum += col.getMinWidth();
        }
        return sum;
    }

    public static double computeWcagContrastRatio(String hexFg, String hexBg) {
        double lum1 = computeRelativeLuminance(hexFg);
        double lum2 = computeRelativeLuminance(hexBg);
        double brighter = Math.max(lum1, lum2);
        double darker = Math.min(lum1, lum2);
        return (brighter + 0.05) / (darker + 0.05);
    }

    public static double computeRelativeLuminance(String hexColor) {
        if (hexColor == null) throw new IllegalArgumentException("Hex color must not be null.");
        String clean = hexColor.trim().replace("#", "");
        if (clean.length() < 6) {
            throw new IllegalArgumentException("Expected 6-digit hex color; got " + hexColor);
        }
        int r = Integer.parseInt(clean.substring(0, 2), 16);
        int g = Integer.parseInt(clean.substring(2, 4), 16);
        int b = Integer.parseInt(clean.substring(4, 6), 16);
        double rs = srgbToLinear(r / 255.0);
        double gs = srgbToLinear(g / 255.0);
        double bs = srgbToLinear(b / 255.0);
        return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
    }

    private static double srgbToLinear(double c) {
        return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    }

    public static String formatCsvRow(SlotAnalysisResult r) {
        return r.toCsvRow();
    }

    void exportToCsv() {
        if (masterData.isEmpty()) {
            Alert alert = new Alert(Alert.AlertType.INFORMATION,
                "No data available to export.\n\nPlease complete Step 1: Load a parking lot photograph and click 'Detect and Analyse' to generate parking space occupancy results before exporting.",
                ButtonType.OK);
            alert.setHeaderText("No Analysis Data");
            alert.showAndWait();
            return;
        }

        boolean hasMultipleImages = appState != null && appState.getTotalImagesAnalysed() > 1;
        boolean exportAccumulated = false;

        if (hasMultipleImages) {
            ButtonType btnCurrentOnly = new ButtonType("Current Image Only (" + masterData.size() + " slots)");
            ButtonType btnAccumulated = new ButtonType("All " + appState.getTotalImagesAnalysed() + " Images (" + appState.getTotalSlotsCollected() + " slots)");
            Alert choice = new Alert(Alert.AlertType.CONFIRMATION,
                "You have analysed " + appState.getTotalImagesAnalysed() + " images (" + appState.getTotalSlotsCollected() + " total slots).\nWhich dataset would you like to export?",
                btnCurrentOnly, btnAccumulated, ButtonType.CANCEL);
            choice.setTitle("Select Export Scope");
            choice.setHeaderText("Export Scope Options");
            var res = choice.showAndWait();
            if (res.isEmpty() || res.get() == ButtonType.CANCEL) {
                return;
            }
            if (res.get() == btnAccumulated) {
                exportAccumulated = true;
            }
        }

        FileChooser fc = new FileChooser();
        fc.setTitle(exportAccumulated ? "Export Accumulated Multi-Image Dataset to CSV" : "Export Current Image Analysis Report to CSV");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV files (*.csv)", "*.csv"));
        fc.setInitialFileName(exportAccumulated ? "parking_accumulated_dataset.csv" : "parking_analysis_report.csv");
        File file = fc.showSaveDialog(getScene() != null ? getScene().getWindow() : null);

        if (file != null) {
            try (BufferedWriter writer = Files.newBufferedWriter(file.toPath(), StandardCharsets.UTF_8)) {
                if (exportAccumulated && appState != null) {
                    writer.write("image_name,image_sha256,slot_id,status,boundary_origin,leaf_purity,mean_intensity,foreground_ratio,dark_pixel_ratio,edge_density,object_area,mean_saturation,saturation_ratio,largest_blob_ratio,decision_rule\n");
                    int totalWritten = 0;
                    for (com.parknchillai.app.ImageSessionRecord session : appState.getAllImageSessions()) {
                        if (session.getAnalysisSummary() != null) {
                            String imgName = session.getFileName();
                            String sha = session.getSourceImageSha256();
                            for (SlotAnalysisResult r : session.getAnalysisSummary().getSlotResults()) {
                                String origin = r.getSlot() != null ? r.getSlot().getOrigin().getDisplayName() : "Unknown";
                                writer.write(String.format(Locale.ROOT, "\"%s\",\"%s\",%s,%s,%s\n",
                                    imgName, sha, r.getId(), r.getStatus().name(), origin));
                                totalWritten++;
                            }
                        }
                    }
                    Alert alert = new Alert(Alert.AlertType.INFORMATION, "Exported accumulated dataset (" + totalWritten + " rows across " + appState.getTotalImagesAnalysed() + " images) to:\n" + file.getAbsolutePath(), ButtonType.OK);
                    alert.showAndWait();
                } else {
                    writer.write("slot_id,status,boundary_origin,leaf_purity,mean_intensity,foreground_ratio,dark_pixel_ratio,edge_density,object_area,mean_saturation,saturation_ratio,largest_blob_ratio,decision_rule\n");
                    for (SlotAnalysisResult r : masterData) {
                        String origin = r.getSlot() != null ? r.getSlot().getOrigin().getDisplayName() : "Unknown";
                        writer.write(r.getId() + "," + r.getStatus().name() + ",\"" + origin + "\","
                            + String.format(Locale.ROOT, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,\"%s\"\n",
                                r.getLeafPurity(),
                                r.getFeatures().getMeanIntensity(),
                                r.getFeatures().getForegroundRatio(),
                                r.getFeatures().getDarkPixelRatio(),
                                r.getFeatures().getEdgeDensity(),
                                r.getFeatures().getObjectArea(),
                                r.getFeatures().getMeanSaturation(),
                                r.getFeatures().getSaturationRatio(),
                                r.getFeatures().getLargestBlobRatio(),
                                r.getDecisionPath() != null ? r.getDecisionPath().replace("\"", "'") : ""));
                    }
                    Alert alert = new Alert(Alert.AlertType.INFORMATION, "Exported " + masterData.size() + " rows to:\n" + file.getAbsolutePath(), ButtonType.OK);
                    alert.showAndWait();
                }
            } catch (IOException ex) {
                Alert alert = new Alert(Alert.AlertType.ERROR, "Failed to export CSV: " + ex.getMessage(), ButtonType.OK);
                alert.showAndWait();
            }
        }
    }
}
