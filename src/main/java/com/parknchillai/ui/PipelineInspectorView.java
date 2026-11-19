package com.parknchillai.ui;

import com.parknchillai.ai.DecisionTreeModel;
import com.parknchillai.app.SlotAnalysisResult;
import com.parknchillai.app.SlotDiagnosticRecord;
import com.parknchillai.detection.VehicleDetectionService;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Cursor;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextField;
import javafx.scene.control.TitledPane;
import javafx.scene.control.Tooltip;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * Per-Slot Advanced IVP &amp; AI Diagnostics View (ParkNChill v2.3 Requirement 4):
 * embedded inside a collapsible advanced diagnostics section in Step 3 so that
 * all 8 intermediate IVP stage images, 10 extracted feature values ({@code v2.3-10feat-raw-v1}),
 * ONNX input/output details, and CART rule paths remain accessible without cluttering
 * the primary review workflow.
 */
public class PipelineInspectorView extends VBox {

    private final Label lblSlotTitle = new Label("Select a slot to inspect");
    private final Label lblInputRegionMeta = new Label("Input Region: —");
    private final Label lblStatusBadge = new Label("NO SELECTION");
    private final Label lblPredictionSource = new Label("Prediction Source: —");
    private final Label lblLeafStats = new Label("Leaf Support: —");
    private final Label lblDecisionSteps = new Label("Step-by-step split trace will appear here after analysis.");
    private final Label lblVehicleDetectionInfo = new Label("Vehicle Detections: —");
    private final Label lblOverlapEvidenceInfo = new Label("Vehicle-to-Slot Overlap Evidence: —");
    private final Label lblReviewReasonInfo = new Label("Review Reason: —");
    private final Label lblHumanExplanation = new Label("Select any parking space to inspect its 8 IVP stages, 10 features, CART decision path, and ONNX overlap diagnostics.");

    private final ComboBox<String> slotSelector = new ComboBox<>();
    private final Button btnPrev = new Button("< Prev");
    private final Button btnNext = new Button("Next >");

    private final TextField txtReviewerId = new TextField("OPERATOR_1");
    private final Button btnLabelOccupied = new Button("Mark Occupied");
    private final Button btnLabelAvailable = new Button("Mark Available");

    // 10 Feature value labels + domain status labels + source tags (v2.3-10feat-enhanced-v1)
    private final Label[] featureValLabels = new Label[FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT];
    private final Label[] featureDomainLabels = new Label[FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT];
    private final Label[] featureSourceLabels = new Label[FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT];
    private final Label lblSelectedFeatureSpec = new Label(
        "Click or hover any feature card below to inspect its mathematical formula, valid bounds, source stage, and classifier role.");

    // Four canonical display options
    private final ComboBox<String> displayOptionSelector = new ComboBox<>();
    private final ImageView ivDisplayOption = createImageView(200);
    private final Label lblDisplayOptionNotes = new Label();
    private BufferedImage fullOriginalImage;
    private BufferedImage fullEnhancedImage;

    // 8 Intermediate image views (140x140)
    private final ImageView ivRoi = createImageView(140);
    private final ImageView ivGray = createImageView(140);
    private final ImageView ivSaturation = createImageView(140);
    private final ImageView ivSmoothed = createImageView(140);
    private final ImageView ivBinary = createImageView(140);
    private final ImageView ivClosed = createImageView(140);
    private final ImageView ivLargestBlob = createImageView(140);
    private final ImageView ivEdges = createImageView(140);

    // Side-by-side stage comparison banner
    private final ImageView ivCompareRoi = createImageView(180);
    private final ImageView ivCompareStage = createImageView(180);
    private final Label lblCompareStageTitle = new Label("Selected Stage: 5. Binary Threshold");
    private final Label lblCompareStageDesc = new Label("Click any of the 8 stage cards below to compare against the original cropped ROI.");

    private DecisionTreeModel currentModel;
    private List<SlotAnalysisResult> currentResults = new ArrayList<>();
    private SlotAnalysisResult activeResult;
    private int currentSlotIndex = -1;
    private Consumer<ParkingSlot> onSlotNavigated;
    private BiConsumer<SlotAnalysisResult, SlotStatus> onSampleLabeled;

    public PipelineInspectorView() {
        setSpacing(8);
        setPadding(new Insets(10));
        getStyleClass().add("pipeline-inspector-container");
        VBox.setVgrow(this, Priority.ALWAYS);

        for (int i = 0; i < FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT; i++) {
            featureValLabels[i] = new Label("—");
            featureDomainLabels[i] = new Label("Domain: —");
            featureSourceLabels[i] = new Label(i < 8
                ? "[Source: Enhanced Slot Pixels (ivp-enhance-v1)]"
                : "[Source: ONNX Detector + Slot Geometry]");
            featureSourceLabels[i].setStyle("-fx-font-size: 9px; -fx-text-fill: #94a3b8;");
            featureSourceLabels[i].setWrapText(true);
        }

        displayOptionSelector.getItems().addAll(
            "Selected Slot Crop (Enhanced)",
            "Selected Slot Crop (Original)",
            "Original Image (Photograph)",
            "Enhanced Image (Canonical Policy: ivp-enhance-v1)"
        );
        displayOptionSelector.getSelectionModel().selectFirst();
        displayOptionSelector.setOnAction(e -> updateDisplayOptionView());
        lblDisplayOptionNotes.setStyle("-fx-text-fill: #94a3b8; -fx-font-size: 11px;");
        lblDisplayOptionNotes.setWrapText(true);

        HBox navHeader = createNavigationHeader();
        VBox decisionCard = createDecisionExplanationCard();
        TitledPane featuresSection = createCollapsibleFeaturesSection();
        TitledPane pipelineSection = createCollapsiblePipelineGallery();

        VBox contentBox = new VBox(8, decisionCard, featuresSection, pipelineSection);
        ScrollPane masterScroll = new ScrollPane(contentBox);
        masterScroll.setFitToWidth(true);
        masterScroll.getStyleClass().add("clean-scroll-pane");
        VBox.setVgrow(masterScroll, Priority.ALWAYS);

        getChildren().addAll(navHeader, masterScroll);
    }

    public String getReviewerId() {
        return txtReviewerId.getText() == null ? "" : txtReviewerId.getText().trim();
    }

    public void setReviewerId(String reviewerId) {
        txtReviewerId.setText(reviewerId == null ? "" : reviewerId);
    }

    public void setModelForDomainInspection(DecisionTreeModel model) {
        this.currentModel = model;
        if (activeResult != null) {
            updateSlot(activeResult);
        }
    }

    private HBox createNavigationHeader() {
        HBox bar = new HBox(10);
        bar.setAlignment(Pos.CENTER_LEFT);

        VBox titleStack = new VBox(2);
        lblSlotTitle.getStyleClass().add("inspector-slot-title");
        lblInputRegionMeta.setStyle("-fx-text-fill: #cbd5e1; -fx-font-size: 11.5px;");
        titleStack.getChildren().addAll(lblSlotTitle, lblInputRegionMeta);
        HBox.setHgrow(titleStack, Priority.ALWAYS);

        lblStatusBadge.getStyleClass().addAll("status-badge", "badge-neutral");

        slotSelector.setPromptText("Select slot...");
        slotSelector.setPrefWidth(145);
        slotSelector.setOnAction(e -> {
            int selected = slotSelector.getSelectionModel().getSelectedIndex();
            if (selected >= 0 && selected < currentResults.size()) {
                selectByIndex(selected);
            }
        });

        btnPrev.getStyleClass().add("btn-tool");
        btnPrev.setOnAction(e -> {
            if (currentSlotIndex > 0) selectByIndex(currentSlotIndex - 1);
        });

        btnNext.getStyleClass().add("btn-tool");
        btnNext.setOnAction(e -> {
            if (currentSlotIndex >= 0 && currentSlotIndex < currentResults.size() - 1) {
                selectByIndex(currentSlotIndex + 1);
            }
        });

        bar.getChildren().addAll(titleStack, lblStatusBadge, slotSelector, btnPrev, btnNext);
        return bar;
    }

    private VBox createDecisionExplanationCard() {
        VBox decisionCard = new VBox(5);
        decisionCard.getStyleClass().add("decision-explanation-card");

        Label decisionTitle = new Label("ONNX Input/Output Details, Overlap Evidence & Decision Tree Rule Path:");
        decisionTitle.setStyle("-fx-font-weight: bold; -fx-text-fill: #38bdf8; -fx-font-size: 12.5px;");

        lblPredictionSource.setWrapText(true);
        lblPredictionSource.setStyle("-fx-text-fill: #f8fafc; -fx-font-weight: bold; -fx-font-size: 12px;");

        lblLeafStats.setWrapText(true);
        lblLeafStats.setStyle("-fx-text-fill: #e2e8f0; -fx-font-weight: bold; -fx-font-size: 11.5px;");

        lblDecisionSteps.setWrapText(true);
        lblDecisionSteps.setStyle("-fx-font-family: 'Consolas', 'Courier New', monospace; -fx-text-fill: #bae6fd; -fx-font-size: 12px;");

        lblVehicleDetectionInfo.setWrapText(true);
        lblVehicleDetectionInfo.setStyle("-fx-text-fill: #38bdf8; -fx-font-size: 12px; -fx-font-weight: bold;");

        lblOverlapEvidenceInfo.setWrapText(true);
        lblOverlapEvidenceInfo.setStyle("-fx-text-fill: #e2e8f0; -fx-font-size: 11.5px;");

        lblReviewReasonInfo.setWrapText(true);
        lblReviewReasonInfo.setStyle("-fx-text-fill: #fde68a; -fx-font-weight: bold; -fx-font-size: 11.5px;");

        lblHumanExplanation.setWrapText(true);
        lblHumanExplanation.getStyleClass().add("decision-path-text");

        btnLabelOccupied.getStyleClass().add("btn-secondary");
        btnLabelOccupied.setDisable(true);
        btnLabelOccupied.setOnAction(e -> {
            if (activeResult != null && onSampleLabeled != null) {
                onSampleLabeled.accept(activeResult, SlotStatus.OCCUPIED);
            }
        });

        btnLabelAvailable.getStyleClass().add("btn-secondary");
        btnLabelAvailable.setDisable(true);
        btnLabelAvailable.setOnAction(e -> {
            if (activeResult != null && onSampleLabeled != null) {
                onSampleLabeled.accept(activeResult, SlotStatus.AVAILABLE);
            }
        });

        decisionCard.getChildren().addAll(
            decisionTitle,
            lblPredictionSource,
            lblVehicleDetectionInfo,
            lblOverlapEvidenceInfo,
            lblLeafStats,
            lblDecisionSteps,
            lblReviewReasonInfo,
            lblHumanExplanation
        );
        return decisionCard;
    }

    private TitledPane createCollapsibleFeaturesSection() {
        VBox container = new VBox(6);
        container.setPadding(new Insets(8));

        lblSelectedFeatureSpec.setWrapText(true);
        lblSelectedFeatureSpec.setStyle(
            "-fx-background-color: #0f172a; -fx-text-fill: #f1f5f9; -fx-padding: 6px 10px; " +
            "-fx-background-radius: 5px; -fx-font-size: 11.5px;");

        GridPane grid = new GridPane();
        grid.setHgap(8);
        grid.setVgap(6);

        for (int i = 0; i < FeatureVector.EXTENDED_SPECS.size(); i++) {
            FeatureVector.FeatureSpec spec = FeatureVector.EXTENDED_SPECS.get(i);
            int col = i % 5;
            int row = i / 5;
            addFeatureCell(grid, spec, featureValLabels[i], featureDomainLabels[i], featureSourceLabels[i], col, row);
        }

        container.getChildren().addAll(lblSelectedFeatureSpec, grid);
        TitledPane tp = new TitledPane("Feature Extraction Diagnostics (10-Feature Schema " + com.parknchillai.imageprocessing.ProcessingPolicy.CANONICAL_SCHEMA_ID + ")", container);
        tp.setCollapsible(true);
        tp.setExpanded(false);
        return tp;
    }

    private void addFeatureCell(GridPane grid,
                                FeatureVector.FeatureSpec spec,
                                Label valueLabel,
                                Label domainLabel,
                                Label sourceLabel,
                                int col,
                                int row) {
        VBox cell = new VBox(2);
        cell.setPadding(new Insets(6, 8, 6, 8));
        cell.getStyleClass().add("feature-cell");
        cell.setCursor(Cursor.HAND);

        Label nameLabel = new Label(spec.displayName());
        nameLabel.getStyleClass().add("feature-name");

        valueLabel.getStyleClass().add("feature-value");
        domainLabel.setStyle("-fx-font-size: 10.5px; -fx-text-fill: #34d399; -fx-font-weight: bold;");

        String fullSpecText = String.format(Locale.ROOT,
            "%s (%s) | Valid Range: [%.1f, %.1f] | %s\nFormula: %s\nClassifier Role: %s",
            spec.displayName(), spec.key(), spec.minValid(), spec.maxValid(),
            sourceLabel.getText(), spec.formula(), spec.classifierRole());

        cell.setOnMouseClicked(e -> lblSelectedFeatureSpec.setText(fullSpecText));
        cell.setOnMouseEntered(e -> lblSelectedFeatureSpec.setText(fullSpecText));

        cell.getChildren().addAll(nameLabel, valueLabel, domainLabel, sourceLabel);
        GridPane.setHgrow(cell, Priority.ALWAYS);
        grid.add(cell, col, row);
    }

    private TitledPane createCollapsiblePipelineGallery() {
        VBox container = new VBox(8);
        container.setPadding(new Insets(8));

        VBox displayOptionsBox = new VBox(6);
        displayOptionsBox.setPadding(new Insets(8));
        displayOptionsBox.setStyle("-fx-background-color: #0f172a; -fx-border-color: #334155; -fx-border-radius: 6px; -fx-background-radius: 6px;");

        HBox displaySelectorRow = new HBox(10);
        displaySelectorRow.setAlignment(Pos.CENTER_LEFT);
        Label lblDisplayTitle = new Label("Canonical View Mode:");
        lblDisplayTitle.setStyle("-fx-font-weight: bold; -fx-text-fill: #38bdf8; -fx-font-size: 12px;");
        displayOptionSelector.setPrefWidth(320);
        displaySelectorRow.getChildren().addAll(lblDisplayTitle, displayOptionSelector);

        HBox displayViewRow = new HBox(14, ivDisplayOption, lblDisplayOptionNotes);
        displayViewRow.setAlignment(Pos.CENTER_LEFT);
        HBox.setHgrow(lblDisplayOptionNotes, Priority.ALWAYS);

        displayOptionsBox.getChildren().addAll(displaySelectorRow, displayViewRow);

        HBox compareCard = new HBox(14);
        compareCard.setAlignment(Pos.CENTER_LEFT);
        compareCard.setPadding(new Insets(8));
        compareCard.setStyle("-fx-background-color: #0f172a; -fx-border-color: #334155; -fx-border-radius: 6px; -fx-background-radius: 6px;");

        VBox origBox = new VBox(4, new Label("Selected Slot Crop (Original)"), ivCompareRoi);
        origBox.setAlignment(Pos.CENTER);
        ((Label) origBox.getChildren().get(0)).getStyleClass().add("stage-card-title");

        VBox selBox = new VBox(4, lblCompareStageTitle, ivCompareStage);
        selBox.setAlignment(Pos.CENTER);
        lblCompareStageTitle.getStyleClass().add("stage-card-title");

        VBox descBox = new VBox(6, new Label("Comparison Notes:"), lblCompareStageDesc);
        ((Label) descBox.getChildren().get(0)).setStyle("-fx-font-weight: bold; -fx-text-fill: #38bdf8; -fx-font-size: 12px;");
        lblCompareStageDesc.setWrapText(true);
        lblCompareStageDesc.setStyle("-fx-text-fill: #e2e8f0; -fx-font-size: 11.5px;");
        HBox.setHgrow(descBox, Priority.ALWAYS);

        compareCard.getChildren().addAll(origBox, selBox, descBox);

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setAlignment(Pos.CENTER);

        grid.add(createStageCard("1. Enhanced Slot Crop", ivRoi,
            "Enhanced RGB crop (Canonical ivp-enhance-v1) from which all 8 numerical IVP features are calculated."), 0, 0);
        grid.add(createStageCard("2. Grayscale (BT.709)", ivGray,
            "Luminance conversion L = 0.2126R + 0.7152G + 0.0722B from the enhanced slot crop."), 1, 0);
        grid.add(createStageCard("3. HSV Saturation Map", ivSaturation,
            "Shadow-invariant color saturation channel S in [0, 255] from enhanced crop (S > 90 identifies chromatic vehicles)."), 2, 0);
        grid.add(createStageCard("4. 5×5 Gaussian Blur", ivSmoothed,
            "Low-pass 5×5 Gaussian smoothing (sigma = 5/6) suppressing asphalt grain prior to segmentation."), 3, 0);
        grid.add(createStageCard("5. Binary Threshold", ivBinary,
            "Foreground mask from unimodal-guarded Otsu or adaptive thresholding + HSV S > 90 on enhanced pixels."), 0, 1);
        grid.add(createStageCard("6. 3×3 Morpho Closing", ivClosed,
            "Dilation followed by erosion (3×3 flat kernel) bridging windshield/roof gaps into solid vehicle regions."), 1, 1);
        grid.add(createStageCard("7. Largest Blob (BFS)", ivLargestBlob,
            "8-connected BFS flood-fill isolating the single largest contiguous foreground component (largestBlobRatio)."), 2, 1);
        grid.add(createStageCard("8. 3×3 Sobel Edges", ivEdges,
            "Gradient magnitude sqrt(Gx^2 + Gy^2) highlighting structural boundaries on smoothed enhanced grayscale."), 3, 1);

        container.getChildren().addAll(displayOptionsBox, compareCard, grid);
        TitledPane tp = new TitledPane("8-Stage Image Processing Pipeline Previews (ivp-enhance-v1)", container);
        tp.setCollapsible(true);
        tp.setExpanded(false);
        return tp;
    }

    private VBox createStageCard(String stageTitle, ImageView imageView, String explanation) {
        VBox card = new VBox(5);
        card.setAlignment(Pos.CENTER);
        card.setPadding(new Insets(8));
        card.getStyleClass().add("pipeline-stage-card");
        card.setCursor(Cursor.HAND);

        Label lbl = new Label(stageTitle);
        lbl.getStyleClass().add("stage-card-title");

        Button btnZoom = new Button("Enlarge");
        btnZoom.getStyleClass().add("btn-tool");
        btnZoom.setOnAction(e -> showEnlargedStageDialog(stageTitle, imageView.getImage(), explanation));

        card.setOnMouseClicked(e -> {
            lblCompareStageTitle.setText("Selected: " + stageTitle);
            ivCompareStage.setImage(imageView.getImage());
            lblCompareStageDesc.setText(explanation);
        });

        Tooltip.install(card, new Tooltip(stageTitle + "\n" + explanation));
        card.getChildren().addAll(lbl, imageView, btnZoom);
        return card;
    }

    private void showEnlargedStageDialog(String stageTitle, Image fxImg, String explanation) {
        if (fxImg == null) return;
        Alert dialog = new Alert(Alert.AlertType.INFORMATION, "", ButtonType.OK);
        dialog.setTitle("Stage Inspector — " + stageTitle);
        dialog.setHeaderText(stageTitle);

        ImageView bigView = createImageView(340);
        bigView.setImage(fxImg);

        Label desc = new Label(explanation);
        desc.setWrapText(true);
        desc.setMaxWidth(360);

        VBox content = new VBox(10, bigView, desc);
        content.setAlignment(Pos.CENTER);
        content.setPadding(new Insets(10));
        dialog.getDialogPane().setContent(content);
        dialog.showAndWait();
    }

    private ImageView createImageView(double size) {
        ImageView iv = new ImageView();
        iv.setFitWidth(size);
        iv.setFitHeight(size);
        iv.setPreserveRatio(true);
        iv.setSmooth(false);
        iv.getStyleClass().add("pipeline-image-view");
        return iv;
    }

    public void setAllResults(List<SlotAnalysisResult> results) {
        this.currentResults = results != null ? new ArrayList<>(results) : new ArrayList<>();
        slotSelector.getItems().clear();
        for (SlotAnalysisResult r : this.currentResults) {
            slotSelector.getItems().add(String.format("Slot #%d (%s)", r.getId(), r.getStatus()));
        }
        currentSlotIndex = -1;
        updateSlot(null);
    }

    private void selectByIndex(int index) {
        if (index < 0 || index >= currentResults.size()) return;
        currentSlotIndex = index;
        slotSelector.getSelectionModel().select(index);
        SlotAnalysisResult res = currentResults.get(index);
        updateSlot(res);
        if (onSlotNavigated != null) {
            onSlotNavigated.accept(res.getSlot());
        }
        btnPrev.setDisable(index <= 0);
        btnNext.setDisable(index >= currentResults.size() - 1);
    }

    public void setOnSlotNavigated(Consumer<ParkingSlot> listener) {
        this.onSlotNavigated = listener;
    }

    public void setOnSampleLabeled(BiConsumer<SlotAnalysisResult, SlotStatus> listener) {
        this.onSampleLabeled = listener;
    }

    public void updateSlot(SlotAnalysisResult result) {
        this.activeResult = result;
        if (result == null) {
            lblSlotTitle.setText("Select a slot to inspect");
            lblInputRegionMeta.setText("Input Region: —");
            lblStatusBadge.setText("NO SELECTION");
            lblStatusBadge.getStyleClass().setAll("status-badge", "badge-neutral");
            lblPredictionSource.setText("Prediction Source: —");
            lblLeafStats.setText("Leaf Support: —");
            lblDecisionSteps.setText("Step-by-step split trace will appear here after analysis.");
            lblVehicleDetectionInfo.setText("Vehicle Detections: —");
            lblOverlapEvidenceInfo.setText("Vehicle-to-Slot Overlap Evidence: —");
            lblReviewReasonInfo.setText("Review Reason: —");
            lblHumanExplanation.setText("Select any parking space to inspect its 8 IVP stages, 10 features, CART decision path, and ONNX overlap diagnostics.");
            btnLabelOccupied.setDisable(true);
            btnLabelAvailable.setDisable(true);
            for (int i = 0; i < FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT; i++) {
                featureValLabels[i].setText("—");
                featureDomainLabels[i].setText("Domain: —");
            }
            ivRoi.setImage(null);
            ivGray.setImage(null);
            ivSaturation.setImage(null);
            ivSmoothed.setImage(null);
            ivBinary.setImage(null);
            ivClosed.setImage(null);
            ivLargestBlob.setImage(null);
            ivEdges.setImage(null);
            ivCompareRoi.setImage(null);
            ivCompareStage.setImage(null);
            return;
        }

        if (!currentResults.isEmpty()) {
            for (int i = 0; i < currentResults.size(); i++) {
                if (currentResults.get(i).getId() == result.getId()) {
                    currentSlotIndex = i;
                    slotSelector.getSelectionModel().select(i);
                    btnPrev.setDisable(i <= 0);
                    btnNext.setDisable(i >= currentResults.size() - 1);
                    break;
                }
            }
        }

        ParkingSlot slot = result.getSlot();
        lblSlotTitle.setText("Slot #" + result.getId() + " Diagnostics");
        BufferedImage roiImg = result.getRoiImage();
        int cropW = roiImg != null ? roiImg.getWidth() : slot.getWidth();
        int cropH = roiImg != null ? roiImg.getHeight() : slot.getHeight();
        lblInputRegionMeta.setText(String.format(Locale.ROOT,
            "Input Region: x=%d, y=%d, w=%d, h=%d (Cropped: %d×%d px) | Origin: %s",
            slot.getX(), slot.getY(), slot.getWidth(), slot.getHeight(),
            cropW, cropH, slot.getOrigin().getDisplayName()));

        SlotStatus status = slot.getStatus() != null ? slot.getStatus() : result.getStatus();
        double purityPct = result.getLeafPurity() * 100.0;
        lblStatusBadge.setText(String.format(Locale.ROOT, "%s (%.0f%% Leaf Purity)", status.name(), purityPct));
        if (status == SlotStatus.OCCUPIED) {
            lblStatusBadge.getStyleClass().setAll("status-badge", "badge-occupied");
        } else if (status == SlotStatus.AVAILABLE) {
            lblStatusBadge.getStyleClass().setAll("status-badge", "badge-available");
        } else {
            lblStatusBadge.getStyleClass().setAll("status-badge", "badge-review");
        }

        SlotDiagnosticRecord diag = result.getDiagnosticRecord();
        lblPredictionSource.setText("Final Outcome: " + status + " | Prediction Source: " + result.getPredictionSource());

        DecisionTreeModel.DecisionTrace trace = result.getDecisionTrace();
        DecisionTreeModel.LeafSummary leaf = trace.leaf();
        lblLeafStats.setText(String.format(Locale.ROOT,
            "Decision Tree Prediction: %s | Terminal Leaf Support: %d Occupied, %d Available (Total n=%d, Empirical Leaf Purity = %.1f%%)",
            result.getTreeStatus(), leaf.occupiedCount(), leaf.availableCount(), leaf.totalSamples(), leaf.leafPurity() * 100.0
        ));

        if (trace.steps().isEmpty()) {
            lblDecisionSteps.setText("Decision Path: " + result.getDecisionPath());
        } else {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < trace.steps().size(); i++) {
                DecisionTreeModel.DecisionStep step = trace.steps().get(i);
                sb.append(String.format(Locale.ROOT, "  Step %d: %s\n", i + 1, step.formatStep()));
            }
            lblDecisionSteps.setText(sb.toString().trim());
        }

        if (diag != null) {
            lblVehicleDetectionInfo.setText("Vehicle Detections: " + diag.formatVehicleDetectionsSummary());
            lblOverlapEvidenceInfo.setText("Vehicle-to-Slot Overlap: " + diag.formatOverlapEvidenceSummary());
            String rev = diag.reviewReason();
            lblReviewReasonInfo.setText("Review Reason: " + ((rev == null || rev.isBlank()) ? "None (Confirmed Classification)" : rev));
        } else {
            lblVehicleDetectionInfo.setText("Vehicle Detections: UNAVAILABLE (" + VehicleDetectionService.DetectorStatus.MODEL_MISSING.getDisplayName() + ")");
            lblOverlapEvidenceInfo.setText("Vehicle-to-Slot Overlap: Not evaluated");
            lblReviewReasonInfo.setText("Review Reason: None");
        }

        lblHumanExplanation.setText(trace.humanReadableExplanation());
        btnLabelOccupied.setDisable(false);
        btnLabelAvailable.setDisable(false);

        FeatureVector fv = result.getFeatures();
        double[] values = fv.toArray(FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT);
        boolean detReady = fv.isDetectorAvailable()
            && (diag == null || diag.detectorStatus() == VehicleDetectionService.DetectorStatus.READY);

        for (int i = 0; i < FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT; i++) {
            String fmt = (i == FeatureVector.IDX_MEAN_INTENSITY || i == FeatureVector.IDX_MEAN_SATURATION)
                ? "%.2f" : "%.4f";

            if (i >= 8 && !detReady) {
                featureValLabels[i].setText("Unavailable");
                featureDomainLabels[i].setText("FLAGGED REVIEW");
                featureDomainLabels[i].setStyle("-fx-font-size: 10.5px; -fx-text-fill: #f87171; -fx-font-weight: bold;");
                featureSourceLabels[i].setText("[Source: ONNX Detector (Unavailable — slot flagged for review)]");
                featureSourceLabels[i].setStyle("-fx-font-size: 9px; -fx-text-fill: #f87171;");
            } else {
                featureValLabels[i].setText(String.format(Locale.ROOT, fmt, values[i]));
                if (i < 8) {
                    featureSourceLabels[i].setText("[Source: Enhanced Slot Pixels (ivp-enhance-v1)]");
                    featureSourceLabels[i].setStyle("-fx-font-size: 9px; -fx-text-fill: #94a3b8;");
                } else {
                    featureSourceLabels[i].setText("[Source: ONNX Detector + Slot Geometry]");
                    featureSourceLabels[i].setStyle("-fx-font-size: 9px; -fx-text-fill: #94a3b8;");
                }

                if (currentModel != null && i < currentModel.getFeatureCount() && !Double.isNaN(currentModel.getTrainMin(i))) {
                    double tMin = currentModel.getTrainMin(i);
                    double tMax = currentModel.getTrainMax(i);
                    boolean outOfDom = currentModel.isFeatureOutOfDomain(i, values[i], 0.0);
                    featureDomainLabels[i].setText(String.format(Locale.ROOT,
                        (outOfDom ? "OUT OF DOMAIN [" : "In Train [") + fmt + ".." + fmt + "]", tMin, tMax));
                    featureDomainLabels[i].setStyle(outOfDom
                        ? "-fx-font-size: 10.5px; -fx-text-fill: #fbbf24; -fx-font-weight: bold;"
                        : "-fx-font-size: 10.5px; -fx-text-fill: #34d399; -fx-font-weight: bold;");
                } else {
                    FeatureVector.FeatureSpec spec = FeatureVector.EXTENDED_SPECS.get(i);
                    featureDomainLabels[i].setText(String.format(Locale.ROOT, "Valid [%.0f..%.0f]", spec.minValid(), spec.maxValid()));
                    featureDomainLabels[i].setStyle("-fx-font-size: 10.5px; -fx-text-fill: #cbd5e1;");
                }
            }
        }

        Image fxEnhancedRoi = ImageFxUtils.toFxImage(result.getEnhancedRoiImage());
        Image fxOrigRoi = ImageFxUtils.toFxImage(result.getOriginalRoiImage());
        Image fxBin = ImageFxUtils.toFxImage(result.getBinaryImage());

        ivRoi.setImage(fxEnhancedRoi);
        ivGray.setImage(ImageFxUtils.toFxImage(result.getGrayImage()));
        ivSaturation.setImage(ImageFxUtils.toFxImage(result.getSaturationImage()));
        ivSmoothed.setImage(ImageFxUtils.toFxImage(result.getSmoothedImage()));
        ivBinary.setImage(fxBin);
        ivClosed.setImage(ImageFxUtils.toFxImage(result.getClosedImage()));
        ivLargestBlob.setImage(ImageFxUtils.toFxImage(result.getLargestBlobMask()));
        ivEdges.setImage(ImageFxUtils.toFxImage(result.getEdgeImage()));

        ivCompareRoi.setImage(fxOrigRoi);
        ivCompareStage.setImage(fxEnhancedRoi);
        lblCompareStageTitle.setText("Selected Slot Crop (Enhanced)");
        lblCompareStageDesc.setText("Side-by-side comparison: Original Crop (left) vs Canonical Enhanced Crop (right). Features are computed strictly from the enhanced crop.");

        updateDisplayOptionView();
    }

    public void setFullImages(BufferedImage original, BufferedImage enhanced) {
        this.fullOriginalImage = original;
        this.fullEnhancedImage = enhanced;
        updateDisplayOptionView();
    }

    private void updateDisplayOptionView() {
        String opt = displayOptionSelector.getValue();
        if (opt == null) return;
        switch (opt) {
            case "Original Image (Photograph)" -> {
                ivDisplayOption.setImage(fullOriginalImage != null ? ImageFxUtils.toFxImage(fullOriginalImage) : null);
                lblDisplayOptionNotes.setText("Unmodified full photograph as captured by the camera. Vehicle detection (ONNX) runs directly on this image.");
            }
            case "Enhanced Image (Canonical Policy: ivp-enhance-v1)" -> {
                ivDisplayOption.setImage(fullEnhancedImage != null ? ImageFxUtils.toFxImage(fullEnhancedImage) : null);
                lblDisplayOptionNotes.setText("Whole photograph after canonical contrast stretching (p02..p98) and tone correction under policy 'ivp-enhance-v1'. All slot crops are sliced from this image.");
            }
            case "Selected Slot Crop (Enhanced)" -> {
                ivDisplayOption.setImage(activeResult != null ? ImageFxUtils.toFxImage(activeResult.getEnhancedRoiImage()) : null);
                lblDisplayOptionNotes.setText("Exact pixel buffer used for 8-feature numerical extraction. What is displayed is what is computed.");
            }
            case "Selected Slot Crop (Original)" -> {
                ivDisplayOption.setImage(activeResult != null ? ImageFxUtils.toFxImage(activeResult.getOriginalRoiImage()) : null);
                lblDisplayOptionNotes.setText("Raw unenhanced crop of the slot region directly from the original photograph.");
            }
        }
    }

    public String getVehicleDetectionInfoText() { return lblVehicleDetectionInfo.getText(); }
    public String getOverlapEvidenceInfoText()  { return lblOverlapEvidenceInfo.getText(); }
    public String getPredictionSourceText()     { return lblPredictionSource.getText(); }
    public String getReviewReasonInfoText()     { return lblReviewReasonInfo.getText(); }
}
