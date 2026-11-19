package com.parknchillai.ui;

import com.parknchillai.app.AppState;
import com.parknchillai.imageprocessing.ImageEnhancer;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.RoiExtractor;
import javafx.animation.PauseTransition;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.stage.Window;
import javafx.util.Duration;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Interactive Image Enhancement and Active Working Image Studio (Problem 1 &amp; WS-3).
 *
 * <p>Responsive Architecture (WS-3):
 * <ul>
 *   <li>Debounced slider input (150 ms) ensures zero UI freezing while dragging.</li>
 *   <li>Downscaled preview bitmap (max 640x480) for instant interactive feedback (&lt;10 ms).</li>
 *   <li>Asynchronous background rendering worker with a monotonic generation counter
 *       discarding stale intermediate frames.</li>
 *   <li>Background Apply operation with responsive state feedback.</li>
 * </ul>
 */
public final class ImageEnhancementDialog {

    public static final int PREVIEW_MAX_WIDTH  = 640;
    public static final int PREVIEW_MAX_HEIGHT = 480;
    public static final int DEBOUNCE_DELAY_MS  = 150;

    private ImageEnhancementDialog() { }

    public static void show(Window owner, BufferedImage fullImage, ParkingSlot selectedSlot) {
        show(owner, null, fullImage, selectedSlot);
    }

    public static void show(Window owner, AppState appState, ParkingSlot selectedSlot) {
        BufferedImage img = appState != null ? appState.getOriginalImage() : null;
        show(owner, appState, img, selectedSlot);
    }

    public static void show(Window owner, AppState appState, BufferedImage fullImage, ParkingSlot selectedSlot) {
        BufferedImage sourceImage = (appState != null && appState.getOriginalImage() != null)
            ? appState.getOriginalImage()
            : fullImage;

        if (sourceImage == null) {
            Alert alert = new Alert(Alert.AlertType.INFORMATION, "Load a parking lot photograph before opening enhancement controls.");
            if (owner != null) alert.initOwner(owner);
            alert.setHeaderText("No image loaded");
            alert.showAndWait();
            return;
        }

        final BufferedImage source;
        final String scope;
        final boolean isSlotScope = selectedSlot != null;
        try {
            if (isSlotScope) {
                source = RoiExtractor.extract(sourceImage, selectedSlot);
                scope = "Slot #" + selectedSlot.getId() + " Inspection Mode — Enhancements preview on this slot crop.";
            } else {
                source = sourceImage;
                scope = "Active Working Image Studio — Adjust contrast, brightness, gamma, and noise reduction for the whole parking photograph.";
            }
        } catch (RuntimeException ex) {
            Alert alert = new Alert(Alert.AlertType.ERROR, "Could not crop selected slot: " + ex.getMessage());
            if (owner != null) alert.initOwner(owner);
            alert.showAndWait();
            return;
        }

        // Downscaled preview source (WS-3)
        final BufferedImage previewSource = createPreviewScale(source, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT);

        Dialog<Void> dialog = new Dialog<>();
        dialog.setTitle("ParkNChill — Image Enhancement Studio");
        if (owner != null) dialog.initOwner(owner);
        dialog.setResizable(true);

        DialogPane pane = dialog.getDialogPane();
        pane.setStyle("-fx-background-color: #0f172a; -fx-text-fill: #f8fafc;");
        pane.getButtonTypes().add(ButtonType.CLOSE);

        // Header Scope Banner
        Label lblScope = new Label(scope);
        lblScope.setWrapText(true);
        lblScope.setStyle("-fx-font-weight: bold; -fx-font-size: 13px; -fx-text-fill: #38bdf8; "
            + "-fx-background-color: #1e293b; -fx-padding: 10px 14px; -fx-background-radius: 6px; -fx-border-color: #334155; -fx-border-radius: 6px;");

        // Status badge indicating active working image state
        String currentWorkingDesc = (appState != null) ? appState.getActiveEnhancementDescription() : "Original";
        Label lblActiveStatus = new Label("Current Active Working Image: " + currentWorkingDesc);
        lblActiveStatus.setStyle("-fx-font-size: 11.5px; -fx-font-weight: bold; -fx-text-fill: #10b981; -fx-padding: 2px 4px;");

        // Image Previews
        ImageView originalView = new ImageView();
        ImageView enhancedView = new ImageView();
        originalView.setFitWidth(450); originalView.setFitHeight(300); originalView.setPreserveRatio(true); originalView.setSmooth(true);
        enhancedView.setFitWidth(450); enhancedView.setFitHeight(300); enhancedView.setPreserveRatio(true); enhancedView.setSmooth(true);

        Image originalFx = ImageFxUtils.toFxImage(previewSource);
        originalView.setImage(originalFx);
        enhancedView.setImage(originalFx);

        Label lblOriginalTitle = new Label("Original Photograph (Immutable Source)");
        lblOriginalTitle.setStyle("-fx-font-weight: bold; -fx-text-fill: #cbd5e1; -fx-font-size: 12px;");

        Label lblEnhancedTitle = new Label("Enhanced Working Image (Active IVP Target)");
        lblEnhancedTitle.setStyle("-fx-font-weight: bold; -fx-text-fill: #38bdf8; -fx-font-size: 12px;");

        VBox leftBox = new VBox(6, lblOriginalTitle, originalView);
        leftBox.setAlignment(Pos.TOP_CENTER);
        leftBox.setStyle("-fx-background-color: #1e293b; -fx-padding: 10px; -fx-background-radius: 6px; -fx-border-color: #334155; -fx-border-radius: 6px;");

        VBox rightBox = new VBox(6, lblEnhancedTitle, enhancedView);
        rightBox.setAlignment(Pos.TOP_CENTER);
        rightBox.setStyle("-fx-background-color: #1e293b; -fx-padding: 10px; -fx-background-radius: 6px; -fx-border-color: #0284c7; -fx-border-radius: 6px;");

        HBox previewBox = new HBox(12, leftBox, rightBox);
        previewBox.setAlignment(Pos.CENTER);
        HBox.setHgrow(leftBox, Priority.ALWAYS);
        HBox.setHgrow(rightBox, Priority.ALWAYS);

        // Sliders & Controls
        Slider contrastSlider = slider(0.5, 2.5, 1.0);
        Slider brightnessSlider = slider(-80, 80, 0);
        Slider gammaSlider = slider(0.4, 2.2, 1.0);
        ComboBox<String> noiseBox = new ComboBox<>();
        noiseBox.getItems().addAll("None (No smoothing)", "Light (3×3 Gaussian)", "Medium (5×5 Gaussian)");
        noiseBox.getSelectionModel().selectFirst();
        noiseBox.setStyle("-fx-background-color: #1e293b; -fx-text-fill: #f8fafc; -fx-border-color: #475569;");

        Label lblContrastVal = createValueLabel("1.00");
        Label lblBrightnessVal = createValueLabel("0");
        Label lblGammaVal = createValueLabel("1.00");

        GridPane grid = new GridPane();
        grid.setHgap(12); grid.setVgap(10);
        grid.setPadding(new Insets(12));
        grid.setStyle("-fx-background-color: #1e293b; -fx-background-radius: 6px; -fx-border-color: #334155; -fx-border-radius: 6px;");

        addControlRow(grid, 0, "Contrast", contrastSlider, lblContrastVal,
            "Expands or narrows luminance range. Higher contrast makes vehicles and lines stand out against pavement.");
        addControlRow(grid, 1, "Brightness", brightnessSlider, lblBrightnessVal,
            "Uniformly shifts pixel intensity. Useful for brightening dark shadows or dimming glaring reflections.");
        addControlRow(grid, 2, "Gamma", gammaSlider, lblGammaVal,
            "Non-linear tone curve. Values <1.0 brighten shadow detail without washing highlights; >1.0 darken midtones.");

        Label lblNoiseName = new Label("Noise Reduction");
        lblNoiseName.setStyle("-fx-font-weight: bold; -fx-text-fill: #f8fafc; -fx-font-size: 12px;");
        Label lblNoiseDesc = new Label("Spatial Gaussian smoothing suppresses camera sensor grain while preserving parking lines.");
        lblNoiseDesc.setStyle("-fx-text-fill: #94a3b8; -fx-font-size: 11px;");
        lblNoiseDesc.setWrapText(true);
        VBox noiseBoxCol = new VBox(2, lblNoiseName, lblNoiseDesc);
        grid.add(noiseBoxCol, 0, 3);
        grid.add(noiseBox, 1, 3);

        // Action Buttons
        Button btnApplyEnhancement = new Button("Apply Enhancement to Active Image");
        btnApplyEnhancement.setStyle("-fx-background-color: #059669; -fx-text-fill: #ffffff; -fx-font-weight: bold; -fx-padding: 8px 16px; -fx-background-radius: 5px; -fx-cursor: hand;");

        Button btnAutoCanonical = new Button("Use Automatic Enhancement (ivp-enhance-v1)");
        btnAutoCanonical.setStyle("-fx-background-color: #2563eb; -fx-text-fill: #ffffff; -fx-font-weight: bold; -fx-padding: 8px 14px; -fx-background-radius: 5px; -fx-cursor: hand;");

        Button btnResetOriginal = new Button("Reset to Original Photograph");
        btnResetOriginal.setStyle("-fx-background-color: #475569; -fx-text-fill: #ffffff; -fx-font-weight: bold; -fx-padding: 8px 14px; -fx-background-radius: 5px; -fx-cursor: hand;");

        HBox actionsBar = new HBox(10, btnApplyEnhancement, btnAutoCanonical, btnResetOriginal);
        actionsBar.setAlignment(Pos.CENTER_LEFT);

        Label note = new Label("Operational Note: When you click 'Apply Enhancement', the enhanced pixels become the active working image for all subsequent IVP processing, slot crops, and feature extraction. ONNX vehicle detection strictly preserves the original photograph.");
        note.setWrapText(true);
        note.setStyle("-fx-text-fill: #fbbf24; -fx-background-color: #3f2e1e; -fx-padding: 10px 14px; -fx-background-radius: 6px; -fx-border-color: #b45309; -fx-border-radius: 6px; -fx-font-size: 11.5px;");

        // Worker Executor & Generation Counter (WS-3)
        ExecutorService previewExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "EnhancementPreviewWorker");
            t.setDaemon(true);
            return t;
        });
        AtomicLong previewGeneration = new AtomicLong(0);
        dialog.setOnHidden(e -> previewExecutor.shutdownNow());

        // Debounce timer (150 ms)
        PauseTransition debounceTimer = new PauseTransition(Duration.millis(DEBOUNCE_DELAY_MS));

        Runnable requestPreviewComputation = () -> {
            long gen = previewGeneration.incrementAndGet();
            int noiseKernel = switch (noiseBox.getSelectionModel().getSelectedIndex()) {
                case 1 -> 3;
                case 2 -> 5;
                default -> 0;
            };
            double c = contrastSlider.getValue();
            double b = brightnessSlider.getValue();
            double g = gammaSlider.getValue();

            previewExecutor.submit(() -> {
                if (previewGeneration.get() != gen) return;
                BufferedImage adjusted = ImageEnhancer.adjust(previewSource, c, b, g, noiseKernel);
                if (previewGeneration.get() != gen) return;
                Image fxImg = ImageFxUtils.toFxImage(adjusted);
                Platform.runLater(() -> {
                    if (previewGeneration.get() == gen) {
                        enhancedView.setImage(fxImg);
                    }
                });
            });
        };

        debounceTimer.setOnFinished(e -> requestPreviewComputation.run());

        Runnable onSliderChanged = () -> {
            lblContrastVal.setText(String.format(Locale.ROOT, "%.2f", contrastSlider.getValue()));
            lblBrightnessVal.setText(String.format(Locale.ROOT, "%.0f", brightnessSlider.getValue()));
            lblGammaVal.setText(String.format(Locale.ROOT, "%.2f", gammaSlider.getValue()));
            debounceTimer.playFromStart();
        };

        contrastSlider.valueProperty().addListener((o, a, b) -> onSliderChanged.run());
        brightnessSlider.valueProperty().addListener((o, a, b) -> onSliderChanged.run());
        gammaSlider.valueProperty().addListener((o, a, b) -> onSliderChanged.run());
        noiseBox.setOnAction(e -> onSliderChanged.run());

        btnAutoCanonical.setOnAction(e -> {
            debounceTimer.stop();
            long gen = previewGeneration.incrementAndGet();
            previewExecutor.submit(() -> {
                if (previewGeneration.get() != gen) return;
                BufferedImage auto = ImageEnhancer.autoEnhance(previewSource);
                Image fxAuto = ImageFxUtils.toFxImage(auto);
                Platform.runLater(() -> {
                    if (previewGeneration.get() == gen) {
                        enhancedView.setImage(fxAuto);
                        contrastSlider.setValue(1.0);
                        brightnessSlider.setValue(0.0);
                        gammaSlider.setValue(1.0);
                        noiseBox.getSelectionModel().select(0);
                    }
                });
            });
        });

        btnResetOriginal.setOnAction(e -> {
            debounceTimer.stop();
            contrastSlider.setValue(1.0);
            brightnessSlider.setValue(0.0);
            gammaSlider.setValue(1.0);
            noiseBox.getSelectionModel().select(0);
            enhancedView.setImage(originalFx);
            if (appState != null) {
                appState.resetWorkingImageToOriginal();
                lblActiveStatus.setText("Current Active Working Image: Original (Unenhanced)");
            }
        });

        btnApplyEnhancement.setOnAction(e -> {
            if (appState == null) {
                Alert a = new Alert(Alert.AlertType.INFORMATION, "Enhancement preview applied in standalone mode.");
                a.showAndWait();
                return;
            }
            int noiseKernel = switch (noiseBox.getSelectionModel().getSelectedIndex()) {
                case 1 -> 3;
                case 2 -> 5;
                default -> 0;
            };
            double c = contrastSlider.getValue();
            double b = brightnessSlider.getValue();
            double g = gammaSlider.getValue();
            String noiseName = noiseBox.getSelectionModel().getSelectedItem();
            String desc = String.format(Locale.ROOT, "Manual (Contrast=%.2f, Bright=%.0f, Gamma=%.2f, Noise=%s)",
                c, b, g, noiseName);

            btnApplyEnhancement.setDisable(true);
            btnApplyEnhancement.setText("Applying Enhancement...");

            Task<BufferedImage> applyTask = new Task<>() {
                @Override
                protected BufferedImage call() {
                    return isSlotScope
                        ? ImageEnhancer.adjust(sourceImage, c, b, g, noiseKernel)
                        : ImageEnhancer.adjust(source, c, b, g, noiseKernel);
                }
            };

            applyTask.setOnSucceeded(ev -> {
                BufferedImage fullWorkingImage = applyTask.getValue();
                appState.applyWorkingImageEnhancement(fullWorkingImage, desc);
                lblActiveStatus.setText("Current Active Working Image: " + desc);
                btnApplyEnhancement.setDisable(false);
                btnApplyEnhancement.setText("Apply Enhancement to Active Image");

                Alert confirm = new Alert(Alert.AlertType.INFORMATION,
                    "Applied enhancement to active working image!\n\nAll subsequent IVP processing, slot crops, and 8 IVP features will be computed from this enhanced image.\nOriginal photograph remains preserved for ONNX vehicle detection.");
                if (dialog.getOwner() != null) confirm.initOwner(dialog.getOwner());
                confirm.setHeaderText("Enhancement Applied");
                confirm.showAndWait();
            });

            applyTask.setOnFailed(ev -> {
                btnApplyEnhancement.setDisable(false);
                btnApplyEnhancement.setText("Apply Enhancement to Active Image");
                Alert err = new Alert(Alert.AlertType.ERROR, "Failed to apply enhancement: "
                    + (applyTask.getException() != null ? applyTask.getException().getMessage() : "Unknown error"));
                if (dialog.getOwner() != null) err.initOwner(dialog.getOwner());
                err.showAndWait();
            });

            Thread applyThread = new Thread(applyTask, "EnhancementApplyWorker");
            applyThread.setDaemon(true);
            applyThread.start();
        });

        // Trigger initial preview
        requestPreviewComputation.run();

        VBox content = new VBox(12, lblScope, lblActiveStatus, previewBox, grid, actionsBar, note);
        content.setPadding(new Insets(14));
        content.setStyle("-fx-background-color: #0f172a;");

        ScrollPane scroll = new ScrollPane(content);
        scroll.setFitToWidth(true);
        scroll.setStyle("-fx-background: #0f172a; -fx-border-color: #0f172a;");
        pane.setContent(scroll);
        pane.setPrefSize(1080, 800);
        dialog.showAndWait();
    }

    public static BufferedImage createPreviewScale(BufferedImage src, int maxW, int maxH) {
        if (src == null) return null;
        int w = src.getWidth();
        int h = src.getHeight();
        if (w <= maxW && h <= maxH) return src;
        double scale = Math.min((double) maxW / w, (double) maxH / h);
        int targetW = Math.max(1, (int) Math.round(w * scale));
        int targetH = Math.max(1, (int) Math.round(h * scale));
        BufferedImage scaled = new BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = scaled.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.drawImage(src, 0, 0, targetW, targetH, null);
        g2.dispose();
        return scaled;
    }

    private static void addControlRow(GridPane grid, int row, String name, Slider slider, Label valLabel, String explanation) {
        Label lblName = new Label(name);
        lblName.setStyle("-fx-font-weight: bold; -fx-text-fill: #f8fafc; -fx-font-size: 12px;");
        Label lblDesc = new Label(explanation);
        lblDesc.setStyle("-fx-text-fill: #94a3b8; -fx-font-size: 11px;");
        lblDesc.setWrapText(true);
        VBox nameBox = new VBox(2, lblName, lblDesc);
        nameBox.setPrefWidth(280);

        slider.setPrefWidth(500);
        slider.setStyle("-fx-control-inner-background: #334155;");

        grid.addRow(row, nameBox, slider, valLabel);
    }

    private static Label createValueLabel(String initialText) {
        Label l = new Label(initialText);
        l.setStyle("-fx-font-weight: bold; -fx-text-fill: #38bdf8; -fx-font-size: 13px; -fx-min-width: 50px;");
        return l;
    }

    private static Slider slider(double min, double max, double value) {
        Slider slider = new Slider(min, max, value);
        slider.setShowTickMarks(true);
        slider.setShowTickLabels(true);
        slider.setMajorTickUnit((max - min) / 4.0);
        slider.setBlockIncrement((max - min) / 20.0);
        return slider;
    }
}
