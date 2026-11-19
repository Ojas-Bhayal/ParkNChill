package com.parknchillai.ui;

import com.parknchillai.features.FeatureExtractor;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.RoiExtractor;
import com.parknchillai.imageprocessing.EnhancedImage;
import com.parknchillai.imageprocessing.ImageEnhancementService;
import com.parknchillai.imageprocessing.OriginalImage;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.image.ImageView;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Window;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Locale;

/**
 * A simple, user-visible explanation of the actual IVP pipeline used by ParkNChill.
 * It can be opened with no slots selected (full-image overview) or with a selected slot
 * (the same ROI used for per-slot feature extraction).
 */
public final class IvpStagesDialog {
    private static final int FULL_IMAGE_MAX_DIMENSION = 640;
    private static final int CARD_WIDTH = 220;
    private static final int CARD_IMAGE_HEIGHT = 126;

    private record Stage(String title, BufferedImage image, String explanation) {}

    private IvpStagesDialog() { }

    public static void show(Window owner, BufferedImage fullImage, ParkingSlot selectedSlot,
                            boolean useAdaptiveThreshold) {
        show(owner, fullImage, null, selectedSlot, useAdaptiveThreshold);
    }

    public static void show(Window owner, BufferedImage rawImage, BufferedImage workingImage,
                            ParkingSlot selectedSlot, boolean useAdaptiveThreshold) {
        if (rawImage == null) {
            javafx.scene.control.Alert alert = new javafx.scene.control.Alert(
                javafx.scene.control.Alert.AlertType.INFORMATION,
                "Load a parking image first, then choose View IVP Stages.");
            if (owner != null) alert.initOwner(owner);
            alert.setHeaderText("No image loaded");
            alert.showAndWait();
            return;
        }

        BufferedImage rawRegion;
        BufferedImage enhancedRegion;
        String scope;
        try {
            BufferedImage enhancedFull = workingImage != null
                ? workingImage
                : ImageEnhancementService.getDefault().enhance(OriginalImage.of(rawImage)).image();
            if (selectedSlot != null) {
                rawRegion = RoiExtractor.extract(rawImage, selectedSlot);
                enhancedRegion = RoiExtractor.extract(enhancedFull, selectedSlot);
                scope = "Selected parking slot #" + selectedSlot.getId()
                    + " — feature values below are computed from this slot crop.";
            } else {
                rawRegion = resizeForOverview(rawImage, FULL_IMAGE_MAX_DIMENSION);
                enhancedRegion = resizeForOverview(enhancedFull, FULL_IMAGE_MAX_DIMENSION);
                scope = "Full-image overview (resized for display). For slot-level features, select a parking-space row first.";
            }
        } catch (RuntimeException ex) {
            javafx.scene.control.Alert alert = new javafx.scene.control.Alert(
                javafx.scene.control.Alert.AlertType.ERROR,
                "Could not extract this region:\n" + ex.getMessage());
            if (owner != null) alert.initOwner(owner);
            alert.setHeaderText("IVP preview failed");
            alert.showAndWait();
            return;
        }

        try {
            FeatureExtractor.ProcessedRoi processed = FeatureExtractor.processWithStages(rawRegion, enhancedRegion, useAdaptiveThreshold);
            FeatureVector vector = processed.getFeatures();
            List<FeatureVector.FeatureSpec> specs = FeatureVector.SPECS;
            double[] values = vector.toArray(FeatureVector.FEATURE_COUNT);

            List<Stage> stages = List.of(
                new Stage("1. Original image / ROI", rawRegion,
                    "Untouched input pixels. The original scene is retained for vehicle detection."),
                new Stage("2. Enhanced image / ROI (Canonical: ivp-enhance-v1)", processed.getEnhanced(),
                    "Canonical auto-enhanced copy. Contrast stretched and gamma-corrected. Numerical features are computed strictly from these pixels."),
                new Stage("3. Grayscale (from Enhanced)", processed.getGray(),
                    "Converts the enhanced region into BT.709 luminance values."),
                new Stage("4. HSV saturation (from Enhanced)", processed.getSaturation(),
                    "Extracts chromatic saturation from the enhanced region (S > 90 indicates vivid paint)."),
                new Stage("5. Gaussian blur", processed.getSmoothed(),
                    "Reduces high-frequency noise from the enhanced grayscale prior to thresholding."),
                new Stage("6. Threshold mask", processed.getBinary(),
                    "Separates foreground pixels from the background."),
                new Stage("7. Morphological closing", processed.getClosed(),
                    "3×3 morphological closing to bridge internal vehicle body gaps."),
                new Stage("8. Largest connected region", processed.getLargestBlobMask(),
                    "Shows the largest 8-connected contiguous foreground component."),
                new Stage("9. Sobel edges", processed.getEdges(),
                    "3×3 Sobel gradient magnitude on smoothed enhanced grayscale.")
            );

            Dialog<Void> dialog = new Dialog<>();
            dialog.setTitle("ParkNChill — Image Processing (IVP) Stages");
            dialog.setHeaderText("What IVP does: converts each image region into visual stages and numerical features.");
            if (owner != null) dialog.initOwner(owner);
            dialog.setResizable(true);
            dialog.getDialogPane().getButtonTypes().add(ButtonType.CLOSE);

            VBox root = new VBox(10);
            root.setPadding(new Insets(10));

            Label scopeLabel = new Label(scope);
            scopeLabel.setWrapText(true);
            scopeLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: #075985; -fx-background-color: #e0f2fe; -fx-padding: 8px; -fx-background-radius: 5px;");

            Label pipelineText = new Label("The actual enhancement shown below is applied to each slot crop before the eight image features are calculated. YOLOX still receives the untouched original scene; its two detection features are added to the eight IVP features for the Decision Tree.");
            pipelineText.setWrapText(true);
            pipelineText.setStyle("-fx-text-fill: #334155; -fx-padding: 2px 0 4px 0;");

            GridPane grid = new GridPane();
            grid.setHgap(10);
            grid.setVgap(10);
            for (int i = 0; i < stages.size(); i++) {
                Stage stage = stages.get(i);
                VBox card = createStageCard(stage);
                grid.add(card, i % 4, i / 4);
            }

            StringBuilder featureText = new StringBuilder("Eight image-processing features (calculated from the enhanced ROI):\n");
            for (int i = 0; i < FeatureVector.FEATURE_COUNT; i++) {
                featureText.append(String.format(Locale.ROOT, "• %s: %s%n", specs.get(i).displayName(),
                    (i == FeatureVector.IDX_MEAN_INTENSITY || i == FeatureVector.IDX_MEAN_SATURATION)
                        ? String.format(Locale.ROOT, "%.2f", values[i])
                        : String.format(Locale.ROOT, "%.4f", values[i])));
            }
            Label features = new Label(featureText.toString());
            features.setWrapText(true);
            features.setStyle("-fx-font-family: 'Consolas', monospace; -fx-font-size: 11px; -fx-background-color: #f1f5f9; -fx-text-fill: #0f172a; -fx-padding: 10px; -fx-background-radius: 5px;");

            Label lastNote = new Label("Note: the two extra v2.3 features (vehicle overlap and vehicle matched) come from AI vehicle detections matched to a parking-slot boundary. They are not generated by this IVP-only preview.");
            lastNote.setWrapText(true);
            lastNote.setStyle("-fx-text-fill: #92400e; -fx-background-color: #fef3c7; -fx-padding: 8px; -fx-background-radius: 5px;");

            root.getChildren().addAll(scopeLabel, pipelineText, grid, features, lastNote);
            ScrollPane scroll = new ScrollPane(root);
            scroll.setFitToWidth(true);
            scroll.setPrefViewportWidth(990);
            scroll.setPrefViewportHeight(620);
            dialog.getDialogPane().setContent(scroll);
            dialog.getDialogPane().setPrefSize(1030, 730);
            dialog.showAndWait();
        } catch (RuntimeException ex) {
            javafx.scene.control.Alert alert = new javafx.scene.control.Alert(
                javafx.scene.control.Alert.AlertType.ERROR,
                "The IVP pipeline could not process this image region:\n" + ex.getMessage());
            if (owner != null) alert.initOwner(owner);
            alert.setHeaderText("IVP processing error");
            alert.showAndWait();
        }
    }

    private static VBox createStageCard(Stage stage) {
        VBox card = new VBox(6);
        card.setPrefWidth(CARD_WIDTH);
        card.setMinWidth(CARD_WIDTH);
        card.setPadding(new Insets(8));
        card.setStyle("-fx-background-color: #f8fafc; -fx-border-color: #cbd5e1; -fx-border-radius: 6px; -fx-background-radius: 6px;");

        Label title = new Label(stage.title());
        title.setWrapText(true);
        title.setStyle("-fx-font-weight: bold; -fx-text-fill: #0f172a; -fx-font-size: 12px;");

        ImageView view = new ImageView(ImageFxUtils.toFxImage(stage.image()));
        view.setFitWidth(CARD_WIDTH - 16);
        view.setFitHeight(CARD_IMAGE_HEIGHT);
        view.setPreserveRatio(true);
        view.setSmooth(true);

        VBox imageHolder = new VBox(view);
        imageHolder.setAlignment(Pos.CENTER);
        imageHolder.setMinHeight(CARD_IMAGE_HEIGHT);
        imageHolder.setStyle("-fx-background-color: #e2e8f0; -fx-background-radius: 4px;");
        VBox.setVgrow(imageHolder, Priority.NEVER);

        Label description = new Label(stage.explanation());
        description.setWrapText(true);
        description.setStyle("-fx-text-fill: #334155; -fx-font-size: 10.5px;");
        card.getChildren().addAll(title, imageHolder, description);
        return card;
    }

    private static BufferedImage resizeForOverview(BufferedImage image, int maxDimension) {
        int width = image.getWidth();
        int height = image.getHeight();
        int longest = Math.max(width, height);
        if (longest <= maxDimension) return image;
        double scale = (double) maxDimension / longest;
        int targetW = Math.max(1, (int) Math.round(width * scale));
        int targetH = Math.max(1, (int) Math.round(height * scale));
        BufferedImage resized = new BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g.drawImage(image, 0, 0, targetW, targetH, null);
        } finally {
            g.dispose();
        }
        return resized;
    }
}
