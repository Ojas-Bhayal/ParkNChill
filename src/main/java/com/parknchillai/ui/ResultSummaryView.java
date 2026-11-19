package com.parknchillai.ui;

import com.parknchillai.app.AnalysisSummary;
import com.parknchillai.app.AppState;
import com.parknchillai.parking.SlotStatus;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Label;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;

import java.util.Locale;

/**
 * Compact Occupancy Summary (ParkNChill v2.3 Requirement 2.C):
 * <ul>
 *   <li>Displays exactly one compact summary with 4 metrics:
 *       {@code Total Slots}, {@code Occupied}, {@code Available}, and {@code Needs Review}.</li>
 *   <li>Distinguishes provisional detector suggestions from human-confirmed labels.</li>
 *   <li>Never displays {@code 0 Occupied} and {@code 0 Available} as if they are confirmed
 *       outcomes when all slots are still awaiting review or before analysis has run.</li>
 * </ul>
 */
public class ResultSummaryView extends VBox {

    private final Label lblTotal = new Label("0");
    private final Label lblOccupied = new Label("—");
    private final Label lblAvailable = new Label("—");
    private final Label lblNeedsReview = new Label("—");
    private final Label lblProvisionalVsConfirmedNote = new Label(
        "Awaiting detection. Load an image and click 'Detect and Analyse' to evaluate parking slots."
    );
    private final Label lblAggregateSummary = new Label("");

    public ResultSummaryView() {
        setSpacing(6);
        setPadding(new Insets(10, 12, 10, 12));
        getStyleClass().add("summary-card-container");

        HBox header = new HBox(10);
        header.setAlignment(Pos.CENTER_LEFT);
        Label title = new Label("Occupancy Summary");
        title.getStyleClass().add("card-section-title");
        HBox.setHgrow(title, Priority.ALWAYS);

        lblProvisionalVsConfirmedNote.setStyle("-fx-font-size: 12px; -fx-text-fill: #cbd5e1;");
        lblProvisionalVsConfirmedNote.setWrapText(true);
        Tooltip.install(lblProvisionalVsConfirmedNote, new Tooltip(
            "Provisional suggestions are produced automatically by the vehicle detector.\n" +
            "Confirmed labels have been explicitly verified by a human reviewer in Step 3 for Decision Tree training."
        ));

        lblAggregateSummary.setStyle("-fx-font-size: 11.5px; -fx-font-weight: bold; -fx-text-fill: #38bdf8; "
            + "-fx-background-color: #0f172a; -fx-padding: 6px 10px; -fx-background-radius: 5px; -fx-border-color: #334155; -fx-border-radius: 5px;");
        lblAggregateSummary.setWrapText(true);
        lblAggregateSummary.setVisible(false);

        header.getChildren().addAll(title);

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(8);

        VBox cardTotal = createKpiCard(
            "Total Slots", lblTotal, Color.web("#38bdf8"), "card-total",
            "Total localized parking spaces in the current image."
        );
        VBox cardOccupied = createKpiCard(
            "Occupied", lblOccupied, Color.web("#ef4444"), "card-occupied",
            "Parking spaces currently classified as Occupied (provisional detector suggestion or reviewer-confirmed)."
        );
        VBox cardAvailable = createKpiCard(
            "Available", lblAvailable, Color.web("#10b981"), "card-available",
            "Parking spaces currently classified as Available (provisional detector suggestion or reviewer-confirmed)."
        );
        VBox cardReview = createKpiCard(
            "Needs Review", lblNeedsReview, Color.web("#f59e0b"), "card-total",
            "Ambiguous or unconfirmed spaces requiring human inspection in Step 3."
        );

        GridPane.setHgrow(cardTotal, Priority.ALWAYS);
        GridPane.setHgrow(cardOccupied, Priority.ALWAYS);
        GridPane.setHgrow(cardAvailable, Priority.ALWAYS);
        GridPane.setHgrow(cardReview, Priority.ALWAYS);

        grid.add(cardTotal, 0, 0);
        grid.add(cardOccupied, 1, 0);
        grid.add(cardAvailable, 2, 0);
        grid.add(cardReview, 3, 0);

        getChildren().addAll(header, grid, lblProvisionalVsConfirmedNote, lblAggregateSummary);
    }

    private VBox createKpiCard(String title, Label valueLabel, Color indicatorColor, String styleClass, String tooltipText) {
        VBox card = new VBox(2);
        card.setPadding(new Insets(6, 10, 6, 10));
        card.getStyleClass().addAll("kpi-card", styleClass);

        HBox top = new HBox(6);
        top.setAlignment(Pos.CENTER_LEFT);

        Circle dot = new Circle(4, indicatorColor);
        Label titleLabel = new Label(title);
        titleLabel.getStyleClass().add("kpi-title");
        top.getChildren().addAll(dot, titleLabel);

        valueLabel.getStyleClass().add("kpi-value");
        Tooltip.install(card, new Tooltip(tooltipText));

        card.getChildren().addAll(top, valueLabel);
        return card;
    }

    /**
     * Updates the compact summary from an {@link AnalysisSummary} (without reviewer state).
     */
    public void update(AnalysisSummary summary) {
        update(summary, null);
    }

    /**
     * Updates the compact summary from the current {@link AnalysisSummary} and {@link AppState},
     * distinguishing provisional detector suggestions from human-confirmed labels and never
     * presenting {@code 0 Occupied / 0 Available} as confirmed outcomes when all slots are awaiting review.
     */
    public void update(AnalysisSummary summary, AppState appState) {
        if (summary == null) {
            int configuredSlots = (appState != null && appState.getCurrentSlots() != null)
                ? appState.getCurrentSlots().size()
                : 0;
            lblTotal.setText(String.valueOf(configuredSlots));
            lblOccupied.setText("—");
            lblAvailable.setText("—");
            lblNeedsReview.setText(configuredSlots > 0 ? String.valueOf(configuredSlots) : "—");
            lblProvisionalVsConfirmedNote.setText(
                configuredSlots > 0
                    ? ("Image loaded with " + configuredSlots + " localized slot(s). Click 'Detect and Analyse' to determine occupancy.")
                    : "No image analyzed yet. Load a parking lot image to begin."
            );
            return;
        }

        int total = summary.getTotalSlots();
        int occupied = 0;
        int available = 0;
        int needsReview = 0;

        for (var r : summary.getSlotResults()) {
            SlotStatus st = r.getSlot().getStatus() != null ? r.getSlot().getStatus() : r.getStatus();
            if (st == SlotStatus.OCCUPIED) occupied++;
            else if (st == SlotStatus.AVAILABLE) available++;
            else needsReview++;
        }

        int confirmedCount = appState != null ? appState.getConfirmedSlotCountForCurrentImage() : 0;

        lblTotal.setText(String.valueOf(total));
        lblNeedsReview.setText(String.valueOf(needsReview));

        if (total > 0 && occupied == 0 && available == 0 && needsReview == total && confirmedCount == 0) {
            // Requirement 2.C: Do not display 0 occupied and 0 available as if these are confirmed outcomes
            // when all slots are still awaiting review!
            lblOccupied.setText("Awaiting Review");
            lblAvailable.setText("Awaiting Review");
            lblProvisionalVsConfirmedNote.setText(String.format(Locale.ROOT,
                "All %d slot(s) are awaiting human review (0 confirmed labels). "
                + "Select slots in Step 3 to mark them Occupied or Available.",
                total
            ));
        } else {
            String suffix = (confirmedCount == total && total > 0) ? " (Confirmed)" : " (Provisional)";
            lblOccupied.setText(occupied + (confirmedCount > 0 && confirmedCount < total ? "" : ""));
            lblAvailable.setText(String.valueOf(available));
            lblProvisionalVsConfirmedNote.setText(String.format(Locale.ROOT,
                "Current State%s: %d Occupied, %d Available, %d Needs Review | Reviewer-Confirmed Labels: %d of %d slot(s).",
                suffix, occupied, available, needsReview, confirmedCount, total
            ));
        }

        if (appState != null) {
            int imgCount = appState.getTotalImagesAnalysed();
            int totalSlotsAll = appState.getTotalSlotsCollected();
            int totalOccAll = appState.getTotalOccupiedSlots();
            int totalAvailAll = appState.getTotalAvailableSlots();
            int totalVerifiedAll = appState.getEligibleReviewedRecords().size();
            lblAggregateSummary.setText(String.format(Locale.ROOT,
                "Active Dataset: %d image(s) analysed | %d total slots across dataset (%d occupied, %d available | %d human-verified)",
                imgCount, totalSlotsAll, totalOccAll, totalAvailAll, totalVerifiedAll));
            lblAggregateSummary.setVisible(true);
        } else {
            lblAggregateSummary.setVisible(false);
        }
    }

    public String getTotalText()       { return lblTotal.getText(); }
    public String getOccupiedText()    { return lblOccupied.getText(); }
    public String getAvailableText()   { return lblAvailable.getText(); }
    public String getNeedsReviewText() { return lblNeedsReview.getText(); }
    public String getSummaryNoteText() { return lblProvisionalVsConfirmedNote.getText(); }
}
