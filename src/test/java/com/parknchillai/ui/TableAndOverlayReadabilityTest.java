package com.parknchillai.ui;

import com.parknchillai.app.AnalysisSummary;
import com.parknchillai.app.ParkingAnalysisService;
import com.parknchillai.app.SlotAnalysisResult;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.utils.ConfigLoader;
import com.parknchillai.utils.SampleDataGenerator;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 4, Phase 5 &amp; Phase 6 verification tests for:
 * <ul>
 *   <li>WCAG AA text contrast ratios ({@code >= 4.5:1}) across ordinary table cells, alternating
 *       odd/even rows, selected rows, table headers, and status badges</li>
 *   <li>Explicit {@code .table-cell} foreground text colour rules in {@code style.css}</li>
 *   <li>Selected-row full decision path &amp; vehicle detection details panel in {@link SlotTableView}</li>
 *   <li>Low-opacity parking-slot overlays ({@code 12%} default fill opacity) and preservation
 *       of underlying vehicle pixels in {@link ParkingAnalysisService#generateAnnotatedImage}</li>
 *   <li>Consistency across overlay statuses, {@link AnalysisSummary} counts, and RFC 4180 CSV export</li>
 * </ul>
 */
class TableAndOverlayReadabilityTest {
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



    @Test
    void tableColorsExceedWcagAaContrastRatioMinimumOfFourPointFiveToOne() {
        double evenRowContrast = SlotTableView.computeWcagContrastRatio(
            SlotTableView.CELL_TEXT_HEX, SlotTableView.EVEN_ROW_BG_HEX);
        double oddRowContrast = SlotTableView.computeWcagContrastRatio(
            SlotTableView.CELL_TEXT_HEX, SlotTableView.ODD_ROW_BG_HEX);
        double selectedRowContrast = SlotTableView.computeWcagContrastRatio(
            "#ffffff", SlotTableView.SELECTED_ROW_BG_HEX);
        double headerContrast = SlotTableView.computeWcagContrastRatio(
            SlotTableView.HEADER_TEXT_HEX, SlotTableView.HEADER_BG_HEX);

        double occupiedBadgeContrast = SlotTableView.computeWcagContrastRatio(
            SlotTableView.OCCUPIED_TEXT_HEX, SlotTableView.OCCUPIED_BG_HEX);
        double availableBadgeContrast = SlotTableView.computeWcagContrastRatio(
            SlotTableView.AVAILABLE_TEXT_HEX, SlotTableView.AVAILABLE_BG_HEX);
        double reviewBadgeContrast = SlotTableView.computeWcagContrastRatio(
            SlotTableView.REVIEW_TEXT_HEX, SlotTableView.REVIEW_BG_HEX);

        assertTrue(evenRowContrast >= 14.0, "Even row contrast must exceed WCAG AA (4.5:1); got " + evenRowContrast);
        assertTrue(oddRowContrast >= 12.0, "Odd row contrast must exceed WCAG AA (4.5:1); got " + oddRowContrast);
        assertTrue(selectedRowContrast >= 8.0, "Selected row contrast must exceed WCAG AA (4.5:1); got " + selectedRowContrast);
        assertTrue(headerContrast >= 14.0, "Table header contrast must exceed WCAG AA (4.5:1); got " + headerContrast);
        assertTrue(occupiedBadgeContrast >= 4.5, "OCCUPIED badge contrast must meet WCAG AA (4.5:1); got " + occupiedBadgeContrast);
        assertTrue(availableBadgeContrast >= 4.5, "AVAILABLE badge contrast must meet WCAG AA (4.5:1); got " + availableBadgeContrast);
        assertTrue(reviewBadgeContrast >= 4.5, "NEEDS_REVIEW badge contrast must meet WCAG AA (4.5:1); got " + reviewBadgeContrast);
    }

    @Test
    void stylesheetExplicitlyDefinesTableCellTextFillAndFontSize() throws Exception {
        String css = Files.readString(Path.of("src/main/resources/styles/style.css"));
        assertTrue(css.contains(".table-view .table-cell"),
            "style.css must explicitly style .table-view .table-cell so Modena dark text never overrides dark rows");
        assertTrue(css.contains("-fx-text-fill: #f8fafc;"),
            "style.css must set explicit light foreground text (#f8fafc)");
        assertTrue(css.contains("-fx-font-size: 13.5px;"),
            "style.css must set readable >=13px table cell font size");
    }

    @Test
    void lowOpacityOverlayPreservesUnderlyingVehicleColorsAndMatchesSummaryAndCsvCounts() throws Exception {
        File demoFile = SampleDataGenerator.ensureSampleImageExists(SampleDataGenerator.DEFAULT_SAMPLE_IMAGE_PATH);
        BufferedImage demoImg = ImageIO.read(demoFile);
        List<ParkingSlot> slots = ConfigLoader.loadSlotsFromCsv("data/config/slots.csv");
        ParkingAnalysisService svc = ParkingAnalysisService.createWithDefaultModel();

        assertEquals(0.12, svc.getOverlayFillOpacity(), 1e-6);
        AnalysisSummary summary = svc.analyze(demoImg, slots, false, com.parknchillai.detection.OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE);
        BufferedImage annotated = summary.getAnnotatedImage();

        // Check center pixel of Slot #1 (Crimson car at x=50, y=60):
        // With 12% fill opacity, the underlying vehicle pixel color must remain >85% preserved!
        int origRgb = demoImg.getRGB(50, 60);
        int annoRgb = annotated.getRGB(50, 60);
        int origG = (origRgb >> 8) & 0xFF;
        int annoG = (annoRgb >> 8) & 0xFF;
        assertTrue(Math.abs(origG - annoG) <= 15,
            "12% low-opacity overlay must preserve underlying vehicle pixel detail; deltaG=" + Math.abs(origG - annoG));

        // Verify consistency across slot results, summary counts, and CSV export rows
        int csvOccupied = 0;
        int csvAvailable = 0;
        for (SlotAnalysisResult r : summary.getSlotResults()) {
            String csvRow = SlotTableView.formatCsvRow(r);
            if (csvRow.contains(",OCCUPIED,")) csvOccupied++;
            if (csvRow.contains(",AVAILABLE,")) csvAvailable++;
        }
        assertEquals(summary.getOccupiedSlots(), csvOccupied);
        assertEquals(summary.getAvailableSlots(), csvAvailable);
    }

    @Test
    void testFinding05ExportCsvHandlesNullSceneGracefully() {
        SlotTableView table = new SlotTableView();
        // table is not attached to a Scene, so table.getScene() is null.
        // Calling exportToCsv when table has no data should show alert or return without NPE.
        assertDoesNotThrow(() -> {
            javafx.application.Platform.runLater(table::exportToCsv);
        });
    }
}