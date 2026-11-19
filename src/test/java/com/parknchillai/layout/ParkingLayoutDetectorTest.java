package com.parknchillai.layout;

import com.parknchillai.app.AnalysisSummary;
import com.parknchillai.app.ParkingAnalysisService;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import com.parknchillai.utils.ConfigLoader;
import com.parknchillai.utils.SampleDataGenerator;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ParkingLayoutDetectorTest {

    @Test
    void autoDetectsTwelveSlotsOnDemoParkingLotWithHighIoU() throws Exception {
        File demoFile = SampleDataGenerator.ensureSampleImageExists(SampleDataGenerator.DEFAULT_SAMPLE_IMAGE_PATH);
        BufferedImage demoImg = ImageIO.read(demoFile);
        List<ParkingSlot> gtSlots = ConfigLoader.loadSlotsFromCsv("data/config/slots.csv");

        LayoutDetectionResult layout = ParkingLayoutDetector.detectLayout(demoImg);
        assertNotNull(layout);
        assertEquals(3, layout.getRows().size(), "Demo parking lot should have 3 detected rows");
        assertEquals(12, layout.getDetectedSlotCount(), "Demo parking lot should auto-detect 12 bays");

        double sumIoU = 0.0;
        for (int i = 0; i < 12; i++) {
            ParkingSlot s = layout.getSlots().get(i);
            assertTrue(s.getX() >= 0 && s.getY() >= 0);
            assertTrue(s.getX() + s.getWidth() <= demoImg.getWidth());
            assertTrue(s.getY() + s.getHeight() <= demoImg.getHeight());

            double iou = computeIoU(gtSlots.get(i), s);
            assertTrue(iou >= 0.95,
                "Auto-detected Slot #" + s.getId() + " must achieve IoU >= 0.95 against slots.csv; got " + iou);
            sumIoU += iou;
        }
        assertTrue(sumIoU / 12.0 >= 0.96, "Mean IoU on demo lot must be >= 0.96");
    }

    @Test
    void autoDetectsAndClassifiesCustomMultiRowParkingLotImage() {
        BufferedImage customLot = new BufferedImage(520, 280, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = customLot.createGraphics();
        g.setColor(new Color(52, 56, 64));
        g.fillRect(0, 0, 520, 280);

        int[] rowY = {15, 155};
        int bayW = 86;
        int bayH = 100;

        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 5; c++) {
                int x = 15 + c * 98;
                int y = rowY[r];
                g.setColor(new Color(235, 235, 240));
                g.setStroke(new BasicStroke(2f));
                g.drawRect(x, y, bayW, bayH);

                if ((r == 0 && c == 1) || (r == 1 && c == 3)) {
                    g.setColor(r == 0 ? new Color(210, 45, 55) : new Color(40, 120, 225));
                    g.fillRoundRect(x + 12, y + 12, bayW - 24, bayH - 24, 10, 10);
                    g.setColor(new Color(25, 35, 50));
                    g.fillRect(x + 18, y + 24, bayW - 36, 18);
                    g.fillRect(x + 18, y + bayH - 42, bayW - 36, 16);
                }
            }
        }
        g.dispose();

        LayoutDetectionResult layout = ParkingLayoutDetector.detectLayout(customLot);
        assertEquals(2, layout.getRows().size(), "Should detect 2 horizontal parking rows");
        assertEquals(10, layout.getDetectedSlotCount(), "Should auto-detect 10 parking bays (2x5)");

        ParkingAnalysisService service = ParkingAnalysisService.createWithDefaultModel();
        AnalysisSummary summary = service.analyze(customLot, layout.getSlots(), false, com.parknchillai.detection.OccupancyDecisionEngine.AnalysisMode.FEATURE_DECISION_TREE);
        assertEquals(10, summary.getTotalSlots());
        assertEquals(2, summary.getOccupiedSlots(), "Exactly 2 bays contain vehicles");
        assertEquals(8, summary.getAvailableSlots(), "Exactly 8 bays are empty");
        assertEquals(SlotStatus.OCCUPIED, summary.getSlotResults().get(1).getStatus());
        assertEquals(SlotStatus.OCCUPIED, summary.getSlotResults().get(8).getStatus());
    }

    @Test
    void autoDetectsSlotsOnLightConcreteLotWithDarkPaintedLinesAndHighIoU() {
        BufferedImage lightConcreteLot = new BufferedImage(400, 260, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = lightConcreteLot.createGraphics();
        g.setColor(new Color(210, 212, 215)); // bright concrete pavement (>= 195)
        g.fillRect(0, 0, 400, 260);
        g.setColor(new Color(40, 42, 45)); // dark painted stall divider lines
        g.setStroke(new BasicStroke(3f));
        for (int r = 0; r < 2; r++) {
            int y = 20 + r * 120;
            for (int c = 0; c < 4; c++) {
                int x = 20 + c * 90;
                g.drawRect(x, y, 80, 100);
            }
        }
        g.dispose();

        LayoutDetectionResult layout = ParkingLayoutDetector.detectLayout(lightConcreteLot);
        assertEquals(2, layout.getRows().size(), "Light concrete lot should detect 2 horizontal rows");
        assertEquals(8, layout.getDetectedSlotCount(), "Light concrete lot should detect 8 bays (2x4)");

        double sumIoU = 0.0;
        int idx = 0;
        for (int r = 0; r < 2; r++) {
            int gy = 20 + r * 120;
            for (int c = 0; c < 4; c++) {
                int gx = 20 + c * 90;
                ParkingSlot gt = new ParkingSlot(idx + 1, gx, gy, 80, 100);
                double iou = computeIoU(gt, layout.getSlots().get(idx));
                assertTrue(iou >= 0.90, "Light concrete Slot #" + (idx + 1) + " IoU must be >= 0.90; got " + iou);
                sumIoU += iou;
                idx++;
            }
        }
        assertTrue(sumIoU / 8.0 >= 0.92, "Mean IoU on light concrete lot must be >= 0.92");
    }

    @Test
    void generateRowGridPartitionsRowIntoExactBayCount() {
        List<ParkingSlot> bays = SlotGenerator.generateRowGrid(10, 20, 30, 400, 90, 5, 4);
        assertEquals(5, bays.size());
        assertEquals(10, bays.get(0).getId());
        assertEquals(14, bays.get(4).getId());
        assertEquals(20, bays.get(0).getX());
        assertEquals(20 + 400, bays.get(4).getX() + bays.get(4).getWidth());
        for (ParkingSlot s : bays) {
            assertEquals(30, s.getY());
            assertEquals(90, s.getHeight());
            assertTrue(s.getWidth() >= 70);
        }
    }

    @Test
    void nullOrTinyImageThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> ParkingLayoutDetector.detectLayout(null));
        BufferedImage tiny = new BufferedImage(5, 5, BufferedImage.TYPE_BYTE_GRAY);
        assertThrows(IllegalArgumentException.class, () -> ParkingBoundaryDetector.detectBoundary(tiny));
    }

    private static double computeIoU(ParkingSlot a, ParkingSlot b) {
        int x1 = Math.max(a.getX(), b.getX());
        int y1 = Math.max(a.getY(), b.getY());
        int x2 = Math.min(a.getX() + a.getWidth(), b.getX() + b.getWidth());
        int y2 = Math.min(a.getY() + a.getHeight(), b.getY() + b.getHeight());
        int interW = Math.max(0, x2 - x1);
        int interH = Math.max(0, y2 - y1);
        double interArea = (double) interW * interH;
        double unionArea = (double) a.getWidth() * a.getHeight()
                         + (double) b.getWidth() * b.getHeight()
                         - interArea;
        return unionArea <= 0 ? 0.0 : interArea / unionArea;
    }
}
