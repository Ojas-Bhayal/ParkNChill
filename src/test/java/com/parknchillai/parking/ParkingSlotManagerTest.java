package com.parknchillai.parking;

import com.parknchillai.utils.ConfigLoader;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ParkingSlotManagerTest {

    @Test
    void slotCountsReflectClassifiedStatusIncludingNeedsReview() {
        ParkingSlotManager manager = new ParkingSlotManager();
        ParkingSlot s1 = new ParkingSlot(1, 0, 0, 10, 10);
        ParkingSlot s2 = new ParkingSlot(2, 20, 0, 10, 10);
        ParkingSlot s3 = new ParkingSlot(3, 40, 0, 10, 10);

        manager.addSlot(s1);
        manager.addSlot(s2);
        manager.addSlot(s3);

        assertEquals(3, manager.getTotalCount());
        assertEquals(0, manager.getOccupiedCount());
        assertEquals(0, manager.getAvailableCount());
        assertEquals(0, manager.getNeedsReviewCount());

        s1.setStatus(SlotStatus.OCCUPIED);
        s2.setStatus(SlotStatus.AVAILABLE);
        s3.setStatus(SlotStatus.NEEDS_REVIEW);

        assertEquals(1, manager.getOccupiedCount());
        assertEquals(1, manager.getAvailableCount());
        assertEquals(1, manager.getNeedsReviewCount());

        manager.resetStatuses();
        assertEquals(0, manager.getOccupiedCount());
        assertEquals(0, manager.getAvailableCount());
        assertEquals(0, manager.getNeedsReviewCount());
        assertFalse(s1.isClassified());
    }

    @Test
    void scaleAllSlotsTransformsCoordinatesToNormalizedImageSpace() {
        ParkingSlotManager manager = new ParkingSlotManager();
        // Original coordinates for a 2400x1800 image (downscaled by 0.533333 to 1280x960)
        manager.addSlot(new ParkingSlot(1, 1500, 900, 150, 180, ParkingSlot.GeometryOrigin.MANUAL_DRAWN));
        double scale = 1280.0 / 2400.0;
        manager.scaleAllSlots(scale, scale, 1280, 960);

        ParkingSlot scaled = manager.getSlots().get(0);
        assertEquals(800, scaled.getX());
        assertEquals(480, scaled.getY());
        assertEquals(80, scaled.getWidth());
        assertEquals(96, scaled.getHeight());
        assertEquals(ParkingSlot.GeometryOrigin.MANUAL_DRAWN, scaled.getOrigin());
    }

    @Test
    void duplicateSlotIdThrowsException() {
        ParkingSlotManager manager = new ParkingSlotManager();
        manager.addSlot(new ParkingSlot(1, 0, 0, 10, 10));
        assertThrows(IllegalArgumentException.class,
            () -> manager.addSlot(new ParkingSlot(1, 20, 0, 10, 10)));
    }

    @Test
    void configLoaderSupportsHeaderRowAndRejectsDuplicateIds() throws IOException {
        Path headerCsv = Files.createTempFile("slots_header", ".csv");
        Files.writeString(headerCsv, "id,x,y,width,height\n1,10,10,80,100\n2,100,10,80,100\n");
        List<ParkingSlot> loaded = ConfigLoader.loadSlotsFromCsv(headerCsv.toString());
        assertEquals(2, loaded.size());
        assertEquals(ParkingSlot.GeometryOrigin.CSV_CONFIG, loaded.get(0).getOrigin());
        Files.deleteIfExists(headerCsv);

        Path dupCsv = Files.createTempFile("slots_dup", ".csv");
        Files.writeString(dupCsv, "1,10,10,80,100\n1,100,10,80,100\n");
        assertThrows(IllegalArgumentException.class,
            () -> ConfigLoader.loadSlotsFromCsv(dupCsv.toString()));
        Files.deleteIfExists(dupCsv);
    }

    @Test
    void saveAndReloadSlotsCsvRoundTripAndInteractiveEditing() throws IOException {
        ParkingSlotManager manager = new ParkingSlotManager();
        assertEquals(1, manager.nextAvailableId());
        manager.addSlot(new ParkingSlot(1, 12, 18, 75, 95));
        manager.addSlot(new ParkingSlot(2, 95, 18, 75, 95));
        assertEquals(3, manager.nextAvailableId());

        Path tempCsv = Files.createTempFile("saved_slots", ".csv");
        manager.saveToCsv(tempCsv.toString());

        ParkingSlotManager reloaded = new ParkingSlotManager();
        reloaded.loadFromCsv(tempCsv.toString());
        assertEquals(2, reloaded.getTotalCount());
        assertEquals(12, reloaded.getSlots().get(0).getX());
        assertEquals(95, reloaded.getSlots().get(1).getX());

        assertTrue(reloaded.removeSlotById(1));
        assertEquals(1, reloaded.getTotalCount());
        reloaded.clearSlots();
        assertEquals(0, reloaded.getTotalCount());

        Files.deleteIfExists(tempCsv);
    }

    @Test
    void negativeDimensionsThrowException() {
        assertThrows(IllegalArgumentException.class,
            () -> new ParkingSlot(1, -1, 0, 10, 10));
        assertThrows(IllegalArgumentException.class,
            () -> new ParkingSlot(1, 0, 0, -5, 10));
        assertThrows(IllegalArgumentException.class,
            () -> new ParkingSlot(1, 0, 0, 10, 0));
    }
}
