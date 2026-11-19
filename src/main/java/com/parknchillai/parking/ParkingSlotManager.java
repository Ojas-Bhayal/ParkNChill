package com.parknchillai.parking;

import com.parknchillai.utils.ConfigLoader;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Manages the collection of {@link ParkingSlot} objects for a parking lot,
 * supporting CSV loading/saving, coordinate scaling after image normalization,
 * automatic layout replacement, and interactive GUI editing.
 */
public class ParkingSlotManager {

    private final List<ParkingSlot> slots = new ArrayList<>();

    /** Adds a slot to the managed collection, rejecting null or duplicate slot IDs. */
    public void addSlot(ParkingSlot slot) {
        if (slot == null) throw new IllegalArgumentException("Slot must not be null.");
        for (ParkingSlot existing : slots) {
            if (existing.getId() == slot.getId()) {
                throw new IllegalArgumentException("Duplicate slot ID: " + slot.getId());
            }
        }
        slots.add(slot);
    }

    /** Replaces all managed slots with a new list of slots (validating non-null and unique IDs). */
    public void setSlots(List<ParkingSlot> newSlots) {
        if (newSlots == null) throw new IllegalArgumentException("Slots list must not be null.");
        Set<Integer> seen = new HashSet<>();
        for (ParkingSlot s : newSlots) {
            if (s == null) throw new IllegalArgumentException("Slot entry must not be null.");
            if (!seen.add(s.getId())) {
                throw new IllegalArgumentException("Duplicate slot ID in list: " + s.getId());
            }
        }
        slots.clear();
        slots.addAll(newSlots);
    }

    /**
     * Scales all managed slots by {@code (scaleX, scaleY)} and clamps them to {@code (imageWidth, imageHeight)}
     * so slot coordinates calibrated for an unnormalized source image remain aligned after resolution normalization.
     */
    public void scaleAllSlots(double scaleX, double scaleY, int imageWidth, int imageHeight) {
        List<ParkingSlot> scaled = new ArrayList<>(slots.size());
        for (ParkingSlot s : slots) {
            scaled.add(s.scaleToImage(scaleX, scaleY, imageWidth, imageHeight));
        }
        slots.clear();
        slots.addAll(scaled);
    }

    /** Removes a slot with the specified ID, returning {@code true} if found and removed. */
    public boolean removeSlotById(int id) {
        return slots.removeIf(s -> s.getId() == id);
    }

    /** Removes all slots from the manager. */
    public void clearSlots() {
        slots.clear();
    }

    /** Returns the next sequential slot ID ({@code maxId + 1}, or {@code 1} if empty). */
    public int nextAvailableId() {
        int max = 0;
        for (ParkingSlot s : slots) {
            if (s.getId() > max) max = s.getId();
        }
        return max + 1;
    }

    /** Returns an unmodifiable view of all managed slots. */
    public List<ParkingSlot> getSlots() {
        return Collections.unmodifiableList(slots);
    }

    /** Returns the total number of managed slots. */
    public int getTotalCount() { return slots.size(); }

    /** Returns the number of slots currently classified as OCCUPIED. */
    public long getOccupiedCount() {
        return slots.stream()
                    .filter(s -> s.getStatus() == SlotStatus.OCCUPIED)
                    .count();
    }

    /** Returns the number of slots currently classified as AVAILABLE. */
    public long getAvailableCount() {
        return slots.stream()
                    .filter(s -> s.getStatus() == SlotStatus.AVAILABLE)
                    .count();
    }

    /** Returns the number of slots currently marked as NEEDS_REVIEW. */
    public long getNeedsReviewCount() {
        return slots.stream()
                    .filter(s -> s.getStatus() == SlotStatus.NEEDS_REVIEW)
                    .count();
    }

    /** Clears all classification results without removing slot definitions. */
    public void resetStatuses() {
        slots.forEach(s -> s.setStatus(null));
    }

    /**
     * Populates this manager from a CSV-formatted slot configuration file.
     *
     * @param csvPath project-relative or absolute path to the configuration CSV
     */
    public void loadFromCsv(String csvPath) {
        List<ParkingSlot> loaded = ConfigLoader.loadSlotsFromCsv(csvPath);
        slots.clear();
        slots.addAll(loaded);
    }

    /**
     * Saves all currently managed slots to a CSV configuration file.
     */
    public void saveToCsv(String csvPath) {
        ConfigLoader.saveSlotsToCsv(csvPath, slots);
    }
}
