package com.parknchillai.utils;

import com.parknchillai.parking.ParkingSlot;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Loads and saves project configuration data (slot definitions) in CSV text files.
 */
public final class ConfigLoader {

    private ConfigLoader() { /* utility class */ }

    /**
     * Parses a slot-definition CSV file.
     *
     * <p>Expected format — one slot per line, optional header row:
     * <pre>{@code id,x,y,width,height}</pre>
     *
     * Lines beginning with {@code #} and blank lines are skipped.
     * Duplicate slot IDs and empty slot configurations are rejected.
     *
     * @param csvPath path to the slot CSV file
     * @return list of {@link ParkingSlot} objects in file order
     * @throws IllegalArgumentException if the file cannot be read, is empty, has duplicate IDs, or a line is malformed
     */
    public static List<ParkingSlot> loadSlotsFromCsv(String csvPath) {
        if (csvPath == null || csvPath.isBlank()) {
            throw new IllegalArgumentException("Slot CSV path must not be null or blank.");
        }
        List<ParkingSlot> slots = new ArrayList<>();
        Set<Integer> seenIds = new HashSet<>();

        try (BufferedReader reader = Files.newBufferedReader(Path.of(csvPath), StandardCharsets.UTF_8)) {
            String line;
            int lineNum = 0;
            while ((line = reader.readLine()) != null) {
                lineNum++;
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;

                // Skip optional CSV header row (e.g. "id,x,y,width,height" or "slot_id,...")
                String lower = line.toLowerCase(Locale.ROOT);
                if (lower.startsWith("id,") || lower.startsWith("slot")) continue;

                String[] parts = line.split(",");
                if (parts.length != 5) {
                    throw new IllegalArgumentException(
                        String.format("Slot CSV line %d: expected 5 fields (id,x,y,w,h), got %d: '%s'",
                            lineNum, parts.length, line));
                }
                try {
                    int id = Integer.parseInt(parts[0].trim());
                    int x  = Integer.parseInt(parts[1].trim());
                    int y  = Integer.parseInt(parts[2].trim());
                    int w  = Integer.parseInt(parts[3].trim());
                    int h  = Integer.parseInt(parts[4].trim());
                    if (!seenIds.add(id)) {
                        throw new IllegalArgumentException(
                            "Slot CSV line " + lineNum + ": duplicate slot ID " + id);
                    }
                    slots.add(new ParkingSlot(id, x, y, w, h));
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                        "Slot CSV line " + lineNum + ": non-integer field in '" + line + "'", e);
                }
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Cannot read slot CSV file: " + csvPath, e);
        }

        if (slots.isEmpty()) {
            throw new IllegalArgumentException("Slot CSV file contains no valid slot definitions: " + csvPath);
        }
        return slots;
    }

    /**
     * Writes a list of {@link ParkingSlot} definitions to a CSV file in {@code id,x,y,width,height} format.
     *
     * @param csvPath destination file path
     * @param slots   non-empty list of slots to persist
     */
    public static void saveSlotsToCsv(String csvPath, List<ParkingSlot> slots) {
        if (csvPath == null || csvPath.isBlank()) {
            throw new IllegalArgumentException("Slot CSV path must not be null or blank.");
        }
        if (slots == null || slots.isEmpty()) {
            throw new IllegalArgumentException("Slots list must not be null or empty when saving CSV.");
        }

        Path path = Path.of(csvPath);
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
                writer.write("# ParkNChill Slot Configuration (id,x,y,width,height)\n");
                writer.write("id,x,y,width,height\n");
                for (ParkingSlot s : slots) {
                    writer.write(String.format(Locale.ROOT, "%d,%d,%d,%d,%d%n",
                        s.getId(), s.getX(), s.getY(), s.getWidth(), s.getHeight()));
                }
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Cannot write slot CSV file: " + csvPath, e);
        }
    }
}
