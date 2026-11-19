package com.parknchillai.parking;

/**
 * Represents the occupancy classification or review status of a single parking slot.
 *
 * <ul>
 *   <li>{@link #OCCUPIED} — Space contains a vehicle assigned to that space under the validated policy.</li>
 *   <li>{@link #AVAILABLE} — A valid parking space was detected and no vehicle occupies it under the validated policy.</li>
 *   <li>{@link #NEEDS_REVIEW} — Geometry, detection evidence, feature domain, or multi-slot assignment is ambiguous
 *       and must not be silently reported as Available.</li>
 *   <li>{@link #UNCLASSIFIED} — Detection or slot processing failed or has not been executed.</li>
 * </ul>
 */
public enum SlotStatus {
    OCCUPIED,
    AVAILABLE,
    NEEDS_REVIEW,
    UNCLASSIFIED;

    /**
     * Parses a case-insensitive string ("OCCUPIED", "AVAILABLE", "NEEDS_REVIEW", "REVIEW", "UNCLASSIFIED")
     * into a {@code SlotStatus}.
     *
     * @throws IllegalArgumentException if the string does not match any valid status
     */
    public static SlotStatus fromString(String value) {
        if (value == null) {
            throw new IllegalArgumentException("SlotStatus string must not be null.");
        }
        return switch (value.trim().toUpperCase()) {
            case "OCCUPIED"                  -> OCCUPIED;
            case "AVAILABLE"                 -> AVAILABLE;
            case "NEEDS_REVIEW", "REVIEW"    -> NEEDS_REVIEW;
            case "UNCLASSIFIED", "UNKNOWN"   -> UNCLASSIFIED;
            default -> throw new IllegalArgumentException(
                "Unknown SlotStatus: '" + value + "'. Expected OCCUPIED, AVAILABLE, NEEDS_REVIEW, or UNCLASSIFIED.");
        };
    }
}
