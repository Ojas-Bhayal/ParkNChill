package com.parknchillai.detection;

import java.util.Locale;
import java.util.Optional;

/**
 * Supported vehicle categories for raw-image vehicle detection.
 * Maps standard COCO vehicle class IDs ({@code 2=car, 3=motorcycle, 5=bus, 7=truck})
 * as well as compact 4-class vehicle detector heads ({@code 0=car, 1=motorcycle, 2=bus, 3=truck}).
 */
public enum VehicleCategory {
    CAR(2, "Car"),
    MOTORCYCLE(3, "Motorcycle"),
    BUS(5, "Bus"),
    TRUCK(7, "Truck");

    private final int cocoClassId;
    private final String displayName;

    VehicleCategory(int cocoClassId, String displayName) {
        this.cocoClassId = cocoClassId;
        this.displayName = displayName;
    }

    public int getCocoClassId()    { return cocoClassId; }
    public String getDisplayName() { return displayName; }

    /**
     * Resolves a COCO 80-class index ({@code 2, 3, 5, 7}) to a supported {@link VehicleCategory},
     * returning {@link Optional#empty()} for non-vehicle classes (e.g. person, traffic light).
     */
    public static Optional<VehicleCategory> fromCocoClassId(int classId) {
        return switch (classId) {
            case 2 -> Optional.of(CAR);
            case 3 -> Optional.of(MOTORCYCLE);
            case 5 -> Optional.of(BUS);
            case 7 -> Optional.of(TRUCK);
            default -> Optional.empty();
        };
    }

    /**
     * Resolves a class index given the total number of classes in the detector head.
     * Supports 1-class ({@code 0=CAR}), 4-class ({@code 0..3}), and 80-class COCO heads.
     */
    public static Optional<VehicleCategory> fromModelClassIndex(int classIndex, int numClasses) {
        if (numClasses == 1 && classIndex == 0) {
            return Optional.of(CAR);
        }
        if (numClasses == 4) {
            return switch (classIndex) {
                case 0 -> Optional.of(CAR);
                case 1 -> Optional.of(MOTORCYCLE);
                case 2 -> Optional.of(BUS);
                case 3 -> Optional.of(TRUCK);
                default -> Optional.empty();
            };
        }
        return fromCocoClassId(classIndex);
    }

    public static VehicleCategory fromString(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("VehicleCategory string must not be null or blank.");
        }
        String norm = value.trim().toUpperCase(Locale.ROOT);
        for (VehicleCategory cat : values()) {
            if (cat.name().equals(norm) || cat.displayName.toUpperCase(Locale.ROOT).equals(norm)) {
                return cat;
            }
        }
        throw new IllegalArgumentException("Unsupported VehicleCategory: '" + value + "'");
    }
}
