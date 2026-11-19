package com.parknchillai.ai;

import com.parknchillai.parking.SlotStatus;

import java.util.Locale;

/**
 * Holds the outcome of a single parking-space classification, including
 * the predicted {@link SlotStatus} and the CART leaf purity confidence in {@code [0.0, 1.0]}.
 */
public final class PredictionResult {

    private final int        slotId;
    private final SlotStatus predictedStatus;
    private final double     confidence;

    public PredictionResult(int slotId, SlotStatus predictedStatus) {
        this(slotId, predictedStatus, 1.0);
    }

    public PredictionResult(int slotId, SlotStatus predictedStatus, double confidence) {
        if (predictedStatus == null) {
            throw new IllegalArgumentException("Predicted status must not be null.");
        }
        if (!Double.isFinite(confidence) || confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("Confidence must be in [0.0, 1.0]; got " + confidence);
        }
        this.slotId          = slotId;
        this.predictedStatus = predictedStatus;
        this.confidence      = confidence;
    }

    public int        getSlotId()          { return slotId; }
    public SlotStatus getPredictedStatus() { return predictedStatus; }
    public double     getConfidence()      { return confidence; }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "PredictionResult[slot=%d, status=%s, conf=%.1f%%]",
            slotId, predictedStatus, confidence * 100.0);
    }
}
