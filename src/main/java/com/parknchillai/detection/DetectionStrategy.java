package com.parknchillai.detection;

/**
 * Strategy controlling vehicle detection computational cost vs. spatial accuracy (WS-2).
 *
 * <ul>
 *   <li>{@link #FAST}: Single-scale full-image ONNX pass (640x640 letterbox).
 *       Fastest execution (1 ONNX pass), ideal for live interactive analysis and real-time responsiveness.</li>
 *   <li>{@link #ACCURATE}: Multi-scale tiled inference (1 full pass + 4 overhead tiles = 5 ONNX passes)
 *       with cross-tile NMS reconciliation. Maximum vehicle recall for small or distant vehicles in large parking lots.</li>
 * </ul>
 */
public enum DetectionStrategy {
    FAST("Fast (Single Pass 640x640)", 1),
    ACCURATE("Accurate (Multi-Scale Tiled 1+4 Passes)", 5);

    private final String displayName;
    private final int expectedPasses;

    DetectionStrategy(String displayName, int expectedPasses) {
        this.displayName = displayName;
        this.expectedPasses = expectedPasses;
    }

    public String getDisplayName() {
        return displayName;
    }

    public int getExpectedPasses() {
        return expectedPasses;
    }
}
