package com.parknchillai.imageprocessing;

import java.awt.image.BufferedImage;

/**
 * Applies the Sobel edge-detection operator to a grayscale image.
 */
public final class EdgeDetector {

    private static final int[][] GX = {
        {-1,  0,  1},
        {-2,  0,  2},
        {-1,  0,  1}
    };

    private static final int[][] GY = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    private EdgeDetector() { /* utility class */ }

    public static BufferedImage applySobel(BufferedImage input) {
        if (input == null) throw new IllegalArgumentException("Input image must not be null.");
        BufferedImage grayInput = GrayscaleConverter.ensureGrayscale(input);

        int width  = grayInput.getWidth();
        int height = grayInput.getHeight();
        BufferedImage output = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double gx = 0.0;
                double gy = 0.0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int nx = clampCoord(x + kx, width);
                        int ny = clampCoord(y + ky, height);
                        int gray = grayInput.getRaster().getSample(nx, ny, 0);
                        gx += gray * GX[ky + 1][kx + 1];
                        gy += gray * GY[ky + 1][kx + 1];
                    }
                }
                int magnitude = clamp((int) Math.round(Math.sqrt(gx * gx + gy * gy)));
                output.getRaster().setSample(x, y, 0, magnitude);
            }
        }
        return output;
    }

    private static int clampCoord(int coord, int size) {
        return Math.max(0, Math.min(size - 1, coord));
    }

    private static int clamp(int v) { return Math.max(0, Math.min(255, v)); }
}
