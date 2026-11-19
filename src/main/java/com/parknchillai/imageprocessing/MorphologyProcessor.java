package com.parknchillai.imageprocessing;

import java.awt.image.BufferedImage;

/**
 * Applies morphological binary operations (erosion, dilation, opening, closing)
 * to a binary image.
 *
 * <p>A flat square structuring element of the given side length is used for
 * all operations. Any non-zero foreground pixel (>= 128) is treated as
 * foreground (255).
 */
public final class MorphologyProcessor {

    private MorphologyProcessor() { /* utility class */ }

    public static BufferedImage erode(BufferedImage input, int kernelSize) {
        validateInput(input, kernelSize);
        return morphOp(GrayscaleConverter.ensureGrayscale(input), kernelSize, true);
    }

    public static BufferedImage dilate(BufferedImage input, int kernelSize) {
        validateInput(input, kernelSize);
        return morphOp(GrayscaleConverter.ensureGrayscale(input), kernelSize, false);
    }

    public static BufferedImage open(BufferedImage input, int kernelSize) {
        return dilate(erode(input, kernelSize), kernelSize);
    }

    public static BufferedImage close(BufferedImage input, int kernelSize) {
        return erode(dilate(input, kernelSize), kernelSize);
    }

    private static BufferedImage morphOp(BufferedImage input, int kernelSize, boolean isErosion) {
        int width  = input.getWidth();
        int height = input.getHeight();
        int r = kernelSize / 2;
        BufferedImage output = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                boolean result = isErosion;
                outer:
                for (int ky = -r; ky <= r; ky++) {
                    for (int kx = -r; kx <= r; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        boolean inBounds = nx >= 0 && nx < width && ny >= 0 && ny < height;
                        boolean isWhite = inBounds && input.getRaster().getSample(nx, ny, 0) >= 128;
                        if (isErosion) {
                            if (!isWhite) { result = false; break outer; }
                        } else {
                            if (isWhite) { result = true; break outer; }
                        }
                    }
                }
                int val = result ? 255 : 0;
                output.getRaster().setSample(x, y, 0, val);
            }
        }
        return output;
    }

    private static void validateInput(BufferedImage input, int kernelSize) {
        if (input == null) throw new IllegalArgumentException("Input image must not be null.");
        if (kernelSize < 3 || kernelSize % 2 == 0) {
            throw new IllegalArgumentException(
                "Kernel size must be odd and >= 3; got " + kernelSize);
        }
    }
}
