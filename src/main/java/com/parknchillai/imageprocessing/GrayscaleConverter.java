package com.parknchillai.imageprocessing;

import java.awt.image.BufferedImage;

/**
 * Converts a colour {@link BufferedImage} to an 8-bit grayscale image.
 *
 * <p>The luminance formula used is the ITU-R BT.601 recommendation:
 * {@code Y = 0.299·R + 0.587·G + 0.114·B}
 *
 * <p>If the source image has an alpha channel, transparent pixels are composited
 * over a neutral dark asphalt background (intensity 0) proportional to alpha.
 *
 * <p>The output image uses {@link BufferedImage#TYPE_BYTE_GRAY} so that
 * every pixel stores a single 8-bit intensity value, which is the expected
 * input representation for the downstream filtering and thresholding steps.
 */
public final class GrayscaleConverter {

    private GrayscaleConverter() { /* utility class */ }

    /**
     * Converts the given image to grayscale, or returns a copy in {@code TYPE_BYTE_GRAY}.
     *
     * @param input source image (any standard BufferedImage type)
     * @return a new {@code TYPE_BYTE_GRAY} image with the same dimensions
     * @throws IllegalArgumentException if {@code input} is null
     */
    public static BufferedImage toGrayscale(BufferedImage input) {
        if (input == null) throw new IllegalArgumentException("Input image must not be null.");

        int width  = input.getWidth();
        int height = input.getHeight();

        BufferedImage gray = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        if (input.getType() == BufferedImage.TYPE_BYTE_GRAY) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    gray.getRaster().setSample(x, y, 0, input.getRaster().getSample(x, y, 0));
                }
            }
            return gray;
        }

        boolean hasAlpha = input.getColorModel().hasAlpha();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int argb = input.getRGB(x, y);
                int a = hasAlpha ? ((argb >>> 24) & 0xFF) : 255;
                int r = (argb >> 16) & 0xFF;
                int g = (argb >>  8) & 0xFF;
                int b =  argb        & 0xFF;

                // BT.601 luminance scaled by alpha opacity
                double lum = 0.299 * r + 0.587 * g + 0.114 * b;
                if (a < 255) {
                    lum = lum * (a / 255.0);
                }
                int luminance = Math.max(0, Math.min(255, (int) Math.round(lum)));
                gray.getRaster().setSample(x, y, 0, luminance);
            }
        }
        return gray;
    }

    /**
     * Ensures that an input image is in {@code TYPE_BYTE_GRAY} format before
     * single-band raster operations.
     */
    public static BufferedImage ensureGrayscale(BufferedImage input) {
        if (input == null) throw new IllegalArgumentException("Input image must not be null.");
        if (input.getType() == BufferedImage.TYPE_BYTE_GRAY) {
            return input;
        }
        return toGrayscale(input);
    }

    /**
     * Reads the gray intensity of a single pixel from a TYPE_BYTE_GRAY image.
     * Convenience method for feature extraction and testing.
     */
    public static int getGrayValue(BufferedImage grayImage, int x, int y) {
        if (grayImage == null) throw new IllegalArgumentException("grayImage must not be null.");
        return grayImage.getRaster().getSample(x, y, 0);
    }
}
