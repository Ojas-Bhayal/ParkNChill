package com.parknchillai.utils;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

/**
 * Normalizes high-resolution parking lot images to a consistent maximum bounding dimension
 * while strictly preserving aspect ratio.
 */
public final class ImageNormalizer {

    public static final int DEFAULT_MAX_DIMENSION = 1280;

    private ImageNormalizer() { /* utility class */ }

    public static BufferedImage normalizeResolution(BufferedImage input) {
        return normalizeResolution(input, DEFAULT_MAX_DIMENSION);
    }

    public static BufferedImage normalizeResolution(BufferedImage input, int maxDimension) {
        if (input == null) {
            throw new IllegalArgumentException("Input image must not be null.");
        }
        if (maxDimension < 10) {
            throw new IllegalArgumentException("maxDimension must be >= 10; got " + maxDimension);
        }

        int width = input.getWidth();
        int height = input.getHeight();
        if (width <= maxDimension && height <= maxDimension) {
            return input;
        }

        double scale = Math.min((double) maxDimension / width, (double) maxDimension / height);
        int targetWidth = Math.max(10, (int) Math.round(width * scale));
        int targetHeight = Math.max(10, (int) Math.round(height * scale));

        int targetType = (input.getType() == BufferedImage.TYPE_BYTE_GRAY)
            ? BufferedImage.TYPE_BYTE_GRAY
            : BufferedImage.TYPE_INT_RGB;

        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, targetType);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g.drawImage(input, 0, 0, targetWidth, targetHeight, null);
        g.dispose();

        return resized;
    }
}
