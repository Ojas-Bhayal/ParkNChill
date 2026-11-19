package com.parknchillai.imageprocessing;

import java.awt.image.BufferedImage;

public final class ColorSpaceConverter {

    private ColorSpaceConverter() { /* utility class */ }

    public static BufferedImage toSaturationMap(BufferedImage input) {
        if (input == null) {
            throw new IllegalArgumentException("Input image must not be null.");
        }
        int width  = input.getWidth();
        int height = input.getHeight();
        BufferedImage satImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        if (input.getType() == BufferedImage.TYPE_BYTE_GRAY) {
            return satImage;
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int argb = input.getRGB(x, y);
                int r = (argb >> 16) & 0xFF;
                int g = (argb >> 8)  & 0xFF;
                int b = argb         & 0xFF;

                int cMax = Math.max(r, Math.max(g, b));
                int cMin = Math.min(r, Math.min(g, b));

                int sat = (cMax == 0) ? 0 : (int) Math.round(((cMax - cMin) * 255.0) / cMax);
                satImage.getRaster().setSample(x, y, 0, Math.max(0, Math.min(255, sat)));
            }
        }
        return satImage;
    }

    public static BufferedImage toValueMap(BufferedImage input) {
        if (input == null) {
            throw new IllegalArgumentException("Input image must not be null.");
        }
        int width  = input.getWidth();
        int height = input.getHeight();
        BufferedImage valImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        if (input.getType() == BufferedImage.TYPE_BYTE_GRAY) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    valImage.getRaster().setSample(x, y, 0, input.getRaster().getSample(x, y, 0));
                }
            }
            return valImage;
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int argb = input.getRGB(x, y);
                int r = (argb >> 16) & 0xFF;
                int g = (argb >> 8)  & 0xFF;
                int b = argb         & 0xFF;
                int cMax = Math.max(r, Math.max(g, b));
                valImage.getRaster().setSample(x, y, 0, cMax);
            }
        }
        return valImage;
    }
}
