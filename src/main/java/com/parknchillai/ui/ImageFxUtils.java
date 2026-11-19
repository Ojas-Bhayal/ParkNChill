package com.parknchillai.ui;

import javafx.scene.image.Image;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;

import java.awt.image.BufferedImage;

/**
 * Utility for converting standard AWT/Java2D {@link BufferedImage} instances
 * into JavaFX {@link Image} representations with zero external dependencies.
 */
public final class ImageFxUtils {

    private ImageFxUtils() { /* utility class */ }

    /**
     * Converts a {@link BufferedImage} to a JavaFX {@link Image}.
     *
     * @param bimg input buffered image (any color type)
     * @return JavaFX image, or null if input is null
     */
    public static Image toFxImage(BufferedImage bimg) {
        if (bimg == null) return null;
        int width = bimg.getWidth();
        int height = bimg.getHeight();
        if (width <= 0 || height <= 0) return null;

        WritableImage wr = new WritableImage(width, height);
        PixelWriter pw = wr.getPixelWriter();

        int[] buffer = new int[width * height];
        bimg.getRGB(0, 0, width, height, buffer, 0, width);
        pw.setPixels(0, 0, width, height, PixelFormat.getIntArgbInstance(), buffer, 0, width);
        return wr;
    }
}
