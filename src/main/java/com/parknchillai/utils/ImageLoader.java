package com.parknchillai.utils;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.Locale;

/**
 * Loads, validates, inspects format/dimension metadata, and normalizes parking-area images from disk.
 */
public final class ImageLoader {

    public static final int MIN_WIDTH  = 10;
    public static final int MIN_HEIGHT = 10;

    private ImageLoader() { /* utility class */ }

    /**
     * Immutable metadata and normalized image container returned by {@link #loadWithMetadata}.
     */
    public record LoadedImageMetadata(
            String fileName,
            String filePath,
            String formatName,
            int originalWidth,
            int originalHeight,
            int processedWidth,
            int processedHeight,
            double scaleX,
            double scaleY,
            boolean wasResized,
            boolean perspectiveWarped,
            BufferedImage image
    ) {
        public LoadedImageMetadata withPerspectiveWarp(BufferedImage warpedImage) {
            if (warpedImage == null) throw new IllegalArgumentException("Warped image must not be null.");
            return new LoadedImageMetadata(
                fileName,
                filePath,
                formatName,
                originalWidth,
                originalHeight,
                warpedImage.getWidth(),
                warpedImage.getHeight(),
                (double) warpedImage.getWidth() / Math.max(1, originalWidth),
                (double) warpedImage.getHeight() / Math.max(1, originalHeight),
                wasResized,
                true,
                warpedImage
            );
        }

        public LoadedImageMetadata withoutImage() {
            return new LoadedImageMetadata(
                fileName,
                filePath,
                formatName,
                originalWidth,
                originalHeight,
                processedWidth,
                processedHeight,
                scaleX,
                scaleY,
                wasResized,
                perspectiveWarped,
                null
            );
        }

        public String describeSummary() {
            StringBuilder sb = new StringBuilder();
            sb.append(fileName).append(" [").append(formatName).append("] ");
            if (wasResized) {
                sb.append(String.format(Locale.ROOT,
                    "Raw: %d×%d px -> Normalized: %d×%d px (scale %.3f)",
                    originalWidth, originalHeight, processedWidth, processedHeight, scaleX));
            } else {
                sb.append(String.format(Locale.ROOT, "%d×%d px (native resolution)", processedWidth, processedHeight));
            }
            if (perspectiveWarped) {
                sb.append(" + 4-Corner Perspective Rectified");
            }
            return sb.toString();
        }
    }

    public static BufferedImage load(String path) throws IOException {
        return loadWithMetadata(path).image();
    }

    public static BufferedImage load(File file) throws IOException {
        return loadWithMetadata(file).image();
    }

    public static LoadedImageMetadata loadWithMetadata(String path) throws IOException {
        if (path == null || path.isBlank()) {
            throw new IllegalArgumentException("Image path must not be null or blank.");
        }
        return loadWithMetadata(new File(path));
    }

    public static LoadedImageMetadata loadWithMetadata(File file) throws IOException {
        if (file == null) {
            throw new IllegalArgumentException("File must not be null.");
        }
        if (!file.exists()) {
            throw new IOException("Image file not found: " + file.getAbsolutePath());
        }
        if (file.length() == 0) {
            throw new IOException("Image file is empty (0 bytes): " + file.getAbsolutePath());
        }

        String formatName = detectFormatName(file);
        BufferedImage rawImage = ImageIO.read(file);
        if (rawImage == null) {
            throw new IOException("Unsupported or corrupt image format: " + file.getAbsolutePath());
        }
        validate(rawImage, file.getAbsolutePath());

        int origW = rawImage.getWidth();
        int origH = rawImage.getHeight();
        BufferedImage normalized = ImageNormalizer.normalizeResolution(rawImage);
        int procW = normalized.getWidth();
        int procH = normalized.getHeight();
        boolean resized = (origW != procW) || (origH != procH);
        double scaleX = (double) procW / origW;
        double scaleY = (double) procH / origH;

        return new LoadedImageMetadata(
            file.getName(),
            file.getAbsolutePath(),
            formatName,
            origW,
            origH,
            procW,
            procH,
            scaleX,
            scaleY,
            resized,
            false,
            normalized
        );
    }

    private static String detectFormatName(File file) {
        try (ImageInputStream iis = ImageIO.createImageInputStream(file)) {
            if (iis != null) {
                Iterator<ImageReader> readers = ImageIO.getImageReaders(iis);
                if (readers.hasNext()) {
                    String fmt = readers.next().getFormatName();
                    if (fmt != null && !fmt.isBlank()) {
                        return fmt.toUpperCase(Locale.ROOT);
                    }
                }
            }
        } catch (IOException ignored) {
            // Fall back to extension inspection below
        }
        String name = file.getName().toLowerCase(Locale.ROOT);
        if (name.endsWith(".png")) return "PNG";
        if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "JPEG";
        if (name.endsWith(".bmp")) return "BMP";
        if (name.endsWith(".gif")) return "GIF";
        return "UNKNOWN";
    }

    private static void validate(BufferedImage image, String source) {
        if (image.getWidth() < MIN_WIDTH || image.getHeight() < MIN_HEIGHT) {
            throw new IllegalArgumentException(
                String.format("Image '%s' is too small (%dx%d); minimum is %dx%d.",
                    source, image.getWidth(), image.getHeight(), MIN_WIDTH, MIN_HEIGHT));
        }
    }
}
