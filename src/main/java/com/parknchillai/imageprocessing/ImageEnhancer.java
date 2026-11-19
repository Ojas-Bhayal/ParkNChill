package com.parknchillai.imageprocessing;

import java.awt.image.BufferedImage;

/**
 * Convenience entry point and interactive preview adapter for image enhancement.
 *
 * <p>Production enhancement delegates directly to {@link ImageEnhancementService} using
 * {@link ProcessingPolicy#CANONICAL}. Interactive slider adjustments remain strictly
 * separated in {@link #adjustForPreview(BufferedImage, double, double, double)} for
 * UI experimentation only.</p>
 */
public final class ImageEnhancer {
    /** Increment whenever the pixel enhancement policy changes. Included in the 10-feature schema version. */
    public static final String PIPELINE_VERSION = ProcessingPolicy.CANONICAL_POLICY_ID;

    private ImageEnhancer() { }

    /**
     * Returns an enhanced RGB copy using the canonical processing policy {@link ProcessingPolicy#CANONICAL}.
     * The supplied source remains unchanged.
     */
    public static BufferedImage autoEnhance(BufferedImage source) {
        if (source == null) {
            throw new IllegalArgumentException("Source image must not be null.");
        }
        return ImageEnhancementService.getDefault().enhance(new OriginalImage(source)).getCopy();
    }

    /**
     * Interactive preview filter: contrast multiplier, additive brightness, and gamma.
     * Delegates to {@link #adjust(BufferedImage, double, double, double, int)} with 0 noise reduction.
     */
    public static BufferedImage adjustForPreview(BufferedImage source, double contrast, double brightness, double gamma) {
        return adjust(source, contrast, brightness, gamma, 0);
    }

    /**
     * Applies contrast, brightness, gamma adjustment, and optional Gaussian spatial smoothing
     * noise reduction to produce an enhanced working image.
     *
     * @param source               source photograph
     * @param contrast             contrast multiplier (&gt; 0)
     * @param brightness           additive brightness offset
     * @param gamma                gamma tone correction (&gt; 0)
     * @param noiseReductionKernel odd kernel size (&gt;= 3) for Gaussian smoothing, or &lt; 3 for none
     * @return new adjusted image without modifying source
     */
    public static BufferedImage adjust(BufferedImage source, double contrast, double brightness, double gamma, int noiseReductionKernel) {
        if (source == null) throw new IllegalArgumentException("Source image must not be null.");
        if (!Double.isFinite(contrast) || !Double.isFinite(brightness) || !Double.isFinite(gamma) || gamma <= 0.0) {
            throw new IllegalArgumentException("Contrast, brightness, and gamma must be finite; gamma must be positive.");
        }
        int w = source.getWidth(), h = source.getHeight();
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = source.getRGB(x, y);
                int r = adjustChannel((rgb >>> 16) & 255, contrast, brightness, gamma);
                int g = adjustChannel((rgb >>> 8) & 255, contrast, brightness, gamma);
                int b = adjustChannel(rgb & 255, contrast, brightness, gamma);
                out.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        if (noiseReductionKernel >= 3) {
            int k = (noiseReductionKernel % 2 == 0) ? noiseReductionKernel + 1 : noiseReductionKernel;
            return FilterProcessor.applyGaussianFilterRgb(out, k);
        }
        return out;
    }

    private static int adjustChannel(int value, double contrast, double brightness, double gamma) {
        double centered = (value - 127.5) * contrast + 127.5 + brightness;
        centered = clamp(centered, 0.0, 255.0);
        return clamp8(255.0 * Math.pow(centered / 255.0, 1.0 / gamma));
    }

    private static int clamp8(double value) {
        return (int) Math.round(clamp(value, 0.0, 255.0));
    }

    private static double clamp(double value, double low, double high) {
        return Math.max(low, Math.min(high, value));
    }
}
