package com.parknchillai.imageprocessing;

import java.awt.image.BufferedImage;

/**
 * Applies binary thresholding to a grayscale image.
 *
 * <p>Three modes are provided:
 * <ul>
 *   <li><strong>Fixed threshold</strong> — pixels above the threshold become
 *       white (255); pixels at or below become black (0).</li>
 *   <li><strong>Otsu's method</strong> — automatically selects the global threshold
 *       that maximises between-class variance across the image histogram.
 *       If the image is completely uniform or exhibits only unimodal low-contrast
 *       background variation ({@code betweenClassSigma < 8.5}), {@link #applyOtsuThreshold}
 *       binarises the region to {@code 0} (background) rather than splitting an
 *       empty asphalt gradient in half into {@code ~50%} false foreground.</li>
 *   <li><strong>Local Adaptive Thresholding (Integral Image)</strong> — computes
 *       the local neighbourhood mean {@code M(x,y)} in {@code O(1)} per pixel using
 *       a 2D integral image (summed-area table) and applies two-sided local contrast
 *       {@code |I(x,y) - M(x,y)| > max(8.0, M(x,y) * k)} so smooth illumination
 *       gradients and outdoor shadows on empty pavement are suppressed while both
 *       bright and dark achromatic vehicle structures are segmented.</li>
 * </ul>
 */
public final class ThresholdProcessor {

    /** Default local window size (odd integer) for integral-image adaptive thresholding. */
    public static final int DEFAULT_ADAPTIVE_WINDOW = 15;

    /** Default fractional sensitivity offset for adaptive thresholding. */
    public static final double DEFAULT_ADAPTIVE_K = 0.12;

    /**
     * Minimum between-class standard deviation {@code sigma_B = sqrt(w0 * w1) * |mu1 - mu0|}
     * required for Otsu's method to segment a genuine bimodal foreground/background object
     * rather than splitting unimodal empty pavement noise/gradients in half.
     */
    public static final double MIN_OTSU_BIMODAL_SIGMA = 8.5;

    private ThresholdProcessor() { /* utility class */ }

    /**
     * Applies a fixed threshold.
     *
     * @param input     grayscale source image
     * @param threshold value in [0, 255]; pixels > threshold → 255, else → 0
     * @return binary output image
     */
    public static BufferedImage applyFixedThreshold(BufferedImage input, int threshold) {
        if (input == null) throw new IllegalArgumentException("Input image must not be null.");
        if (threshold < 0 || threshold > 255) {
            throw new IllegalArgumentException("Threshold must be in [0, 255]; got " + threshold);
        }
        BufferedImage grayInput = GrayscaleConverter.ensureGrayscale(input);
        return binarise(grayInput, threshold);
    }

    /**
     * Applies Otsu's global thresholding method with a unimodal low-contrast background guard.
     * When the maximum between-class standard deviation {@code sigma_B < MIN_OTSU_BIMODAL_SIGMA},
     * the input contains only unimodal background variation (such as empty asphalt with a
     * gentle shadow or lighting gradient) and binarises to all {@code 0}s.
     *
     * @param input grayscale source image
     * @return binary output image
     */
    public static BufferedImage applyOtsuThreshold(BufferedImage input) {
        if (input == null) throw new IllegalArgumentException("Input image must not be null.");
        BufferedImage grayInput = GrayscaleConverter.ensureGrayscale(input);
        double sigmaB = computeBetweenClassSigma(grayInput);
        if (sigmaB < MIN_OTSU_BIMODAL_SIGMA) {
            return new BufferedImage(grayInput.getWidth(), grayInput.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        }
        int t = computeOtsuThreshold(grayInput);
        return binarise(grayInput, t);
    }

    /**
     * Applies local adaptive thresholding using default parameters
     * ({@code windowSize = 15, k = 0.12}).
     */
    public static BufferedImage applyAdaptiveThreshold(BufferedImage input) {
        return applyAdaptiveThreshold(input, DEFAULT_ADAPTIVE_WINDOW, DEFAULT_ADAPTIVE_K);
    }

    /**
     * Applies two-sided local adaptive thresholding using a 2D integral image
     * (summed-area table) in {@code O(W * H)} time.
     *
     * <p>A pixel {@code I(x, y)} is marked as foreground ({@code 255}) if its absolute
     * deviation from the local {@code windowSize x windowSize} neighbourhood mean
     * {@code M(x, y)} exceeds {@code max(8.0, M(x, y) * k)}, suppressing smooth shadows
     * and lighting gradients across empty asphalt while preserving both bright and dark
     * achromatic vehicle structures.
     *
     * @param input      source image (converted to grayscale if needed)
     * @param windowSize odd neighbourhood size {@code >= 3} (e.g. 15)
     * @param k          relative threshold sensitivity in {@code (0, 1)} (e.g. 0.12)
     * @return binary {@link BufferedImage#TYPE_BYTE_GRAY} image with values {@code 0} or {@code 255}
     */
    public static BufferedImage applyAdaptiveThreshold(BufferedImage input, int windowSize, double k) {
        if (input == null) throw new IllegalArgumentException("Input image must not be null.");
        if (windowSize < 3 || windowSize % 2 == 0) {
            throw new IllegalArgumentException("windowSize must be an odd integer >= 3; got " + windowSize);
        }
        if (!Double.isFinite(k) || k <= 0.0 || k >= 1.0) {
            throw new IllegalArgumentException("Sensitivity k must be in (0, 1); got " + k);
        }

        BufferedImage gray = GrayscaleConverter.ensureGrayscale(input);
        int w = gray.getWidth();
        int h = gray.getHeight();

        // Build 1-indexed 2D integral image: integral[y + 1][x + 1] = sum_{j<=y, i<=x} I(i, j)
        long[][] integral = new long[h + 1][w + 1];
        for (int y = 0; y < h; y++) {
            long rowSum = 0;
            for (int x = 0; x < w; x++) {
                rowSum += gray.getRaster().getSample(x, y, 0);
                integral[y + 1][x + 1] = integral[y][x + 1] + rowSum;
            }
        }

        int r = windowSize / 2;
        BufferedImage output = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < h; y++) {
            int y0 = Math.max(0, y - r);
            int y1 = Math.min(h - 1, y + r);
            for (int x = 0; x < w; x++) {
                int x0 = Math.max(0, x - r);
                int x1 = Math.min(w - 1, x + r);

                int count = (x1 - x0 + 1) * (y1 - y0 + 1);
                long sum = integral[y1 + 1][x1 + 1]
                         - integral[y0][x1 + 1]
                         - integral[y1 + 1][x0]
                         + integral[y0][x0];
                double localMean = (double) sum / count;
                int pixel = gray.getRaster().getSample(x, y, 0);

                double delta = Math.max(8.0, localMean * k);
                int val = (Math.abs(pixel - localMean) > delta) ? 255 : 0;
                output.getRaster().setSample(x, y, 0, val);
            }
        }
        return output;
    }

    /**
     * Computes the maximum between-class standard deviation {@code sigma_B = sqrt(w0 * w1) * |mu1 - mu0|}
     * across all 256 Otsu threshold candidates.
     */
    public static double computeBetweenClassSigma(BufferedImage input) {
        if (input == null) throw new IllegalArgumentException("Input image must not be null.");
        BufferedImage grayInput = GrayscaleConverter.ensureGrayscale(input);
        int[] histogram = buildHistogram(grayInput);
        int total = grayInput.getWidth() * grayInput.getHeight();
        if (total <= 0) return 0.0;

        double sumAll = 0.0;
        for (int i = 0; i < 256; i++) {
            sumAll += (double) i * histogram[i];
        }

        double sumBackground = 0.0;
        int weightBackground = 0;
        double maxVariance = 0.0;

        for (int t = 0; t < 256; t++) {
            weightBackground += histogram[t];
            if (weightBackground == 0) continue;
            int weightForeground = total - weightBackground;
            if (weightForeground == 0) break;

            sumBackground += (double) t * histogram[t];
            double meanBackground = sumBackground / weightBackground;
            double meanForeground = (sumAll - sumBackground) / weightForeground;
            double variance = (double) weightBackground * weightForeground
                * (meanBackground - meanForeground)
                * (meanBackground - meanForeground);
            if (variance > maxVariance) {
                maxVariance = variance;
            }
        }
        return Math.sqrt(maxVariance / ((double) total * total));
    }

    /**
     * Calculates the Otsu threshold value without applying it.
     * Exposed for testing and for documenting the selected value.
     */
    public static int computeOtsuThreshold(BufferedImage input) {
        if (input == null) throw new IllegalArgumentException("Input image must not be null.");
        BufferedImage grayInput = GrayscaleConverter.ensureGrayscale(input);

        int[] histogram = buildHistogram(grayInput);
        int total = grayInput.getWidth() * grayInput.getHeight();

        double sumAll = 0.0;
        int minNonZeroBin = 0;
        boolean foundNonZero = false;
        for (int i = 0; i < 256; i++) {
            sumAll += (double) i * histogram[i];
            if (!foundNonZero && histogram[i] > 0) {
                minNonZeroBin = i;
                foundNonZero = true;
            }
        }

        double sumBackground = 0.0;
        int weightBackground = 0;
        double maxVariance = 0.0;
        int firstT = minNonZeroBin;
        int lastT = minNonZeroBin;

        for (int t = 0; t < 256; t++) {
            weightBackground += histogram[t];
            if (weightBackground == 0) continue;

            int weightForeground = total - weightBackground;
            if (weightForeground == 0) break;

            sumBackground += (double) t * histogram[t];
            double meanBackground = sumBackground / weightBackground;
            double meanForeground = (sumAll - sumBackground) / weightForeground;

            double variance = (double) weightBackground * weightForeground
                * (meanBackground - meanForeground)
                * (meanBackground - meanForeground);

            if (variance > maxVariance + 1e-6) {
                maxVariance = variance;
                firstT = t;
                lastT = t;
            } else if (Math.abs(variance - maxVariance) <= 1e-6 && maxVariance > 0) {
                lastT = t;
            }
        }
        return (firstT + lastT) / 2;
    }

    private static BufferedImage binarise(BufferedImage input, int threshold) {
        int width  = input.getWidth();
        int height = input.getHeight();
        BufferedImage output = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = input.getRaster().getSample(x, y, 0);
                int val  = gray > threshold ? 255 : 0;
                output.getRaster().setSample(x, y, 0, val);
            }
        }
        return output;
    }

    /** Builds a 256-bin intensity histogram from a grayscale image. */
    static int[] buildHistogram(BufferedImage input) {
        int[] hist = new int[256];
        int width  = input.getWidth();
        int height = input.getHeight();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int sample = Math.max(0, Math.min(255, input.getRaster().getSample(x, y, 0)));
                hist[sample]++;
            }
        }
        return hist;
    }
}
