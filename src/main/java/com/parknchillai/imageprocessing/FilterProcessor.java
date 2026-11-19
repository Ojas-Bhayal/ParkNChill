package com.parknchillai.imageprocessing;

import java.awt.image.BufferedImage;

/**
 * Applies spatial smoothing filters to a grayscale image.
 *
 * <p>Two filters are provided:
 * <ul>
 *   <li><strong>Box (mean) filter</strong> — simple average over a square
 *       neighbourhood; fast and easy to understand.</li>
 *   <li><strong>Gaussian filter</strong> — weighted average that gives more
 *       weight to the central pixel; reduces noise while preserving edges
 *       better than the box filter.</li>
 * </ul>
 *
 * <p>Both filters use reflected (mirror) boundary padding so edge pixels
 * are not lost, safely handling narrow ROIs down to 1×1 pixels.
 */
public final class FilterProcessor {

    private FilterProcessor() { /* utility class */ }

    /**
     * Applies a box (mean) filter with the given kernel size.
     *
     * @param input      grayscale source image
     * @param kernelSize side length of the square kernel (must be odd, >= 3)
     * @return new filtered grayscale image
     */
    public static BufferedImage applyBoxFilter(BufferedImage input, int kernelSize) {
        validateInputAndKernel(input, kernelSize);
        BufferedImage grayInput = GrayscaleConverter.ensureGrayscale(input);
        int r = kernelSize / 2;
        double weight = 1.0 / (kernelSize * kernelSize);
        double[] kernel = new double[kernelSize * kernelSize];
        java.util.Arrays.fill(kernel, weight);
        return convolve(grayInput, kernel, kernelSize, r);
    }

    /**
     * Applies a Gaussian smoothing filter with the given kernel size.
     *
     * <p>The Gaussian kernel is generated using the standard formula
     * {@code G(x,y) = exp(-(x²+y²)/(2σ²))} where σ is derived from the
     * kernel size as {@code σ = kernelSize / 6.0} so that 3σ fits within
     * the half-window.
     *
     * @param input      grayscale source image
     * @param kernelSize side length of the square kernel (must be odd, >= 3)
     * @return new filtered grayscale image
     */
    public static BufferedImage applyGaussianFilter(BufferedImage input, int kernelSize) {
        validateInputAndKernel(input, kernelSize);
        BufferedImage grayInput = GrayscaleConverter.ensureGrayscale(input);
        int r = kernelSize / 2;
        double sigma = kernelSize / 6.0;
        double[] kernel = buildGaussianKernel(kernelSize, sigma);
        return convolve(grayInput, kernel, kernelSize, r);
    }

    /**
     * Applies a Gaussian smoothing filter to a color (RGB) image across each channel.
     *
     * @param input      color RGB or ARGB source image
     * @param kernelSize side length of the square kernel (must be odd, >= 3)
     * @return new filtered RGB image
     */
    public static BufferedImage applyGaussianFilterRgb(BufferedImage input, int kernelSize) {
        validateInputAndKernel(input, kernelSize);
        int width = input.getWidth();
        int height = input.getHeight();
        int r = kernelSize / 2;
        double sigma = kernelSize / 6.0;
        double[] kernel = buildGaussianKernel(kernelSize, sigma);

        BufferedImage output = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double accR = 0.0, accG = 0.0, accB = 0.0;
                for (int ky = -r; ky <= r; ky++) {
                    for (int kx = -r; kx <= r; kx++) {
                        int nx = reflect(x + kx, width);
                        int ny = reflect(y + ky, height);
                        int rgb = input.getRGB(nx, ny);
                        double w = kernel[(ky + r) * kernelSize + (kx + r)];
                        accR += ((rgb >>> 16) & 0xff) * w;
                        accG += ((rgb >>> 8) & 0xff) * w;
                        accB += (rgb & 0xff) * w;
                    }
                }
                int cr = clamp((int) Math.round(accR));
                int cg = clamp((int) Math.round(accG));
                int cb = clamp((int) Math.round(accB));
                output.setRGB(x, y, (cr << 16) | (cg << 8) | cb);
            }
        }
        return output;
    }

    /** Builds and normalises a 2-D Gaussian kernel stored as a flat array. */
    private static double[] buildGaussianKernel(int size, double sigma) {
        double[] kernel = new double[size * size];
        int r = size / 2;
        double sum = 0.0;
        for (int ky = -r; ky <= r; ky++) {
            for (int kx = -r; kx <= r; kx++) {
                double val = Math.exp(-(kx * kx + ky * ky) / (2 * sigma * sigma));
                kernel[(ky + r) * size + (kx + r)] = val;
                sum += val;
            }
        }
        for (int i = 0; i < kernel.length; i++) kernel[i] /= sum;
        return kernel;
    }

    /** Applies a 2-D spatial convolution with the given normalized flat kernel. */
    private static BufferedImage convolve(BufferedImage input, double[] kernel,
                                          int kernelSize, int r) {
        int width  = input.getWidth();
        int height = input.getHeight();
        BufferedImage output = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double acc = 0.0;
                for (int ky = -r; ky <= r; ky++) {
                    for (int kx = -r; kx <= r; kx++) {
                        int nx = reflect(x + kx, width);
                        int ny = reflect(y + ky, height);
                        int gray = input.getRaster().getSample(nx, ny, 0);
                        acc += gray * kernel[(ky + r) * kernelSize + (kx + r)];
                    }
                }
                int clamped = clamp((int) Math.round(acc));
                output.getRaster().setSample(x, y, 0, clamped);
            }
        }
        return output;
    }

    /**
     * Reflects a coordinate at the image boundary (mirror padding).
     * Safely wraps coordinates even when {@code size < kernelRadius} or {@code size == 1}.
     */
    static int reflect(int coord, int size) {
        if (size <= 1) return 0;
        while (coord < 0 || coord >= size) {
            if (coord < 0) {
                coord = -coord;
            }
            if (coord >= size) {
                coord = 2 * size - coord - 2;
            }
        }
        return coord;
    }

    private static int clamp(int v) { return Math.max(0, Math.min(255, v)); }

    private static void validateInputAndKernel(BufferedImage input, int kernelSize) {
        if (input == null) throw new IllegalArgumentException("Input image must not be null.");
        if (kernelSize < 3 || kernelSize % 2 == 0) {
            throw new IllegalArgumentException(
                "Kernel size must be odd and >= 3; got " + kernelSize);
        }
    }
}
