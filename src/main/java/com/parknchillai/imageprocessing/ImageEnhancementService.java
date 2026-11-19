package com.parknchillai.imageprocessing;

import java.awt.image.BufferedImage;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Thread-safe enhancement service executing the versioned {@link ProcessingPolicy}.
 *
 * <p>Enforces the canonical enhancement policy across all callers (analysis, review,
 * dataset import, and preview). Rejects double-enhancement through type safety and
 * maintains a bounded LRU cache for deterministic performance.</p>
 */
public final class ImageEnhancementService {

    private static final int DEFAULT_CACHE_CAPACITY = 16;
    private static final ImageEnhancementService DEFAULT_INSTANCE = new ImageEnhancementService(DEFAULT_CACHE_CAPACITY);

    private final int cacheCapacity;
    private final Map<String, EnhancedImage> lruCache;

    public ImageEnhancementService() {
        this(DEFAULT_CACHE_CAPACITY);
    }

    public ImageEnhancementService(int cacheCapacity) {
        this.cacheCapacity = Math.max(1, cacheCapacity);
        this.lruCache = Collections.synchronizedMap(
                new LinkedHashMap<>(this.cacheCapacity, 0.75f, true) {
                    @Override
                    protected boolean removeEldestEntry(Map.Entry<String, EnhancedImage> eldest) {
                        return size() > ImageEnhancementService.this.cacheCapacity;
                    }
                }
        );
    }

    public static ImageEnhancementService getDefault() {
        return DEFAULT_INSTANCE;
    }

    /**
     * Enhances an original image using the canonical policy {@link ProcessingPolicy#CANONICAL}.
     */
    public EnhancedImage enhance(OriginalImage original) {
        return enhance(original, ProcessingPolicy.CANONICAL);
    }

    /**
     * Enhances an original image using a specific {@link ProcessingPolicy}.
     */
    public EnhancedImage enhance(OriginalImage original, ProcessingPolicy policy) {
        Objects.requireNonNull(original, "OriginalImage must not be null.");
        Objects.requireNonNull(policy, "ProcessingPolicy must not be null.");

        String cacheKey = original.getSha256() + ":" + policy.fingerprint();
        EnhancedImage cached = lruCache.get(cacheKey);
        if (cached != null) {
            return cached;
        }

        BufferedImage src = original.getInternalBuffer();
        BufferedImage enhanced = executeEnhancement(src, policy);
        EnhancedImage result = new EnhancedImage(enhanced, policy.policyId(), policy.fingerprint());
        lruCache.put(cacheKey, result);
        return result;
    }

    /**
     * Convenience method for existing raw BufferedImages, wrapping in OriginalImage first.
     */
    public EnhancedImage enhance(BufferedImage rawImage) {
        return enhance(new OriginalImage(rawImage), ProcessingPolicy.CANONICAL);
    }

    /**
     * Compile-time guard and runtime rejection against double enhancement.
     */
    public EnhancedImage enhance(EnhancedImage alreadyEnhanced) {
        throw new IllegalStateException(
                "Attempted to enhance an already-enhanced image (policy: "
                        + (alreadyEnhanced != null ? alreadyEnhanced.getPolicyId() : "null")
                        + "). Processing pipeline must enhance only the OriginalImage exactly once."
        );
    }

    /**
     * Executes deterministic contrast stretching and gamma correction according to the policy.
     */
    static BufferedImage executeEnhancement(BufferedImage source, ProcessingPolicy policy) {
        int width = source.getWidth();
        int height = source.getHeight();

        int[] luminanceHistogram = new int[256];
        long sumLuminance = 0L;
        int pixels = Math.multiplyExact(width, height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = source.getRGB(x, y);
                int r = (rgb >>> 16) & 0xff;
                int g = (rgb >>> 8) & 0xff;
                int b = rgb & 0xff;
                int lum = computeLuminance(r, g, b, policy.lumaR(), policy.lumaG(), policy.lumaB());
                luminanceHistogram[lum]++;
                sumLuminance += lum;
            }
        }

        int low = computePercentile(luminanceHistogram, pixels, policy.lowPercentile());
        int high = computePercentile(luminanceHistogram, pixels, policy.highPercentile());
        boolean stretch = (high - low) >= policy.minContrastRange();

        double mean = (double) sumLuminance / pixels;
        double gamma = mean < policy.darkMeanCutoff()
                ? policy.darkGamma()
                : (mean > policy.brightMeanCutoff() ? policy.brightGamma() : policy.neutralGamma());

        BufferedImage enhanced = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = source.getRGB(x, y);
                int r = (rgb >>> 16) & 0xff;
                int g = (rgb >>> 8) & 0xff;
                int b = rgb & 0xff;
                int oldLum = computeLuminance(r, g, b, policy.lumaR(), policy.lumaG(), policy.lumaB());

                double target = oldLum;
                if (stretch) {
                    target = (oldLum - low) * 255.0 / (high - low);
                }
                target = clamp(target, 0.0, 255.0);
                if (gamma != 1.0) {
                    target = 255.0 * Math.pow(target / 255.0, gamma);
                }
                target = clamp(target, 0.0, 255.0);

                int nr;
                int ng;
                int nb;
                if (oldLum <= 1) {
                    nr = ng = nb = (int) Math.round(target);
                } else {
                    double ratio = target / oldLum;
                    nr = clamp8(r * ratio);
                    ng = clamp8(g * ratio);
                    nb = clamp8(b * ratio);
                }
                enhanced.setRGB(x, y, (nr << 16) | (ng << 8) | nb);
            }
        }
        return enhanced;
    }

    private static int computePercentile(int[] histogram, int pixelCount, double percentile) {
        long target = Math.max(0L, (long) Math.floor((pixelCount - 1) * percentile));
        long cumulative = 0L;
        for (int i = 0; i < histogram.length; i++) {
            cumulative += histogram[i];
            if (cumulative > target) return i;
        }
        return 255;
    }

    private static int computeLuminance(int r, int g, int b, double lumaR, double lumaG, double lumaB) {
        return clamp8(lumaR * r + lumaG * g + lumaB * b);
    }

    private static int clamp8(double value) {
        return (int) Math.round(clamp(value, 0.0, 255.0));
    }

    private static double clamp(double value, double low, double high) {
        return Math.max(low, Math.min(high, value));
    }
}
