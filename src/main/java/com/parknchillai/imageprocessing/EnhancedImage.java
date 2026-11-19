package com.parknchillai.imageprocessing;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;
import java.util.Objects;

/**
 * Type-safe immutable wrapper around an enhanced image, explicitly tagged with
 * the {@link ProcessingPolicy#policyId()} and {@link ProcessingPolicy#fingerprint()}.
 *
 * <p>Prevents accidental double-enhancement by enforcing type separation between
 * raw {@link OriginalImage} inputs and processed {@link EnhancedImage} products.</p>
 */
public final class EnhancedImage {

    private final BufferedImage image;
    private final int width;
    private final int height;
    private final String policyId;
    private final String policyFingerprint;
    private final String sha256;

    public EnhancedImage(BufferedImage source, String policyId, String policyFingerprint) {
        Objects.requireNonNull(source, "Source image must not be null.");
        this.policyId = Objects.requireNonNull(policyId, "policyId must not be null.");
        this.policyFingerprint = Objects.requireNonNull(policyFingerprint, "policyFingerprint must not be null.");
        this.width = source.getWidth();
        this.height = source.getHeight();
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Image dimensions must be positive; got " + width + "x" + height);
        }
        // Defensive copy to prevent external mutation
        this.image = copyImage(source);
        this.sha256 = computePixelSha256(this.image);
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public String getPolicyId() {
        return policyId;
    }

    public String getPolicyFingerprint() {
        return policyFingerprint;
    }

    public String getSha256() {
        return sha256;
    }

    /**
     * Returns a defensive copy of the underlying enhanced image.
     */
    public BufferedImage getCopy() {
        return copyImage(image);
    }

    /**
     * Provides direct read-only access to the underlying BufferedImage raster for high-performance
     * pipeline components (e.g. ROI cropping and feature extraction). Callers must not mutate this buffer.
     */
    public BufferedImage getInternalBuffer() {
        return image;
    }

    public BufferedImage image() {
        return image;
    }

    public int getRGB(int x, int y) {
        return image.getRGB(x, y);
    }

    private static BufferedImage copyImage(BufferedImage src) {
        int type = src.getType() == BufferedImage.TYPE_CUSTOM ? BufferedImage.TYPE_INT_RGB : src.getType();
        BufferedImage copy = new BufferedImage(src.getWidth(), src.getHeight(), type);
        Graphics2D g = copy.createGraphics();
        g.drawImage(src, 0, 0, null);
        g.dispose();
        return copy;
    }

    private static String computePixelSha256(BufferedImage img) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            int w = img.getWidth();
            int h = img.getHeight();
            ByteBuffer buffer = ByteBuffer.allocate(w * 4);
            for (int y = 0; y < h; y++) {
                buffer.clear();
                for (int x = 0; x < w; x++) {
                    buffer.putInt(img.getRGB(x, y));
                }
                buffer.flip();
                md.update(buffer);
            }
            byte[] digest = md.digest();
            StringBuilder sb = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                sb.append(String.format(Locale.ROOT, "%02x", b & 0xff));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }
    }
}
