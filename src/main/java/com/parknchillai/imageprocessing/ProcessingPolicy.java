package com.parknchillai.imageprocessing;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;
import java.util.Objects;

/**
 * Immutable canonical processing policy for image enhancement and feature extraction.
 *
 * <p>Contains exact, versioned parameters for contrast stretching, gamma correction,
 * channel handling, and coordinate mapping, along with a cryptographically verifiable
 * SHA-256 canonical fingerprint. All training, inference, preview, and review paths
 * must execute under an identical policy instance to guarantee numerical consistency.</p>
 */
public record ProcessingPolicy(
        String policyId,
        double lowPercentile,
        double highPercentile,
        int minContrastRange,
        double darkMeanCutoff,
        double darkGamma,
        double brightMeanCutoff,
        double brightGamma,
        double neutralGamma,
        double lumaR,
        double lumaG,
        double lumaB,
        String colorSpace,
        String cropInterpolation,
        String fingerprint
) {

    /**
     * Authoritative canonical feature-schema ID for ParkNChill v2.3 (WS-6 requirement).
     * All components must reference this single authoritative constant.
     */
    public static final String CANONICAL_SCHEMA_ID = "v2.3-10feat-enhanced-v1";

    /**
     * Authoritative canonical image-enhancement policy ID for ParkNChill v2.3 (WS-6 requirement).
     * All components must reference this single authoritative constant.
     */
    public static final String CANONICAL_POLICY_ID = "ivp-enhance-v1";

    public static final ProcessingPolicy CANONICAL = create(
            CANONICAL_POLICY_ID,
            0.02,
            0.98,
            24,
            82.0,
            0.86,
            190.0,
            1.08,
            1.0,
            0.2126,
            0.7152,
            0.0722,
            "sRGB->BT.709_Luma+HSV_Saturation",
            "BILINEAR"
    );

    public ProcessingPolicy {
        Objects.requireNonNull(policyId, "policyId must not be null");
        Objects.requireNonNull(colorSpace, "colorSpace must not be null");
        Objects.requireNonNull(cropInterpolation, "cropInterpolation must not be null");
        Objects.requireNonNull(fingerprint, "fingerprint must not be null");
        if (policyId.isBlank()) {
            throw new IllegalArgumentException("policyId must not be blank");
        }
    }

    public static ProcessingPolicy create(
            String policyId,
            double lowPercentile,
            double highPercentile,
            int minContrastRange,
            double darkMeanCutoff,
            double darkGamma,
            double brightMeanCutoff,
            double brightGamma,
            double neutralGamma,
            double lumaR,
            double lumaG,
            double lumaB,
            String colorSpace,
            String cropInterpolation) {

        String stableForm = computeStableRepresentation(
                policyId, lowPercentile, highPercentile, minContrastRange,
                darkMeanCutoff, darkGamma, brightMeanCutoff, brightGamma, neutralGamma,
                lumaR, lumaG, lumaB, colorSpace, cropInterpolation
        );
        String fp = computeSha256Hex(stableForm);
        return new ProcessingPolicy(
                policyId, lowPercentile, highPercentile, minContrastRange,
                darkMeanCutoff, darkGamma, brightMeanCutoff, brightGamma, neutralGamma,
                lumaR, lumaG, lumaB, colorSpace, cropInterpolation, fp
        );
    }

    public static String computeStableRepresentation(
            String policyId,
            double lowPercentile,
            double highPercentile,
            int minContrastRange,
            double darkMeanCutoff,
            double darkGamma,
            double brightMeanCutoff,
            double brightGamma,
            double neutralGamma,
            double lumaR,
            double lumaG,
            double lumaB,
            String colorSpace,
            String cropInterpolation) {
        return String.format(Locale.ROOT,
                "policyId=%s|pLow=%.4f|pHigh=%.4f|minRange=%d|darkCutoff=%.2f|darkGamma=%.4f|"
                        + "brightCutoff=%.2f|brightGamma=%.4f|neutralGamma=%.4f|"
                        + "luma=%.4f,%.4f,%.4f|cs=%s|crop=%s",
                policyId, lowPercentile, highPercentile, minContrastRange,
                darkMeanCutoff, darkGamma, brightMeanCutoff, brightGamma, neutralGamma,
                lumaR, lumaG, lumaB, colorSpace, cropInterpolation);
    }

    private static String computeSha256Hex(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(input.getBytes(StandardCharsets.UTF_8));
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
