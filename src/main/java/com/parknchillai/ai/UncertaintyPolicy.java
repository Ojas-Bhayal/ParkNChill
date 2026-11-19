package com.parknchillai.ai;

/**
 * Configurable, provisional uncertainty policy governing when a classification
 * prediction must be flagged as {@link com.parknchillai.parking.SlotStatus#NEEDS_REVIEW}
 * rather than forced into a binary {@code OCCUPIED} or {@code AVAILABLE} state.
 *
 * <p>Because the bundled dataset is a small (48-sample) mixed demo/procedural corpus
 * without large-scale probability calibration, this policy is explicitly marked
 * {@code provisional = true} and exposes user-configurable thresholds for:
 * <ul>
 *   <li>Minimum terminal leaf class purity ({@code minLeafPurity}, default {@code 0.75})</li>
 *   <li>Minimum training samples supporting the leaf ({@code minLeafSupport}, default {@code 2})</li>
 *   <li>Maximum number of features allowed outside the observed training domain before
 *       flagging out-of-distribution review ({@code maxOutOfDomainFeatures}, default {@code 5})</li>
 *   <li>Relative domain tolerance margin ({@code domainMarginFraction}, default {@code 0.25})</li>
 * </ul>
 */
public final class UncertaintyPolicy {

    private final double minLeafPurity;
    private final int minLeafSupport;
    private final int maxOutOfDomainFeatures;
    private final double domainMarginFraction;
    private final boolean provisional;

    public UncertaintyPolicy(double minLeafPurity,
                             int minLeafSupport,
                             int maxOutOfDomainFeatures,
                             double domainMarginFraction,
                             boolean provisional) {
        if (!Double.isFinite(minLeafPurity) || minLeafPurity < 0.5 || minLeafPurity > 1.0) {
            throw new IllegalArgumentException("minLeafPurity must be in [0.5, 1.0]; got " + minLeafPurity);
        }
        if (minLeafSupport < 1) {
            throw new IllegalArgumentException("minLeafSupport must be >= 1; got " + minLeafSupport);
        }
        if (maxOutOfDomainFeatures < 1) {
            throw new IllegalArgumentException("maxOutOfDomainFeatures must be >= 1; got " + maxOutOfDomainFeatures);
        }
        if (!Double.isFinite(domainMarginFraction) || domainMarginFraction < 0.0) {
            throw new IllegalArgumentException("domainMarginFraction must be >= 0.0; got " + domainMarginFraction);
        }
        this.minLeafPurity          = minLeafPurity;
        this.minLeafSupport         = minLeafSupport;
        this.maxOutOfDomainFeatures = maxOutOfDomainFeatures;
        this.domainMarginFraction   = domainMarginFraction;
        this.provisional            = provisional;
    }

    /**
     * Default provisional policy derived from validation inspection:
     * requires leaf purity {@code >= 0.75}, leaf sample support {@code >= 2},
     * and flags inputs when {@code >= 5} of the 8 features fall outside the observed
     * training range expanded by {@code 25%} tolerance.
     */
    public static UncertaintyPolicy defaultProvisionalPolicy() {
        return new UncertaintyPolicy(0.75, 2, 5, 0.25, true);
    }

    /**
     * Strict domain policy useful for testing out-of-distribution detection.
     */
    public static UncertaintyPolicy strictDomainPolicy() {
        return new UncertaintyPolicy(0.85, 2, 2, 0.05, true);
    }

    public double getMinLeafPurity()          { return minLeafPurity; }
    public int getMinLeafSupport()            { return minLeafSupport; }
    public int getMaxOutOfDomainFeatures()    { return maxOutOfDomainFeatures; }
    public double getDomainMarginFraction()   { return domainMarginFraction; }
    public boolean isProvisional()            { return provisional; }
}
