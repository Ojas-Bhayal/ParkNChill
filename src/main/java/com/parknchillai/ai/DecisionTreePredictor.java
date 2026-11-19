package com.parknchillai.ai;

import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.SlotStatus;

/**
 * Wraps a trained {@link DecisionTreeModel} and provides a typed prediction
 * API that accepts {@link FeatureVector} objects (adapting seamlessly across
 * 5-feature, 8-feature, and 10-feature detection-augmented schemas), evaluates
 * {@link UncertaintyPolicy} rules, and produces {@link PredictionResult} and
 * {@link DecisionTreeModel.DecisionTrace} objects.
 */
public class DecisionTreePredictor {

    private final DecisionTreeModel model;
    private UncertaintyPolicy uncertaintyPolicy;

    public DecisionTreePredictor(DecisionTreeModel model) {
        this(model, UncertaintyPolicy.defaultProvisionalPolicy());
    }

    public DecisionTreePredictor(DecisionTreeModel model, UncertaintyPolicy uncertaintyPolicy) {
        if (model == null) throw new IllegalArgumentException("Model must not be null.");
        this.model = model;
        this.uncertaintyPolicy = uncertaintyPolicy != null
            ? uncertaintyPolicy
            : UncertaintyPolicy.defaultProvisionalPolicy();
    }

    public SlotStatus predict(FeatureVector features) {
        if (features == null) throw new IllegalArgumentException("FeatureVector must not be null.");
        double[] arr = adaptFeatures(features);
        return model.predict(arr);
    }

    /**
     * Predicts occupancy status while honoring the active {@link UncertaintyPolicy},
     * returning {@link SlotStatus#NEEDS_REVIEW} if the feature vector violates domain
     * or leaf-reliability criteria.
     */
    public SlotStatus predictWithUncertainty(FeatureVector features) {
        if (features == null) return SlotStatus.NEEDS_REVIEW;
        DecisionTreeModel.DecisionTrace trace = trace(features);
        return trace.finalStatus();
    }

    public double predictConfidence(FeatureVector features) {
        if (features == null) throw new IllegalArgumentException("FeatureVector must not be null.");
        double[] arr = adaptFeatures(features);
        return model.predictConfidence(arr);
    }

    /**
     * Predicts the occupancy status (routing uncertain predictions to {@link SlotStatus#NEEDS_REVIEW}
     * under the active {@link UncertaintyPolicy}) and leaf purity for a specific parking slot.
     */
    public PredictionResult predictSlot(int slotId, FeatureVector features) {
        if (features == null) throw new IllegalArgumentException("FeatureVector must not be null.");
        DecisionTreeModel.DecisionTrace trace = trace(features);
        return new PredictionResult(slotId, trace.finalStatus(), trace.leaf().leafPurity());
    }

    /**
     * Returns the full structured {@link DecisionTreeModel.DecisionTrace} for {@code features}.
     */
    public DecisionTreeModel.DecisionTrace trace(FeatureVector features) {
        if (features == null) {
            throw new IllegalArgumentException("FeatureVector must not be null.");
        }
        return model.traceDecision(adaptFeatures(features), uncertaintyPolicy);
    }

    public String explain(FeatureVector features) {
        if (features == null) throw new IllegalArgumentException("FeatureVector must not be null.");
        return model.explain(adaptFeatures(features));
    }

    private double[] adaptFeatures(FeatureVector features) {
        int expected = model.getFeatureCount();
        if (expected == FeatureVector.LEGACY_FEATURE_COUNT
                || expected == FeatureVector.FEATURE_COUNT
                || expected == FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT) {
            return features.toArray(expected);
        }
        return features.toArray();
    }

    public DecisionTreeModel getModel()                     { return model; }
    public UncertaintyPolicy getUncertaintyPolicy()         { return uncertaintyPolicy; }
    public void setUncertaintyPolicy(UncertaintyPolicy pol) {
        if (pol == null) throw new IllegalArgumentException("UncertaintyPolicy must not be null.");
        this.uncertaintyPolicy = pol;
    }
}
