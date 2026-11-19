package com.parknchillai.ai;

import com.parknchillai.app.AppState;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.imageprocessing.ProcessingPolicy;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests authoritative schema and enhancement policy consistency and fail-closed rejection
 * (WS-6 requirement).
 */
public class SchemaConsistencyAndRejectionTest {

    @Test
    @DisplayName("WS-6.1: Authoritative schema and policy constants")
    void testAuthoritativeConstants() {
        assertEquals("v2.3-10feat-enhanced-v1", ProcessingPolicy.CANONICAL_SCHEMA_ID);
        assertEquals("ivp-enhance-v1", ProcessingPolicy.CANONICAL_POLICY_ID);
        assertEquals(ProcessingPolicy.CANONICAL_SCHEMA_ID, FeatureVector.ENHANCED_SCHEMA_VERSION);
        assertEquals(ProcessingPolicy.CANONICAL_SCHEMA_ID, FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION);
    }

    @Test
    @DisplayName("WS-6.2: Strict model loading rejects schema mismatch with SchemaMismatchException")
    void testStrictModelLoadRejectsMismatch() throws IOException {
        Path tempModel = Files.createTempFile("mismatch_model", ".model");
        String payload = """
            format=PARKNCHILL_CART_MODEL_V23
            schemaVersion=v2.3-10feat-raw-v1
            featureCount=10
            maxDepth=4
            datasetSource=Raw Baseline Set
            savedAt=2026-10-22T00:00:00Z
            policyId=ivp-raw-v1
            policyFingerprint=0000000000000000000000000000000000000000000000000000000000000000
            trainMin=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
            trainMax=255.0,1.0,1.0,1.0,1.0,255.0,1.0,1.0,1.0,1.0
            NODE|0|LEAF|AVAILABLE|0|10
            """;
        String sha = ModelSerializer.computeSha256(payload);
        Files.writeString(tempModel, payload + "sha256=" + sha + "\n");

        SchemaMismatchException ex = assertThrows(SchemaMismatchException.class, () -> {
            ModelSerializer.loadModelStrict(tempModel);
        });

        assertEquals(ProcessingPolicy.CANONICAL_SCHEMA_ID, ex.getExpectedSchema());
        assertEquals("v2.3-10feat-raw-v1", ex.getActualSchema());
        assertTrue(ex.getMessage().contains("Model failed schema/policy compatibility"));
    }

    @Test
    @DisplayName("WS-6.3: AppState rejects activating incompatible 10-feature model with SchemaMismatchException")
    void testAppStateRejectsActivatingIncompatibleModel() throws IOException {
        Path tempModel = Files.createTempFile("mismatch_model_appstate", ".model");
        String payload = """
            format=PARKNCHILL_CART_MODEL_V23
            schemaVersion=v2.3-10feat-raw-v1
            featureCount=10
            maxDepth=4
            datasetSource=Raw Baseline Set
            savedAt=2026-10-22T00:00:00Z
            policyId=ivp-raw-v1
            policyFingerprint=0000000000000000000000000000000000000000000000000000000000000000
            trainMin=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
            trainMax=255.0,1.0,1.0,1.0,1.0,255.0,1.0,1.0,1.0,1.0
            NODE|0|LEAF|AVAILABLE|0|10
            """;
        String sha = ModelSerializer.computeSha256(payload);
        Files.writeString(tempModel, payload + "sha256=" + sha + "\n");

        ModelSerializer.LoadedModelBundle bundle = ModelSerializer.loadModel(tempModel);
        assertFalse(bundle.compatibleTenFeatureModel());

        AppState state = new AppState(new com.parknchillai.app.ParkingAnalysisService(new DecisionTreePredictor(new DecisionTreeModel())));
        assertThrows(SchemaMismatchException.class, () -> {
            state.applyLoadedModelBundle(bundle);
        });
    }

    @Test
    @DisplayName("WS-6.4: AppState keeps legacy 8-feature model isolated in legacy container")
    void testAppStateIsolatesLegacyEightFeatureModel() throws IOException {
        Path tempModel = Files.createTempFile("legacy_8feat_model", ".model");
        String payload = """
            format=PARKNCHILL_CART_MODEL_V23
            schemaVersion=v2.2-8feat
            featureCount=8
            maxDepth=3
            datasetSource=Legacy Corpus
            savedAt=2026-10-22T00:00:00Z
            policyId=legacy-none
            policyFingerprint=
            trainMin=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
            trainMax=255.0,1.0,1.0,1.0,1.0,255.0,1.0,1.0
            NODE|0|LEAF|AVAILABLE|0|10
            """;
        String sha = ModelSerializer.computeSha256(payload);
        Files.writeString(tempModel, payload + "sha256=" + sha + "\n");

        ModelSerializer.LoadedModelBundle bundle = ModelSerializer.loadModel(tempModel);
        assertFalse(bundle.compatibleTenFeatureModel());
        assertEquals(8, bundle.featureCount());

        AppState state = new AppState(new com.parknchillai.app.ParkingAnalysisService(new DecisionTreePredictor(new DecisionTreeModel())));
        assertDoesNotThrow(() -> state.applyLoadedModelBundle(bundle));
        assertFalse(state.isCompatibleTenFeatureModelActive());
        assertTrue(state.isLegacyEightFeatureModelPresent());
        assertEquals(ModelSerializer.LEGACY_EIGHT_FEATURE_MESSAGE, state.getLegacyModelExplanation());
    }

    @Test
    @DisplayName("WS-1: Legacy 8-feature dataset cannot be evaluated directly by 10-feature model without explicit rejection")
    void testLegacyDatasetRejectedByTenFeaturePredictor() throws IOException {
        DatasetLoader.Dataset legacyDataset = DatasetLoader.load("data/labels/features.csv");
        assertEquals(8, legacyDataset.featureCount());
        assertEquals(48, legacyDataset.size());

        // Train a 10-feature model
        double[][] dummyTenTrain = new double[][] {
            new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            new double[]{2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
        };
        com.parknchillai.parking.SlotStatus[] dummyLabels = new com.parknchillai.parking.SlotStatus[]{
            com.parknchillai.parking.SlotStatus.OCCUPIED, com.parknchillai.parking.SlotStatus.AVAILABLE
        };
        DecisionTreeModel tenFeatureModel = new DecisionTreeTrainer(2, 1).train(dummyTenTrain, dummyLabels);
        assertEquals(10, tenFeatureModel.getFeatureCount());

        // Attempting to predict on 8-feature data with 10-feature model throws IllegalArgumentException
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> {
            tenFeatureModel.predict(legacyDataset.features[0]);
        });
        assertTrue(ex.getMessage().contains("Expected 10 features; got 8"));
    }
}
