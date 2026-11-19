package com.parknchillai.ai;

import com.parknchillai.features.FeatureExtractor;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.imageprocessing.EnhancedImage;
import com.parknchillai.imageprocessing.ImageEnhancementService;
import com.parknchillai.imageprocessing.OriginalImage;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.RoiExtractor;
import com.parknchillai.parking.SlotStatus;
import com.parknchillai.utils.ConfigLoader;
import com.parknchillai.utils.SampleDataGenerator;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DatasetLoaderTest {

    @Test
    void loadParsesValidCsvFile() throws IOException {
        Path tempFile = Files.createTempFile("dataset_test", ".csv");
        String content = """
            meanIntensity,foregroundRatio,darkPixelRatio,edgeDensity,objectArea,label
            100.0,0.5,0.2,0.3,0.4,OCCUPIED
            200.0,0.1,0.01,0.02,0.05,AVAILABLE
            """;
        Files.writeString(tempFile, content);

        DatasetLoader.Dataset dataset = DatasetLoader.load(tempFile.toString());
        assertEquals(2, dataset.size());
        assertEquals(SlotStatus.OCCUPIED, dataset.labels[0]);
        assertEquals(SlotStatus.AVAILABLE, dataset.labels[1]);
        assertEquals(100.0, dataset.features[0][0]);
        assertEquals(200.0, dataset.features[1][0]);

        Files.deleteIfExists(tempFile);
    }

    @Test
    void trainTestSplitWithSeedIsDeterministicAndStratified() throws IOException {
        Path tempFile = Files.createTempFile("dataset_split_test", ".csv");
        StringBuilder sb = new StringBuilder("meanIntensity,foregroundRatio,darkPixelRatio,edgeDensity,objectArea,label\n");
        for (int i = 0; i < 10; i++) {
            sb.append("100.0,0.7,0.3,0.4,0.6,OCCUPIED\n");
        }
        for (int i = 0; i < 10; i++) {
            sb.append("200.0,0.05,0.02,0.03,0.04,AVAILABLE\n");
        }
        Files.writeString(tempFile, sb.toString());

        DatasetLoader.Dataset dataset = DatasetLoader.load(tempFile.toString());
        DatasetLoader.Dataset[] splits = dataset.trainTestSplit(0.5, 42L);

        assertEquals(10, splits[0].size(), "Train set should have 10 samples");
        assertEquals(10, splits[1].size(), "Test set should have 10 samples");

        int occInTrain = 0;
        for (SlotStatus s : splits[0].labels) if (s == SlotStatus.OCCUPIED) occInTrain++;
        assertEquals(5, occInTrain, "Stratified 50% split must place exactly 5 of 10 OCCUPIED samples in train set");

        Files.deleteIfExists(tempFile);
    }

    @Test
    void groupAwareSplitProducesStrictlyDisjointCaptureGroups() {
        double[][] feats = new double[15][8];
        SlotStatus[] lbls = new SlotStatus[15];
        List<String> groups = new ArrayList<>(15);
        for (int i = 0; i < 15; i++) {
            feats[i] = new double[]{80.0, 0.3, 0.5, 0.4, 0.3, 60.0, 0.2, 0.25};
            lbls[i] = (i % 2 == 0) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
            groups.add("LOT_GROUP_" + (i / 3)); // 5 distinct capture groups (0..4)
        }
        DatasetLoader.Dataset ds = new DatasetLoader.Dataset(feats, lbls);
        DatasetLoader.GroupedPartitions parts = DatasetLoader.groupAwareSplit(ds, groups, 42L);

        assertTrue(parts.groupDisjoint());
        assertTrue(Collections.disjoint(parts.trainGroups(), parts.validationGroups()));
        assertTrue(Collections.disjoint(parts.trainGroups(), parts.testGroups()));
        assertTrue(Collections.disjoint(parts.validationGroups(), parts.testGroups()));
        assertEquals(15, parts.train().size() + parts.validation().size() + parts.test().size());
        parts.assertValidInvariants(15);
    }

    @Test
    void fewerThanThreeGroupsIsMarkedAsFallbackNonGroupDisjointAndRejectsBlankGroups() {
        double[][] feats = new double[8][8];
        SlotStatus[] lbls = new SlotStatus[8];
        List<String> twoGroups = new ArrayList<>(8);
        for (int i = 0; i < 8; i++) {
            feats[i] = new double[]{80.0, 0.3, 0.5, 0.4, 0.3, 60.0, 0.2, 0.25};
            lbls[i] = (i % 2 == 0) ? SlotStatus.OCCUPIED : SlotStatus.AVAILABLE;
            twoGroups.add(i < 4 ? "LOT_G1" : "LOT_G2");
        }
        DatasetLoader.Dataset ds = new DatasetLoader.Dataset(feats, lbls);
        DatasetLoader.GroupedPartitions fallback = DatasetLoader.groupAwareSplit(ds, twoGroups, 42L);

        assertFalse(fallback.groupDisjoint(), "Fewer than 3 capture groups must set groupDisjoint = false");
        assertEquals(8, fallback.train().size() + fallback.validation().size() + fallback.test().size());

        List<String> blankGroups = new ArrayList<>(twoGroups);
        blankGroups.set(2, "   ");
        assertThrows(IllegalArgumentException.class, () -> DatasetLoader.groupAwareSplit(ds, blankGroups, 42L));
    }

    @Test
    void allTrainManifestRequiresExplicitPartitionInitializationAndPersistsAssignmentsAndSplitId() throws IOException {
        Path tempDir = Files.createTempDirectory("pnc_part_init_");
        Path fFile = tempDir.resolve("features.csv");
        Path mFile = tempDir.resolve("manifest.csv");
        Files.copy(Path.of("data/labels/features.csv"), fFile, StandardCopyOption.REPLACE_EXISTING);

        // Build an all-TRAIN manifest (where VALIDATION and TEST are empty)
        DatasetLoader.Dataset ds = DatasetLoader.load(fFile.toString());
        List<DatasetManifest.ManifestRecord> origMan = DatasetManifest.loadManifest("data/labels/manifest.csv");
        List<DatasetManifest.ManifestRecord> allTrainMan = new ArrayList<>(origMan.size());
        for (DatasetManifest.ManifestRecord r : origMan) {
            allTrainMan.add(new DatasetManifest.ManifestRecord(
                r.sampleId(), r.sourceImageId(), r.captureGroup(),
                r.slotId(), r.x(), r.y(), r.width(), r.height(),
                r.features(), r.label(), r.schemaVersion(),
                r.reviewStatus(), r.provenance(), "TRAIN", r.timestamp()
            ));
        }
        DatasetManifest.saveManifest(mFile.toString(), allTrainMan);

        // 1. Calling loadPartitions / buildValidatedPartitions on an all-TRAIN manifest must FAIL CLOSED
        assertThrows(IllegalStateException.class, () ->
            DatasetLoader.loadPartitions(fFile.toString(), mFile.toString())
        );

        // 2. Explicit initializeAndPersistPartitions persists TRAIN, VALIDATION, and TEST to manifest.csv
        DatasetLoader.GroupedPartitions initialized = DatasetLoader.initializeAndPersistPartitions(
            fFile.toString(), mFile.toString(), 42L
        );
        assertTrue(initialized.groupDisjoint());
        assertNotNull(initialized.splitId());
        assertTrue(initialized.splitId().startsWith("SPLIT-GD-"));

        // 3. Repeated loading returns the exact same partition assignments and splitId without recalculating
        DatasetLoader.GroupedPartitions reloaded = DatasetLoader.loadPartitions(fFile.toString(), mFile.toString());
        assertEquals(initialized.splitId(), reloaded.splitId());
        assertEquals(initialized.trainGroups(), reloaded.trainGroups());
        assertEquals(initialized.validationGroups(), reloaded.validationGroups());
        assertEquals(initialized.testGroups(), reloaded.testGroups());
        assertEquals(initialized.test().size(), reloaded.test().size());

        // 4. Appending a new TRAIN sample from a new capture group leaves VALIDATION and TEST untouched
        FeatureVector newFv = new FeatureVector(new double[]{88.0, 0.44, 0.51, 0.46, 0.40, 110.0, 0.52, 0.35});
        DatasetManifest.recordUserLabeledSample(
            fFile.toString(), mFile.toString(),
            "new_train_cam.png", "LOT_NEW_TRAIN_GRP", new ParkingSlot(1, 10, 10, 80, 100),
            newFv, SlotStatus.OCCUPIED, "REV_1", null, false
        );
        DatasetLoader.GroupedPartitions afterAdd = DatasetLoader.loadPartitions(fFile.toString(), mFile.toString());
        assertEquals(initialized.train().size() + 1, afterAdd.train().size());
        assertEquals(initialized.validation().size(), afterAdd.validation().size());
        assertEquals(initialized.test().size(), afterAdd.test().size());
        assertEquals(initialized.validationGroups(), afterAdd.validationGroups());
        assertEquals(initialized.testGroups(), afterAdd.testGroups());

        Files.walk(tempDir).sorted((a, b) -> b.compareTo(a)).forEach(p -> {
            try { Files.deleteIfExists(p); } catch (IOException ignored) {}
        });
    }

    @Test
    void duplicatedGroupAcrossPartitionsOrMissingManifestRecordFailsClosed() {
        DatasetLoader.Dataset ds = DatasetLoader.load("data/labels/features.csv");
        List<DatasetManifest.ManifestRecord> manifest = new ArrayList<>(
            DatasetManifest.loadOrSync("data/labels/features.csv", "data/labels/manifest.csv"));

        List<DatasetManifest.ManifestRecord> missingOne = manifest.subList(0, manifest.size() - 1);
        assertThrows(IllegalStateException.class,
            () -> DatasetLoader.buildValidatedPartitions(ds, missingOne, 42L));

        DatasetManifest.ManifestRecord first = manifest.get(0);
        manifest.set(0, new DatasetManifest.ManifestRecord(
            first.sampleId(), first.sourceImageId(),
            "LOT_TEST_OCC_HOLD",
            first.slotId(), first.x(), first.y(), first.width(), first.height(),
            first.features(), first.label(), first.schemaVersion(),
            first.reviewStatus(), first.provenance(), "TRAIN", first.timestamp()
        ));
        assertThrows(IllegalStateException.class,
            () -> DatasetLoader.buildValidatedPartitions(ds, manifest, 42L));
    }

    @Test
    void datasetValidatorRejectsOutOfBoundsFeaturesAndSingleClassDatasets() {
        double[][] invalidBounds = {
            {100.0, 1.45, 0.2, 0.3, 0.4, 50.0, 0.1, 0.2},
            {100.0, 0.50, 0.2, 0.3, 0.4, 50.0, 0.1, 0.2},
            {200.0, 0.05, 0.1, 0.1, 0.1, 10.0, 0.0, 0.01},
            {200.0, 0.05, 0.1, 0.1, 0.1, 10.0, 0.0, 0.01}
        };
        SlotStatus[] labels = {
            SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.AVAILABLE, SlotStatus.AVAILABLE
        };
        DatasetLoader.Dataset badDs = new DatasetLoader.Dataset(invalidBounds, labels);
        DatasetValidator.ValidationSummary summary = DatasetValidator.validate(badDs, null);
        assertFalse(summary.validForTraining());
        assertThrows(IllegalArgumentException.class, () -> DatasetValidator.validateOrThrow(badDs, null));

        SlotStatus[] singleClass = {
            SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED, SlotStatus.OCCUPIED
        };
        double[][] validFeats = {
            {100.0, 0.50, 0.2, 0.3, 0.4, 50.0, 0.1, 0.2},
            {105.0, 0.52, 0.2, 0.3, 0.4, 50.0, 0.1, 0.2},
            {110.0, 0.54, 0.2, 0.3, 0.4, 50.0, 0.1, 0.2},
            {115.0, 0.56, 0.2, 0.3, 0.4, 50.0, 0.1, 0.2}
        };
        DatasetLoader.Dataset oneClassDs = new DatasetLoader.Dataset(validFeats, singleClass);
        assertFalse(DatasetValidator.validate(oneClassDs, null).validForTraining());
    }

    @Test
    void defaultDatasetDemoRowsMatchActualFeatureExtractorOutputs() throws Exception {
        File demoFile = SampleDataGenerator.ensureSampleImageExists(SampleDataGenerator.DEFAULT_SAMPLE_IMAGE_PATH);
        BufferedImage demoImg = ImageIO.read(demoFile);
        EnhancedImage enhancedImg = ImageEnhancementService.getDefault().enhance(OriginalImage.of(demoImg));
        List<ParkingSlot> slots = ConfigLoader.loadSlotsFromCsv("data/config/slots.csv");
        DatasetLoader.Dataset ds = DatasetLoader.load("data/labels/features.csv");

        int[] occupiedSlotIndices = {0, 5, 6, 11};
        for (int r = 0; r < occupiedSlotIndices.length; r++) {
            ParkingSlot slot = slots.get(occupiedSlotIndices[r]);
            BufferedImage rawCrop = RoiExtractor.extract(demoImg, slot);
            BufferedImage enhancedCrop = RoiExtractor.extract(enhancedImg.image(), slot);
            FeatureVector actual = FeatureExtractor.extract(rawCrop, enhancedCrop, false);
            for (int f = 0; f < FeatureVector.FEATURE_COUNT; f++) {
                assertEquals(actual.get(f), ds.features[r][f], 1e-5,
                    "Dataset row " + r + " feature " + f + " must match actual FeatureExtractor output for Slot #" + slot.getId());
            }
        }
    }

    @Test
    void appendSampleDeduplicatesIdenticalFeatureRowsAndUpdatesLabel() throws IOException {
        Path tempFile = Files.createTempFile("dataset_dedup", ".csv");
        Files.deleteIfExists(tempFile);

        FeatureVector fv = new FeatureVector(new double[]{80.0, 0.45, 0.50, 0.48, 0.42, 95.0, 0.40, 0.38});
        DatasetLoader.appendSample(tempFile.toString(), fv, SlotStatus.OCCUPIED);
        DatasetLoader.appendSample(tempFile.toString(), fv, SlotStatus.OCCUPIED);

        DatasetLoader.Dataset ds1 = DatasetLoader.load(tempFile.toString());
        assertEquals(1, ds1.size(), "Duplicate appendSample call must not add a second identical row");
        assertEquals(SlotStatus.OCCUPIED, ds1.labels[0]);

        DatasetLoader.appendSample(tempFile.toString(), fv, SlotStatus.AVAILABLE);
        DatasetLoader.Dataset ds2 = DatasetLoader.load(tempFile.toString());
        assertEquals(1, ds2.size());
        assertEquals(SlotStatus.AVAILABLE, ds2.labels[0]);

        Files.deleteIfExists(tempFile);
    }

    @Test
    void invalidFieldCountThrowsException() throws IOException {
        Path tempFile = Files.createTempFile("dataset_bad", ".csv");
        Files.writeString(tempFile, "100.0,0.5,OCCUPIED\n");

        assertThrows(IllegalArgumentException.class,
            () -> DatasetLoader.load(tempFile.toString()));

        Files.deleteIfExists(tempFile);
    }

    @Test
    void nanAndInfinityValuesThrowException() throws IOException {
        Path nanFile = Files.createTempFile("dataset_nan", ".csv");
        Files.writeString(nanFile, "100.0,NaN,0.2,0.3,0.4,OCCUPIED\n");
        assertThrows(IllegalArgumentException.class,
            () -> DatasetLoader.load(nanFile.toString()));
        Files.deleteIfExists(nanFile);

        Path infFile = Files.createTempFile("dataset_inf", ".csv");
        Files.writeString(infFile, "100.0,Infinity,0.2,0.3,0.4,AVAILABLE\n");
        assertThrows(IllegalArgumentException.class,
            () -> DatasetLoader.load(infFile.toString()));
        Files.deleteIfExists(infFile);
    }
}
