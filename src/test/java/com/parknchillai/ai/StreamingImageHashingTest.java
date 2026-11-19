package com.parknchillai.ai;

import com.parknchillai.features.FeatureVector;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies buffered streaming source-image SHA-256 hashing and deterministic ID scoping
 * (WS-5 requirement).
 */
public class StreamingImageHashingTest {

    @BeforeEach
    void setUp() {
        DatasetManifest.resetHashComputationCountForTesting();
    }

    @Test
    @DisplayName("WS-5.1: Known-vector SHA-256 correctness against reference implementation")
    void testKnownVectorsAgainstReference() throws IOException, NoSuchAlgorithmException {
        // Vector 1: Empty file
        Path emptyFile = Files.createTempFile("sha_empty", ".bin");
        String emptyHash = DatasetManifest.computeFileSha256OrEmpty(emptyFile);
        assertEquals("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", emptyHash);
        assertEquals(independentDigest(emptyFile), emptyHash);

        // Vector 2: "abc"
        Path abcFile = Files.createTempFile("sha_abc", ".bin");
        Files.writeString(abcFile, "abc", StandardCharsets.UTF_8);
        String abcHash = DatasetManifest.computeFileSha256OrEmpty(abcFile);
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", abcHash);
        assertEquals(independentDigest(abcFile), abcHash);

        // Vector 3: Multi-buffer-size file (150 KiB > 64 KiB buffer)
        Path largeFile = Files.createTempFile("sha_large", ".bin");
        byte[] pattern = new byte[150 * 1024];
        for (int i = 0; i < pattern.length; i++) {
            pattern[i] = (byte) ((i * 31 + 17) & 0xFF);
        }
        Files.write(largeFile, pattern);
        String largeHash = DatasetManifest.computeFileSha256OrEmpty(largeFile);
        assertFalse(largeHash.isEmpty());
        assertEquals(independentDigest(largeFile), largeHash);
    }

    @Test
    @DisplayName("WS-5.2: Exactly one hash computation per image regardless of slot count (counting seam)")
    void testOneComputationPerImageRegardlessOfSlotCount() throws IOException {
        Path imageFile = Files.createTempFile("lot_image", ".png");
        byte[] payload = "SIMULATED_PNG_PIXELS_FOR_COUNTING_SEAM".getBytes(StandardCharsets.UTF_8);
        Files.write(imageFile, payload);

        assertEquals(0, DatasetManifest.getHashComputationCount());

        // First query calculates the hash and increments the seam counter
        String sha1 = DatasetManifest.computeFileSha256OrEmpty(imageFile);
        assertEquals(1, DatasetManifest.getHashComputationCount());

        // Simulate 50 slot sample ID computations for this single image
        for (int slotId = 1; slotId <= 50; slotId++) {
            String sampleId = DatasetManifest.computeSampleId(
                imageFile.toString(),
                "LOT_CAPTURE_GROUP",
                slotId,
                slotId * 10, 20, 80, 150,
                FeatureVector.ENHANCED_SCHEMA_VERSION
            );
            assertNotNull(sampleId);
            assertTrue(sampleId.contains("S" + String.format(Locale.ROOT, "%02d", slotId)));
        }

        // Must still be exactly 1 computation because of metadata caching and single hashing pass
        assertEquals(1, DatasetManifest.getHashComputationCount(),
            "Hash computation must be executed exactly once per image regardless of slot count");

        // Subsequent direct call for the same file also hits the cache
        String sha2 = DatasetManifest.computeFileSha256OrEmpty(imageFile);
        assertEquals(sha1, sha2);
        assertEquals(1, DatasetManifest.getHashComputationCount());
    }

    @Test
    @DisplayName("WS-5.3: Two different images with identical filename yield different sample IDs")
    void testTwoDifferentImagesWithSameFilenameYieldDifferentSampleIds() throws IOException {
        Path dirA = Files.createTempDirectory("lotA");
        Path dirB = Files.createTempDirectory("lotB");
        Path imgA = dirA.resolve("parking.png");
        Path imgB = dirB.resolve("parking.png");

        Files.writeString(imgA, "IMAGE_CONTENT_ALPHA_DIFFERENT_PIXELS");
        Files.writeString(imgB, "IMAGE_CONTENT_BETA_COMPLETELY_DIFFERENT_PIXELS");

        String sampleIdA = DatasetManifest.computeSampleId(
            imgA.toString(), "SHARED_GROUP", 1, 10, 10, 100, 200, FeatureVector.ENHANCED_SCHEMA_VERSION
        );
        String sampleIdB = DatasetManifest.computeSampleId(
            imgB.toString(), "SHARED_GROUP", 1, 10, 10, 100, 200, FeatureVector.ENHANCED_SCHEMA_VERSION
        );

        assertNotEquals(sampleIdA, sampleIdB,
            "Sample IDs must not collide even if filename and slot index are identical, because content hashes differ");
    }

    @Test
    @DisplayName("WS-5.4: Same bytes yield deterministic sample IDs across reloads")
    void testSameBytesYieldDeterministicIdsAcrossReload() throws IOException {
        Path imgFile = Files.createTempFile("parking_reload", ".png");
        Files.writeString(imgFile, "DETERMINISTIC_PERSISTENT_PIXEL_DATA");

        String sampleId1 = DatasetManifest.computeSampleId(
            imgFile.toString(), "GRP_A", 5, 15, 25, 90, 180, FeatureVector.ENHANCED_SCHEMA_VERSION
        );

        // Simulate reload: clear in-memory caches
        DatasetManifest.resetHashComputationCountForTesting();

        String sampleId2 = DatasetManifest.computeSampleId(
            imgFile.toString(), "GRP_A", 5, 15, 25, 90, 180, FeatureVector.ENHANCED_SCHEMA_VERSION
        );

        assertEquals(sampleId1, sampleId2,
            "Sample ID must be 100% deterministic across application reloads for identical file bytes");
    }

    private static String independentDigest(Path p) throws IOException, NoSuchAlgorithmException {
        byte[] bytes = Files.readAllBytes(p);
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] dig = md.digest(bytes);
        StringBuilder sb = new StringBuilder();
        for (byte b : dig) {
            sb.append(String.format(Locale.ROOT, "%02x", b & 0xff));
        }
        return sb.toString();
    }
}
