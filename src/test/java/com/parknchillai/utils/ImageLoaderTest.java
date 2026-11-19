package com.parknchillai.utils;

import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link ImageLoader} covering path validation, empty/corrupt file rejection,
 * minimum size rejection, format detection ({@code PNG}/{@code JPEG}), {@link ImageLoader.LoadedImageMetadata},
 * and automatic resolution normalization ({@code <= 1280px}).
 */
class ImageLoaderTest {

    @Test
    void nullOrBlankPathThrowsIllegalArgumentException() {
        assertThrows(IllegalArgumentException.class, () -> ImageLoader.load((String) null));
        assertThrows(IllegalArgumentException.class, () -> ImageLoader.load("   "));
        assertThrows(IllegalArgumentException.class, () -> ImageLoader.load((File) null));
    }

    @Test
    void missingEmptyOrCorruptImageFileThrowsIOException() throws IOException {
        assertThrows(IOException.class, () -> ImageLoader.load("non_existent_parking_lot_12345.png"));

        Path empty = Files.createTempFile("empty_img", ".png");
        assertThrows(IOException.class, () -> ImageLoader.loadWithMetadata(empty.toFile()));
        Files.deleteIfExists(empty);

        Path corrupt = Files.createTempFile("corrupt_img", ".png");
        Files.writeString(corrupt, "not a valid png binary");
        assertThrows(IOException.class, () -> ImageLoader.load(corrupt.toFile()));
        Files.deleteIfExists(corrupt);
    }

    @Test
    void imageSmallerThan10x10ThrowsIllegalArgumentException() throws IOException {
        BufferedImage tiny = new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB);
        Path tinyFile = Files.createTempFile("tiny_img", ".png");
        ImageIO.write(tiny, "png", tinyFile.toFile());

        assertThrows(IllegalArgumentException.class, () -> ImageLoader.load(tinyFile.toFile()));
        Files.deleteIfExists(tinyFile);
    }

    @Test
    void oversizedImageIsAutomaticallyNormalizedToMax1280PxAndReportsMetadata() throws IOException {
        BufferedImage oversized = new BufferedImage(1600, 1200, BufferedImage.TYPE_INT_RGB);
        Path oversizedFile = Files.createTempFile("oversized_img", ".png");
        ImageIO.write(oversized, "png", oversizedFile.toFile());

        ImageLoader.LoadedImageMetadata meta = ImageLoader.loadWithMetadata(oversizedFile.toFile());
        assertEquals("PNG", meta.formatName());
        assertEquals(1600, meta.originalWidth());
        assertEquals(1200, meta.originalHeight());
        assertEquals(1280, meta.processedWidth());
        assertEquals(960, meta.processedHeight());
        assertEquals(0.80, meta.scaleX(), 1e-4);
        assertEquals(0.80, meta.scaleY(), 1e-4);
        assertTrue(meta.wasResized());
        assertFalse(meta.perspectiveWarped());
        assertTrue(meta.describeSummary().contains("1600×1200 px -> Normalized: 1280×960 px"));

        ImageLoader.LoadedImageMetadata warpedMeta = meta.withPerspectiveWarp(meta.image());
        assertTrue(warpedMeta.perspectiveWarped());
        assertTrue(warpedMeta.describeSummary().contains("4-Corner Perspective Rectified"));

        Files.deleteIfExists(oversizedFile);
    }

    @Test
    void jpegImageLoadsWithCorrectFormatMetadata() throws IOException {
        BufferedImage jpgImg = new BufferedImage(240, 180, BufferedImage.TYPE_INT_RGB);
        Path jpgFile = Files.createTempFile("parking_test", ".jpg");
        ImageIO.write(jpgImg, "jpg", jpgFile.toFile());

        ImageLoader.LoadedImageMetadata meta = ImageLoader.loadWithMetadata(jpgFile.toFile());
        assertTrue(meta.formatName().contains("JPEG") || meta.formatName().contains("JPG"));
        assertEquals(240, meta.originalWidth());
        assertEquals(180, meta.processedHeight());
        assertFalse(meta.wasResized());

        Files.deleteIfExists(jpgFile);
    }
}
