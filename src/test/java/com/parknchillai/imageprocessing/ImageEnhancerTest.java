package com.parknchillai.imageprocessing;

import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

class ImageEnhancerTest {
    @Test
    void enhancementReturnsSameSizeCopyAndDoesNotMutateOriginal() {
        BufferedImage original = new BufferedImage(40, 20, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < original.getHeight(); y++) {
            for (int x = 0; x < original.getWidth(); x++) {
                int value = 70 + (x % 10) * 3;
                original.setRGB(x, y, (value << 16) | (value << 8) | value);
            }
        }
        int before = original.getRGB(5, 5);

        BufferedImage enhanced = ImageEnhancer.autoEnhance(original);

        assertNotSame(original, enhanced);
        assertEquals(original.getWidth(), enhanced.getWidth());
        assertEquals(original.getHeight(), enhanced.getHeight());
        assertEquals(before, original.getRGB(5, 5), "Enhancement must not modify source pixels");
        assertNotEquals(original.getRGB(5, 5), enhanced.getRGB(5, 5),
            "Low-contrast input should be enhanced");
    }

    @Test
    void nullInputIsRejectedClearly() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
            () -> ImageEnhancer.autoEnhance(null));
        assertTrue(ex.getMessage().contains("must not be null"));
    }
}
