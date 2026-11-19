package com.parknchillai.utils;

import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class SampleDataGeneratorTest {

    @Test
    void generateDemoParkingLotCreatesNonEmptyImage() {
        BufferedImage img = SampleDataGenerator.generateDemoParkingLot();
        assertNotNull(img);
        assertEquals(380, img.getWidth());
        assertEquals(350, img.getHeight());
    }

    @Test
    void ensureSampleImageExistsWritesFile() throws IOException {
        String path = "data/sample/sample_parking_lot.png";
        File file = SampleDataGenerator.ensureSampleImageExists(path);
        assertTrue(file.exists());
        assertTrue(file.length() > 0);
    }
}
