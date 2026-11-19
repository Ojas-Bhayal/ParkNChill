package com.parknchillai.imageprocessing;

import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

class ColorSpaceConverterTest {

    @Test
    void achromaticGrayAndShadowedAsphaltHaveZeroSaturation() {
        BufferedImage asphaltWithShadow = new BufferedImage(20, 20, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 20; y++) {
            for (int x = 0; x < 20; x++) {
                int v = (x < 10) ? 120 : 35;
                asphaltWithShadow.setRGB(x, y, (v << 16) | (v << 8) | v);
            }
        }

        BufferedImage satMap = ColorSpaceConverter.toSaturationMap(asphaltWithShadow);
        for (int y = 0; y < 20; y++) {
            for (int x = 0; x < 20; x++) {
                assertEquals(0, satMap.getRaster().getSample(x, y, 0),
                    "Achromatic sunlit or shadowed pavement must have 0 HSV saturation");
            }
        }
    }

    @Test
    void purePaintedVehicleColorsHaveHighSaturation() {
        BufferedImage carImg = new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = carImg.createGraphics();
        g.setColor(new Color(220, 20, 20));
        g.fillRect(0, 0, 10, 10);
        g.dispose();

        BufferedImage satMap = ColorSpaceConverter.toSaturationMap(carImg);
        int sat = satMap.getRaster().getSample(5, 5, 0);
        assertEquals(232, sat, 1, "Crimson red vehicle should have saturation ~232");
    }

    @Test
    void valueMapMatchesMaxRgbChannel() {
        BufferedImage img = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
        img.setRGB(0, 0, (40 << 16) | (180 << 8) | 90);
        BufferedImage valMap = ColorSpaceConverter.toValueMap(img);
        assertEquals(180, valMap.getRaster().getSample(0, 0, 0));
    }

    @Test
    void nullInputThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> ColorSpaceConverter.toSaturationMap(null));
        assertThrows(IllegalArgumentException.class, () -> ColorSpaceConverter.toValueMap(null));
    }
}
