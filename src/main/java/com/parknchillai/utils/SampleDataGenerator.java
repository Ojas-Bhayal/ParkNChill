package com.parknchillai.utils;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Utility for generating synthetic realistic parking lot demonstration images
 * formatted to match the default {@code data/config/slots.csv} layout.
 */
public final class SampleDataGenerator {

    public static final String DEFAULT_SAMPLE_IMAGE_PATH = "data/sample/sample_parking_lot.png";

    private SampleDataGenerator() { /* utility class */ }

    /**
     * Generates a 380x350 parking lot image with 12 slots matching {@code data/config/slots.csv}.
     * Slots 1, 6, 7, and 12 are occupied with cars; other slots are available.
     *
     * @return a newly synthesized {@link BufferedImage}
     */
    public static BufferedImage generateDemoParkingLot() {
        int width = 380;
        int height = 350;
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // 1. Asphalt parking lot background with fine noise
        g.setColor(new Color(52, 58, 64));
        g.fillRect(0, 0, width, height);

        // Subtle asphalt grain texture
        for (int y = 0; y < height; y += 4) {
            for (int x = 0; x < width; x += 4) {
                int noise = ((x * 17 + y * 31) % 15) - 7;
                int base = Math.min(255, Math.max(0, 52 + noise));
                g.setColor(new Color(base, base + 2, base + 4));
                g.fillRect(x, y, 4, 4);
            }
        }

        // Slot definitions (matching slots.csv)
        int[][] slots = {
            {1, 10, 10, 80, 100},
            {2, 100, 10, 80, 100},
            {3, 190, 10, 80, 100},
            {4, 280, 10, 80, 100},
            {5, 10, 120, 80, 100},
            {6, 100, 120, 80, 100},
            {7, 190, 120, 80, 100},
            {8, 280, 120, 80, 100},
            {9, 10, 230, 80, 100},
            {10, 100, 230, 80, 100},
            {11, 190, 230, 80, 100},
            {12, 280, 230, 80, 100}
        };

        // Draw parking stall demarcations (white parking lines)
        g.setStroke(new BasicStroke(3.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        g.setColor(new Color(240, 240, 245));

        for (int[] s : slots) {
            int x = s[1];
            int y = s[2];
            int w = s[3];
            int h = s[4];

            // Draw white stall U-shape (parking bay lines)
            g.drawRect(x, y, w, h);

            // Draw painted slot numbers on the asphalt
            g.setFont(new Font("SansSerif", Font.BOLD, 12));
            g.setColor(new Color(180, 185, 190, 120));
            g.drawString(String.format("P-%02d", s[0]), x + 25, y + h - 12);
            g.setColor(new Color(240, 240, 245));
        }

        // 2. Render vehicles in occupied slots (1, 6, 7, 12)
        drawVehicle(g, 10, 10, 80, 100, new Color(220, 38, 38));    // Slot 1: Crimson Red
        drawVehicle(g, 100, 120, 80, 100, new Color(37, 99, 235));  // Slot 6: Royal Blue
        drawVehicle(g, 190, 120, 80, 100, new Color(245, 158, 11)); // Slot 7: Amber
        drawVehicle(g, 280, 230, 80, 100, new Color(75, 85, 99));   // Slot 12: Graphite Gray

        g.dispose();
        return img;
    }

    /**
     * Draws a realistic top-down vehicle graphic with windows, roof, and tires inside the slot ROI.
     */
    private static void drawVehicle(Graphics2D g, int sx, int sy, int sw, int sh, Color bodyColor) {
        int carW = 54;
        int carH = 82;
        int cx = sx + (sw - carW) / 2;
        int cy = sy + (sh - carH) / 2;

        // Shadow under the vehicle
        g.setColor(new Color(15, 23, 42, 160));
        g.fillRoundRect(cx + 2, cy + 4, carW, carH, 14, 14);

        // Tires
        g.setColor(new Color(20, 20, 25));
        g.fillRect(cx - 3, cy + 12, 5, 14); // Front-left
        g.fillRect(cx + carW - 2, cy + 12, 5, 14); // Front-right
        g.fillRect(cx - 3, cy + carH - 26, 5, 14); // Rear-left
        g.fillRect(cx + carW - 2, cy + carH - 26, 5, 14); // Rear-right

        // Car main body
        g.setColor(bodyColor);
        g.fill(new RoundRectangle2D.Double(cx, cy, carW, carH, 16, 16));

        // Body outline
        g.setColor(bodyColor.darker());
        g.setStroke(new BasicStroke(1.5f));
        g.draw(new RoundRectangle2D.Double(cx, cy, carW, carH, 16, 16));

        // Front Windshield (curved dark glass)
        g.setColor(new Color(30, 41, 59));
        g.fillRoundRect(cx + 6, cy + 18, carW - 12, 16, 8, 8);

        // Roof panel
        g.setColor(bodyColor.brighter());
        g.fillRoundRect(cx + 7, cy + 34, carW - 14, 26, 6, 6);

        // Rear Windshield
        g.setColor(new Color(30, 41, 59));
        g.fillRoundRect(cx + 7, cy + 62, carW - 14, 12, 6, 6);

        // Headlights (warm white/yellow)
        g.setColor(new Color(254, 240, 138));
        g.fillOval(cx + 4, cy + 2, 8, 5);
        g.fillOval(cx + carW - 12, cy + 2, 8, 5);

        // Taillights (red)
        g.setColor(new Color(239, 68, 68));
        g.fillOval(cx + 4, cy + carH - 5, 8, 4);
        g.fillOval(cx + carW - 12, cy + carH - 5, 8, 4);

        // Side mirrors
        g.setColor(bodyColor);
        g.fillOval(cx - 4, cy + 24, 5, 8);
        g.fillOval(cx + carW - 1, cy + 24, 5, 8);
    }

    /**
     * Generates and writes the sample parking image to disk if not already present.
     *
     * @param targetPath file path to save image
     * @return File object pointing to the image
     */
    public static File ensureSampleImageExists(String targetPath) throws IOException {
        File file = new File(targetPath);
        if (!file.exists()) {
            File parent = file.getParentFile();
            if (parent != null && !parent.exists()) {
                parent.mkdirs();
            }
            BufferedImage sample = generateDemoParkingLot();
            ImageIO.write(sample, "png", file);
        }
        return file;
    }
}
