package com.parknchillai.detection;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Responsibility 1 of the Vehicle Detection Pipeline:
 * <ul>
 *   <li>Model-specific letterbox resizing to model input dimensions (e.g. {@code 640x640})
 *       with neutral padding {@code 114}</li>
 *   <li>Generic models: planar {@code CHW} RGB normalization to {@code [0.0f, 1.0f]};
 *       official YOLOX: planar {@code CHW} BGR values in {@code [0,255]}</li>
 *   <li>Exact inverse coordinate mapping from model letterbox coordinates back to
 *       original image dimensions {@code (originalWidth, originalHeight)}</li>
 *   <li>Validated overlapping tile generation and tile-to-global coordinate transformation
 *       for overhead parking-lot images (such as {@code 739x415} px) where individual vehicles
 *       occupy small pixel regions (Phase 3)</li>
 * </ul>
 */
public final class DetectionPreprocessor {

    public static final int DEFAULT_INPUT_SIZE = 640;
    public static final double DEFAULT_TILE_OVERLAP_FRACTION = 0.25;
    private static final Color LETTERBOX_PAD_COLOR = new Color(114, 114, 114);

    private DetectionPreprocessor() { /* utility class */ }

    public record LetterboxResult(
            BufferedImage letterboxedImage,
            float[] chwNormalizedTensor,
            int inputWidth,
            int inputHeight,
            int originalWidth,
            int originalHeight,
            double scale,
            int padX,
            int padY
    ) {}

    /**
     * Represents a sub-image tile extracted from a full parking-lot image for
     * high-resolution small-vehicle inference (Phase 3).
     */
    public record ImageTile(
            int tileIndex,
            int offsetX,
            int offsetY,
            int width,
            int height,
            int fullImageWidth,
            int fullImageHeight,
            BufferedImage tileImage,
            LetterboxResult letterbox
    ) {}

    public static LetterboxResult preprocess(BufferedImage originalImage) {
        return preprocess(originalImage, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE);
    }

    /**
     * Preprocessing used by the bundled official YOLOX-s model. YOLOX's standard
     * inference path expects OpenCV-style BGR channels in the 0..255 range, CHW,
     * with padding at the bottom/right (content anchored at top-left). The generic
     * preprocess() method below intentionally remains RGB-normalized/centered for
     * models using that convention.
     */
    public static LetterboxResult preprocessYoloX(BufferedImage originalImage, int targetWidth, int targetHeight) {
        if (originalImage == null) throw new IllegalArgumentException("Input image must not be null.");
        if (targetWidth < 32 || targetHeight < 32) {
            throw new IllegalArgumentException("Target dimensions must be >= 32x32; got " + targetWidth + "x" + targetHeight);
        }
        int origW = originalImage.getWidth();
        int origH = originalImage.getHeight();
        if (origW <= 0 || origH <= 0) throw new IllegalArgumentException("Original image dimensions must be positive.");

        double scale = Math.min((double) targetWidth / origW, (double) targetHeight / origH);
        // Match YOLOX's reference preprocessing: integer-truncated resized dimensions and top-left padding.
        int resizedW = Math.max(1, Math.min(targetWidth, (int) (origW * scale)));
        int resizedH = Math.max(1, Math.min(targetHeight, (int) (origH * scale)));
        BufferedImage letterboxed = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = letterboxed.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setColor(LETTERBOX_PAD_COLOR);
        g.fillRect(0, 0, targetWidth, targetHeight);
        g.drawImage(originalImage, 0, 0, resizedW, resizedH, null);
        g.dispose();

        int planeSize = targetWidth * targetHeight;
        float[] tensor = new float[3 * planeSize];
        for (int y = 0; y < targetHeight; y++) {
            int rowOffset = y * targetWidth;
            for (int x = 0; x < targetWidth; x++) {
                int rgb = letterboxed.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int gr = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                int idx = rowOffset + x;
                // YOLOX receives BGR in NCHW float32 without dividing by 255.
                tensor[idx] = b;
                tensor[planeSize + idx] = gr;
                tensor[2 * planeSize + idx] = r;
            }
        }
        return new LetterboxResult(letterboxed, tensor, targetWidth, targetHeight,
            origW, origH, scale, 0, 0);
    }

    public static LetterboxResult preprocess(BufferedImage originalImage, int targetWidth, int targetHeight) {
        if (originalImage == null) {
            throw new IllegalArgumentException("Input image must not be null.");
        }
        if (targetWidth < 32 || targetHeight < 32) {
            throw new IllegalArgumentException("Target dimensions must be >= 32x32; got " + targetWidth + "x" + targetHeight);
        }
        int origW = originalImage.getWidth();
        int origH = originalImage.getHeight();
        if (origW <= 0 || origH <= 0) {
            throw new IllegalArgumentException("Original image dimensions must be positive.");
        }

        double scale = Math.min((double) targetWidth / origW, (double) targetHeight / origH);
        int resizedW = Math.max(1, (int) Math.round(origW * scale));
        int resizedH = Math.max(1, (int) Math.round(origH * scale));
        int padX = (targetWidth - resizedW) / 2;
        int padY = (targetHeight - resizedH) / 2;

        BufferedImage letterboxed = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = letterboxed.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setColor(LETTERBOX_PAD_COLOR);
        g.fillRect(0, 0, targetWidth, targetHeight);
        g.drawImage(originalImage, padX, padY, resizedW, resizedH, null);
        g.dispose();

        int planeSize = targetWidth * targetHeight;
        float[] tensor = new float[3 * planeSize];
        for (int y = 0; y < targetHeight; y++) {
            int rowOffset = y * targetWidth;
            for (int x = 0; x < targetWidth; x++) {
                int rgb = letterboxed.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int gr = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                int idx = rowOffset + x;
                tensor[idx] = r / 255.0f;
                tensor[planeSize + idx] = gr / 255.0f;
                tensor[2 * planeSize + idx] = b / 255.0f;
            }
        }

        return new LetterboxResult(
            letterboxed, tensor, targetWidth, targetHeight, origW, origH, scale, padX, padY
        );
    }

    /**
     * Generates an overlapping grid of sub-image tiles ({@code gridCols x gridRows}) with
     * {@code overlapFraction} (e.g. {@code 0.25} = 25% overlap) so small overhead vehicles
     * (such as in a {@code 739x415} 26-slot lot) are upscaled cleanly inside each {@code 640x640}
     * tile without blind model input-shape changes.
     */
    public static List<ImageTile> generateOverlappingTiles(BufferedImage fullImage,
                                                           int gridCols,
                                                           int gridRows,
                                                           double overlapFraction,
                                                           int targetWidth,
                                                           int targetHeight) {
        return generateOverlappingTilesInternal(fullImage, gridCols, gridRows, overlapFraction, targetWidth, targetHeight, false);
    }

    /** Generates tiles using the bundled YOLOX model's BGR/0..255/top-left input convention. */
    public static List<ImageTile> generateOverlappingTilesYoloX(BufferedImage fullImage,
                                                                int gridCols,
                                                                int gridRows,
                                                                double overlapFraction,
                                                                int targetWidth,
                                                                int targetHeight) {
        return generateOverlappingTilesInternal(fullImage, gridCols, gridRows, overlapFraction, targetWidth, targetHeight, true);
    }

    private static List<ImageTile> generateOverlappingTilesInternal(BufferedImage fullImage,
                                                                    int gridCols,
                                                                    int gridRows,
                                                                    double overlapFraction,
                                                                    int targetWidth,
                                                                    int targetHeight,
                                                                    boolean yoloXProfile) {
        if (fullImage == null) throw new IllegalArgumentException("Full image must not be null.");
        if (gridCols < 1 || gridRows < 1) {
            throw new IllegalArgumentException("gridCols and gridRows must be >= 1.");
        }
        if (!Double.isFinite(overlapFraction) || overlapFraction < 0.0 || overlapFraction >= 0.60) {
            throw new IllegalArgumentException("overlapFraction must be in [0.0, 0.60); got " + overlapFraction);
        }

        int fullW = fullImage.getWidth();
        int fullH = fullImage.getHeight();
        int baseTileW = (int) Math.ceil((double) fullW / gridCols);
        int baseTileH = (int) Math.ceil((double) fullH / gridRows);
        int padW = (int) Math.round(baseTileW * overlapFraction);
        int padH = (int) Math.round(baseTileH * overlapFraction);

        List<ImageTile> tiles = new ArrayList<>(gridCols * gridRows);
        int idx = 0;
        for (int r = 0; r < gridRows; r++) {
            int y0 = Math.max(0, r * baseTileH - (r > 0 ? padH : 0));
            int y1 = (r == gridRows - 1) ? fullH : Math.min(fullH, (r + 1) * baseTileH + padH);
            int th = Math.max(32, y1 - y0);
            if (y0 + th > fullH) y0 = Math.max(0, fullH - th);

            for (int c = 0; c < gridCols; c++) {
                int x0 = Math.max(0, c * baseTileW - (c > 0 ? padW : 0));
                int x1 = (c == gridCols - 1) ? fullW : Math.min(fullW, (c + 1) * baseTileW + padW);
                int tw = Math.max(32, x1 - x0);
                if (x0 + tw > fullW) x0 = Math.max(0, fullW - tw);

                BufferedImage sub = fullImage.getSubimage(x0, y0, tw, th);
                BufferedImage copy = new BufferedImage(tw, th, BufferedImage.TYPE_INT_RGB);
                Graphics2D g = copy.createGraphics();
                g.drawImage(sub, 0, 0, null);
                g.dispose();

                LetterboxResult lb = yoloXProfile
                    ? preprocessYoloX(copy, targetWidth, targetHeight)
                    : preprocess(copy, targetWidth, targetHeight);
                tiles.add(new ImageTile(idx++, x0, y0, tw, th, fullW, fullH, copy, lb));
            }
        }
        return List.copyOf(tiles);
    }

    /**
     * Maps a bounding box {@code (lx, ly, lw, lh)} from letterboxed model input coordinates
     * back to the original image dimensions {@code [0, originalWidth] x [0, originalHeight]}.
     *
     * @return 4-element array {@code [origX, origY, origW, origH]}
     */
    public static double[] mapBoxToOriginalImage(double lx,
                                                 double ly,
                                                 double lw,
                                                 double lh,
                                                 LetterboxResult meta) {
        if (meta == null) {
            throw new IllegalArgumentException("LetterboxResult metadata must not be null.");
        }
        double x0 = (lx - meta.padX()) / meta.scale();
        double y0 = (ly - meta.padY()) / meta.scale();
        double x1 = (lx + lw - meta.padX()) / meta.scale();
        double y1 = (ly + lh - meta.padY()) / meta.scale();

        double clampedX0 = Math.max(0.0, Math.min(meta.originalWidth(), x0));
        double clampedY0 = Math.max(0.0, Math.min(meta.originalHeight(), y0));
        double clampedX1 = Math.max(clampedX0, Math.min(meta.originalWidth(), x1));
        double clampedY1 = Math.max(clampedY0, Math.min(meta.originalHeight(), y1));

        return new double[]{
            clampedX0,
            clampedY0,
            clampedX1 - clampedX0,
            clampedY1 - clampedY0
        };
    }

    /**
     * Maps a bounding box {@code (lx, ly, lw, lh)} from a tile's letterbox coordinate space
     * back into the full source image coordinate space {@code [0, fullImageWidth] x [0, fullImageHeight]}.
     */
    public static double[] mapTileBoxToFullImage(double lx,
                                                 double ly,
                                                 double lw,
                                                 double lh,
                                                 ImageTile tile) {
        if (tile == null) {
            throw new IllegalArgumentException("ImageTile must not be null.");
        }
        double[] local = mapBoxToOriginalImage(lx, ly, lw, lh, tile.letterbox());
        double gx0 = Math.max(0.0, Math.min(tile.fullImageWidth(), tile.offsetX() + local[0]));
        double gy0 = Math.max(0.0, Math.min(tile.fullImageHeight(), tile.offsetY() + local[1]));
        double gx1 = Math.max(gx0, Math.min(tile.fullImageWidth(), tile.offsetX() + local[0] + local[2]));
        double gy1 = Math.max(gy0, Math.min(tile.fullImageHeight(), tile.offsetY() + local[1] + local[3]));
        return new double[]{gx0, gy0, gx1 - gx0, gy1 - gy0};
    }
}
