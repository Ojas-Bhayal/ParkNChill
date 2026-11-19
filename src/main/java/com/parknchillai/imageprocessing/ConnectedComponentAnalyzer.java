package com.parknchillai.imageprocessing;

import java.awt.image.BufferedImage;
import java.util.ArrayDeque;
import java.util.Queue;

public final class ConnectedComponentAnalyzer {

    private static final int[] DX = {-1,  0,  1, -1, 1, -1, 0, 1};
    private static final int[] DY = {-1, -1, -1,  0, 0,  1, 1, 1};

    private ConnectedComponentAnalyzer() { /* utility class */ }

    public static final class ComponentSummary {
        private final int largestBlobPixels;
        private final double largestBlobRatio;
        private final int significantBlobCount;
        private final BufferedImage largestBlobMask;

        public ComponentSummary(int largestBlobPixels,
                                double largestBlobRatio,
                                int significantBlobCount,
                                BufferedImage largestBlobMask) {
            this.largestBlobPixels = largestBlobPixels;
            this.largestBlobRatio = largestBlobRatio;
            this.significantBlobCount = significantBlobCount;
            this.largestBlobMask = largestBlobMask;
        }

        public int getLargestBlobPixels()         { return largestBlobPixels; }
        public double getLargestBlobRatio()       { return largestBlobRatio; }
        public int getSignificantBlobCount()      { return significantBlobCount; }
        public BufferedImage getLargestBlobMask() { return largestBlobMask; }
    }

    public static double computeLargestBlobRatio(BufferedImage binaryInput) {
        return analyze(binaryInput).getLargestBlobRatio();
    }

    public static ComponentSummary analyze(BufferedImage binaryInput) {
        if (binaryInput == null) {
            throw new IllegalArgumentException("Binary input image must not be null.");
        }
        BufferedImage gray = GrayscaleConverter.ensureGrayscale(binaryInput);
        int w = gray.getWidth();
        int h = gray.getHeight();
        int totalPixels = w * h;

        int[][] labels = new int[h][w];
        int currentLabel = 0;
        int bestLabel = 0;
        int maxPixels = 0;
        int significantCount = 0;
        int minSignificantPixels = Math.max(4, totalPixels / 100);

        Queue<int[]> queue = new ArrayDeque<>();

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (labels[y][x] == 0 && gray.getRaster().getSample(x, y, 0) > 127) {
                    currentLabel++;
                    int compSize = 0;
                    labels[y][x] = currentLabel;
                    queue.add(new int[]{x, y});

                    while (!queue.isEmpty()) {
                        int[] pt = queue.poll();
                        int cx = pt[0];
                        int cy = pt[1];
                        compSize++;

                        for (int d = 0; d < 8; d++) {
                            int nx = cx + DX[d];
                            int ny = cy + DY[d];
                            if (nx >= 0 && nx < w && ny >= 0 && ny < h
                                    && labels[ny][nx] == 0
                                    && gray.getRaster().getSample(nx, ny, 0) > 127) {
                                labels[ny][nx] = currentLabel;
                                queue.add(new int[]{nx, ny});
                            }
                        }
                    }

                    if (compSize >= minSignificantPixels) {
                        significantCount++;
                    }
                    if (compSize > maxPixels) {
                        maxPixels = compSize;
                        bestLabel = currentLabel;
                    }
                }
            }
        }

        BufferedImage mask = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        if (bestLabel > 0) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    if (labels[y][x] == bestLabel) {
                        mask.getRaster().setSample(x, y, 0, 255);
                    }
                }
            }
        }

        double ratio = totalPixels > 0 ? (double) maxPixels / totalPixels : 0.0;
        return new ComponentSummary(maxPixels, ratio, significantCount, mask);
    }
}
