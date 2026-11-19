package com.parknchillai.imageprocessing;

import java.awt.image.BufferedImage;

/**
 * Performs 4-point projective homography transformation (perspective rectification)
 * to warp an angled parking-lot view into a flat top-down bird's-eye grid.
 */
public final class PerspectiveTransformer {

    private PerspectiveTransformer() { /* utility class */ }

    public static BufferedImage warpPerspective(BufferedImage src, double[][] srcQuad, int dstWidth, int dstHeight) {
        if (src == null) {
            throw new IllegalArgumentException("Source image must not be null.");
        }
        validateQuad(srcQuad);
        if (dstWidth < 1 || dstHeight < 1) {
            throw new IllegalArgumentException("Destination dimensions must be positive.");
        }

        double maxU = Math.max(1.0, dstWidth - 1.0);
        double maxV = Math.max(1.0, dstHeight - 1.0);
        double[][] dstQuad = new double[][]{
            {0.0,  0.0},
            {maxU, 0.0},
            {maxU, maxV},
            {0.0,  maxV}
        };

        double[] h = computeHomography(dstQuad, srcQuad);

        boolean isGray = (src.getType() == BufferedImage.TYPE_BYTE_GRAY);
        BufferedImage out = new BufferedImage(
            dstWidth,
            dstHeight,
            isGray ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_INT_RGB
        );

        int srcW = src.getWidth();
        int srcH = src.getHeight();

        for (int v = 0; v < dstHeight; v++) {
            for (int u = 0; u < dstWidth; u++) {
                double denom = h[6] * u + h[7] * v + 1.0;
                if (Math.abs(denom) < 1e-12) continue;
                double sx = (h[0] * u + h[1] * v + h[2]) / denom;
                double sy = (h[3] * u + h[4] * v + h[5]) / denom;

                if (isGray) {
                    int grayVal = sampleBilinearGray(src, sx, sy, srcW, srcH);
                    out.getRaster().setSample(u, v, 0, grayVal);
                } else {
                    int rgbVal = sampleBilinearRgb(src, sx, sy, srcW, srcH);
                    out.setRGB(u, v, rgbVal);
                }
            }
        }
        return out;
    }

    static double[] computeHomography(double[][] fromQuad, double[][] toQuad) {
        validateQuad(fromQuad);
        validateQuad(toQuad);

        double[][] a = new double[8][8];
        double[] b = new double[8];

        for (int i = 0; i < 4; i++) {
            double u = fromQuad[i][0];
            double v = fromQuad[i][1];
            double x = toQuad[i][0];
            double y = toQuad[i][1];

            int r1 = 2 * i;
            int r2 = r1 + 1;

            a[r1][0] = u;
            a[r1][1] = v;
            a[r1][2] = 1.0;
            a[r1][6] = -u * x;
            a[r1][7] = -v * x;
            b[r1]    = x;

            a[r2][3] = u;
            a[r2][4] = v;
            a[r2][5] = 1.0;
            a[r2][6] = -u * y;
            a[r2][7] = -v * y;
            b[r2]    = y;
        }

        return solveLinearSystem8x8(a, b);
    }

    private static double[] solveLinearSystem8x8(double[][] a, double[] b) {
        int n = 8;
        for (int col = 0; col < n; col++) {
            int pivot = col;
            double maxAbs = Math.abs(a[col][col]);
            for (int row = col + 1; row < n; row++) {
                double val = Math.abs(a[row][col]);
                if (val > maxAbs) {
                    maxAbs = val;
                    pivot = row;
                }
            }
            if (maxAbs < 1e-11) {
                throw new IllegalArgumentException("Degenerate quadrilateral: corners are collinear or coincident.");
            }
            if (pivot != col) {
                double[] tmpRow = a[col];
                a[col] = a[pivot];
                a[pivot] = tmpRow;
                double tmpB = b[col];
                b[col] = b[pivot];
                b[pivot] = tmpB;
            }

            double diag = a[col][col];
            for (int row = col + 1; row < n; row++) {
                double factor = a[row][col] / diag;
                b[row] -= factor * b[col];
                for (int k = col; k < n; k++) {
                    a[row][k] -= factor * a[col][k];
                }
            }
        }

        double[] x = new double[n];
        for (int row = n - 1; row >= 0; row--) {
            double sum = b[row];
            for (int col = row + 1; col < n; col++) {
                sum -= a[row][col] * x[col];
            }
            x[row] = sum / a[row][row];
        }
        return x;
    }

    private static int sampleBilinearGray(BufferedImage src, double sx, double sy, int w, int h) {
        double cx = Math.max(0.0, Math.min(w - 1.0, sx));
        double cy = Math.max(0.0, Math.min(h - 1.0, sy));
        int x0 = (int) Math.floor(cx);
        int y0 = (int) Math.floor(cy);
        int x1 = Math.min(w - 1, x0 + 1);
        int y1 = Math.min(h - 1, y0 + 1);
        double dx = cx - x0;
        double dy = cy - y0;

        double i00 = src.getRaster().getSample(x0, y0, 0);
        double i10 = src.getRaster().getSample(x1, y0, 0);
        double i01 = src.getRaster().getSample(x0, y1, 0);
        double i11 = src.getRaster().getSample(x1, y1, 0);

        double val = (1 - dx) * (1 - dy) * i00
                   + dx * (1 - dy) * i10
                   + (1 - dx) * dy * i01
                   + dx * dy * i11;
        return Math.max(0, Math.min(255, (int) Math.round(val)));
    }

    private static int sampleBilinearRgb(BufferedImage src, double sx, double sy, int w, int h) {
        double cx = Math.max(0.0, Math.min(w - 1.0, sx));
        double cy = Math.max(0.0, Math.min(h - 1.0, sy));
        int x0 = (int) Math.floor(cx);
        int y0 = (int) Math.floor(cy);
        int x1 = Math.min(w - 1, x0 + 1);
        int y1 = Math.min(h - 1, y0 + 1);
        double dx = cx - x0;
        double dy = cy - y0;

        int c00 = src.getRGB(x0, y0);
        int c10 = src.getRGB(x1, y0);
        int c01 = src.getRGB(x0, y1);
        int c11 = src.getRGB(x1, y1);

        int r = interpolateChannel((c00 >> 16) & 0xFF, (c10 >> 16) & 0xFF, (c01 >> 16) & 0xFF, (c11 >> 16) & 0xFF, dx, dy);
        int g = interpolateChannel((c00 >> 8)  & 0xFF, (c10 >> 8)  & 0xFF, (c01 >> 8)  & 0xFF, (c11 >> 8)  & 0xFF, dx, dy);
        int b = interpolateChannel(c00 & 0xFF,         c10 & 0xFF,         c01 & 0xFF,         c11 & 0xFF,         dx, dy);

        return (0xFF << 24) | (r << 16) | (g << 8) | b;
    }

    private static int interpolateChannel(int v00, int v10, int v01, int v11, double dx, double dy) {
        double val = (1 - dx) * (1 - dy) * v00
                   + dx * (1 - dy) * v10
                   + (1 - dx) * dy * v01
                   + dx * dy * v11;
        return Math.max(0, Math.min(255, (int) Math.round(val)));
    }

    private static void validateQuad(double[][] quad) {
        if (quad == null || quad.length != 4) {
            throw new IllegalArgumentException("Quadrilateral must contain exactly 4 corner points.");
        }
        for (int i = 0; i < 4; i++) {
            if (quad[i] == null || quad[i].length != 2
                    || !Double.isFinite(quad[i][0]) || !Double.isFinite(quad[i][1])) {
                throw new IllegalArgumentException("Corner " + i + " must be a finite [x, y] pair.");
            }
        }
    }
}
