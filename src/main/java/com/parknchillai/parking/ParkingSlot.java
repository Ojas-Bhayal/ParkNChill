package com.parknchillai.parking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Represents a single parking space within the parking-area image, preserving both:
 * <ul>
 *   <li>Its axis-aligned bounding box {@code (x, y, width, height)} in active processed image coordinates</li>
 *   <li>Its exact quadrilateral or multi-point polygon contour {@code polygonVertices} and optional
 *       {@code rotationAngleDegrees} when imported from angled/trapezoidal PKLot XML contours or
 *       rotated rectangles (ParkNChill v2.3 Requirement 5).</li>
 * </ul>
 */
public final class ParkingSlot {

    /**
     * Indicates how this parking-space ROI geometry was defined or proposed.
     */
    public enum GeometryOrigin {
        CSV_CONFIG("CSV Configuration"),
        PKLOT_POLYGON("PKLot Polygon Contour"),
        PKLOT_ROTATED_RECT("PKLot Rotated Rectangle"),
        AUTO_DETECTED("Auto-Detected (IVP)"),
        MANUAL_DRAWN("Manual Box"),
        ROW_GRID("Manual Row Grid");

        private final String displayName;
        GeometryOrigin(String displayName) { this.displayName = displayName; }
        public String getDisplayName()     { return displayName; }
        @Override public String toString() { return displayName; }
    }

    private final int id;
    private final int x;
    private final int y;
    private final int width;
    private final int height;
    private final GeometryOrigin origin;
    private final List<double[]> polygonVertices;
    private final boolean customPolygon;
    private final Double rotationAngleDegrees;

    /** Classification result assigned after prediction; null means not yet classified. */
    private SlotStatus status;

    public ParkingSlot(int id, int x, int y, int width, int height) {
        this(id, x, y, width, height, GeometryOrigin.CSV_CONFIG, null, null);
    }

    public ParkingSlot(int id, int x, int y, int width, int height, GeometryOrigin origin) {
        this(id, x, y, width, height, origin, null, null);
    }

    public ParkingSlot(int id,
                       int x,
                       int y,
                       int width,
                       int height,
                       GeometryOrigin origin,
                       List<double[]> polygonVertices,
                       Double rotationAngleDegrees) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException(
                "Slot " + id + ": width and height must be positive (got " + width + "x" + height + ").");
        }
        if (x < 0 || y < 0) {
            throw new IllegalArgumentException(
                "Slot " + id + ": x and y must be non-negative (got " + x + ", " + y + ").");
        }
        this.id = id;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.origin = origin != null ? origin : GeometryOrigin.CSV_CONFIG;
        this.rotationAngleDegrees = rotationAngleDegrees;

        if (polygonVertices != null && polygonVertices.size() >= 3) {
            List<double[]> copied = new ArrayList<>(polygonVertices.size());
            for (double[] pt : polygonVertices) {
                if (pt == null || pt.length < 2 || !Double.isFinite(pt[0]) || !Double.isFinite(pt[1])) {
                    throw new IllegalArgumentException("Slot " + id + ": polygon vertex must have finite [x, y].");
                }
                copied.add(new double[]{pt[0], pt[1]});
            }
            this.polygonVertices = Collections.unmodifiableList(copied);
            this.customPolygon = !isAxisAlignedRectangle(copied, x, y, width, height);
        } else {
            this.polygonVertices = List.of(
                new double[]{x, y},
                new double[]{x + width, y},
                new double[]{x + width, y + height},
                new double[]{x, y + height}
            );
            this.customPolygon = false;
        }
    }

    /**
     * Constructs a {@link ParkingSlot} directly from a multi-point PKLot contour or rotated quadrilateral,
     * preserving the exact polygon vertices while deriving the enclosing axis-aligned bounding box.
     */
    public static ParkingSlot fromPolygon(int id,
                                          List<double[]> vertices,
                                          GeometryOrigin origin,
                                          Double rotationAngleDegrees) {
        if (vertices == null || vertices.size() < 3) {
            throw new IllegalArgumentException("Polygon slot requires at least 3 vertices.");
        }
        double minX = Double.POSITIVE_INFINITY;
        double minY = Double.POSITIVE_INFINITY;
        double maxX = Double.NEGATIVE_INFINITY;
        double maxY = Double.NEGATIVE_INFINITY;
        for (double[] pt : vertices) {
            if (pt == null || pt.length < 2 || !Double.isFinite(pt[0]) || !Double.isFinite(pt[1])) {
                throw new IllegalArgumentException("Invalid polygon vertex.");
            }
            minX = Math.min(minX, pt[0]);
            minY = Math.min(minY, pt[1]);
            maxX = Math.max(maxX, pt[0]);
            maxY = Math.max(maxY, pt[1]);
        }
        int bx = Math.max(0, (int) Math.floor(minX));
        int by = Math.max(0, (int) Math.floor(minY));
        int bw = Math.max(1, (int) Math.ceil(maxX) - bx);
        int bh = Math.max(1, (int) Math.ceil(maxY) - by);
        return new ParkingSlot(id, bx, by, bw, bh,
            origin != null ? origin : GeometryOrigin.PKLOT_POLYGON,
            vertices, rotationAngleDegrees);
    }

    private static boolean isAxisAlignedRectangle(List<double[]> pts, int rx, int ry, int rw, int rh) {
        if (pts.size() != 4) return false;
        double tol = 1e-3;
        for (double[] p : pts) {
            boolean xMatch = Math.abs(p[0] - rx) <= tol || Math.abs(p[0] - (rx + rw)) <= tol;
            boolean yMatch = Math.abs(p[1] - ry) <= tol || Math.abs(p[1] - (ry + rh)) <= tol;
            if (!xMatch || !yMatch) return false;
        }
        return true;
    }

    public int getId()                  { return id; }
    public int getX()                   { return x; }
    public int getY()                   { return y; }
    public int getWidth()               { return width; }
    public int getHeight()              { return height; }
    public GeometryOrigin getOrigin()   { return origin; }

    /**
     * Returns the immutable list of {@code [x, y]} polygon vertices in image coordinates.
     */
    public List<double[]> getPolygonVertices() {
        List<double[]> copy = new ArrayList<>(polygonVertices.size());
        for (double[] p : polygonVertices) {
            copy.add(new double[]{p[0], p[1]});
        }
        return Collections.unmodifiableList(copy);
    }

    /**
     * Returns {@code true} if this slot has a non-axis-aligned quadrilateral or polygon contour.
     */
    public boolean hasCustomPolygon() {
        return customPolygon;
    }

    public Double getRotationAngleDegrees() {
        return rotationAngleDegrees;
    }

    public SlotStatus getStatus()            { return status; }
    public void setStatus(SlotStatus status) { this.status = status; }

    public boolean isClassified() { return status != null; }

    /**
     * Computes the exact geometric area of this parking space using the Shoelace formula
     * over {@link #polygonVertices} (or {@code width * height} for axis-aligned rectangles).
     */
    public double computeExactArea() {
        if (!customPolygon) {
            return (double) width * height;
        }
        double area = computePolygonArea(polygonVertices);
        return area > 1.0 ? area : (double) width * height;
    }

    /**
     * Checks whether point {@code (px, py)} lies inside or on the boundary of this parking space's
     * polygon contour using ray-casting point-in-polygon testing.
     */
    public boolean containsPoint(double px, double py) {
        if (px < x || px > x + width || py < y || py > y + height) {
            return false;
        }
        if (!customPolygon) {
            return true;
        }
        int n = polygonVertices.size();
        boolean inside = false;
        for (int i = 0, j = n - 1; i < n; j = i++) {
            double xi = polygonVertices.get(i)[0], yi = polygonVertices.get(i)[1];
            double xj = polygonVertices.get(j)[0], yj = polygonVertices.get(j)[1];
            if (isPointOnSegment(px, py, xi, yi, xj, yj)) {
                return true;
            }
            boolean intersect = ((yi > py) != (yj > py))
                && (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi);
            if (intersect) {
                inside = !inside;
            }
        }
        return inside;
    }

    private static boolean isPointOnSegment(double px, double py, double x1, double y1, double x2, double y2) {
        double cross = Math.abs((px - x1) * (y2 - y1) - (py - y1) * (x2 - x1));
        if (cross > 1e-4) return false;
        double minX = Math.min(x1, x2) - 1e-4, maxX = Math.max(x1, x2) + 1e-4;
        double minY = Math.min(y1, y2) - 1e-4, maxY = Math.max(y1, y2) + 1e-4;
        return px >= minX && px <= maxX && py >= minY && py <= maxY;
    }

    /**
     * Computes the exact intersection area between this slot's polygon contour and an axis-aligned
     * rectangle {@code [rx, ry, rx + rw, ry + rh]} using Sutherland-Hodgman half-plane polygon clipping.
     */
    public double computeIntersectionAreaWithRect(double rx, double ry, double rw, double rh) {
        if (rw <= 0.0 || rh <= 0.0) return 0.0;
        double ix0 = Math.max(this.x, rx);
        double iy0 = Math.max(this.y, ry);
        double ix1 = Math.min(this.x + this.width, rx + rw);
        double iy1 = Math.min(this.y + this.height, ry + rh);
        if (ix1 <= ix0 || iy1 <= iy0) {
            return 0.0;
        }
        if (!customPolygon) {
            return (ix1 - ix0) * (iy1 - iy0);
        }

        List<double[]> poly = new ArrayList<>(polygonVertices);
        poly = clipPolygonAgainstHalfPlane(poly, 1, 0, rx);          // x >= rx
        poly = clipPolygonAgainstHalfPlane(poly, -1, 0, -(rx + rw)); // x <= rx + rw
        poly = clipPolygonAgainstHalfPlane(poly, 0, 1, ry);          // y >= ry
        poly = clipPolygonAgainstHalfPlane(poly, 0, -1, -(ry + rh)); // y <= ry + rh

        if (poly.size() < 3) {
            return 0.0;
        }
        return computePolygonArea(poly);
    }

    /**
     * Clips {@code poly} against half-plane {@code nx * x + ny * y >= c}.
     */
    private static List<double[]> clipPolygonAgainstHalfPlane(List<double[]> poly, double nx, double ny, double c) {
        if (poly.isEmpty()) return poly;
        List<double[]> out = new ArrayList<>();
        int n = poly.size();
        for (int i = 0; i < n; i++) {
            double[] prev = poly.get((i + n - 1) % n);
            double[] curr = poly.get(i);
            double prevVal = nx * prev[0] + ny * prev[1] - c;
            double currVal = nx * curr[0] + ny * curr[1] - c;
            boolean prevIn = prevVal >= -1e-9;
            boolean currIn = currVal >= -1e-9;

            if (prevIn && currIn) {
                out.add(curr);
            } else if (prevIn && !currIn) {
                out.add(intersectSegment(prev, curr, prevVal, currVal));
            } else if (!prevIn && currIn) {
                out.add(intersectSegment(prev, curr, prevVal, currVal));
                out.add(curr);
            }
        }
        return out;
    }

    private static double[] intersectSegment(double[] a, double[] b, double valA, double valB) {
        double denom = valA - valB;
        double t = Math.abs(denom) > 1e-12 ? (valA / denom) : 0.5;
        t = Math.max(0.0, Math.min(1.0, t));
        return new double[]{
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1])
        };
    }

    public static double computePolygonArea(List<double[]> vertices) {
        if (vertices == null || vertices.size() < 3) return 0.0;
        double sum = 0.0;
        int n = vertices.size();
        for (int i = 0; i < n; i++) {
            double[] p1 = vertices.get(i);
            double[] p2 = vertices.get((i + 1) % n);
            sum += (p1[0] * p2[1]) - (p2[0] * p1[1]);
        }
        return 0.5 * Math.abs(sum);
    }

    /**
     * Serializes polygon vertices as {@code "x1,y1;x2,y2;x3,y3;x4,y4"}.
     */
    public String formatPolygonCoordinates() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < polygonVertices.size(); i++) {
            if (i > 0) sb.append(';');
            double[] pt = polygonVertices.get(i);
            sb.append(String.format(Locale.ROOT, "%.1f,%.1f", pt[0], pt[1]));
        }
        return sb.toString();
    }

    public ParkingSlot withOrigin(GeometryOrigin newOrigin) {
        ParkingSlot copy = new ParkingSlot(id, x, y, width, height, newOrigin, polygonVertices, rotationAngleDegrees);
        copy.setStatus(this.status);
        return copy;
    }

    public ParkingSlot scaleToImage(double scaleX, double scaleY, int imageWidth, int imageHeight) {
        if (!Double.isFinite(scaleX) || !Double.isFinite(scaleY) || scaleX <= 0 || scaleY <= 0) {
            throw new IllegalArgumentException("Scale factors must be positive finite numbers.");
        }
        if (imageWidth < 1 || imageHeight < 1) {
            throw new IllegalArgumentException("Target image dimensions must be positive.");
        }
        int sx = Math.max(0, Math.min(imageWidth - 1, (int) Math.round(x * scaleX)));
        int sy = Math.max(0, Math.min(imageHeight - 1, (int) Math.round(y * scaleY)));
        int sw = Math.max(1, Math.min(imageWidth - sx, (int) Math.round(width * scaleX)));
        int sh = Math.max(1, Math.min(imageHeight - sy, (int) Math.round(height * scaleY)));

        List<double[]> scaledPoly = null;
        if (customPolygon) {
            scaledPoly = new ArrayList<>(polygonVertices.size());
            for (double[] pt : polygonVertices) {
                double px = Math.max(0.0, Math.min(imageWidth, pt[0] * scaleX));
                double py = Math.max(0.0, Math.min(imageHeight, pt[1] * scaleY));
                scaledPoly.add(new double[]{px, py});
            }
        }
        return new ParkingSlot(id, sx, sy, sw, sh, origin, scaledPoly, rotationAngleDegrees);
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT,
            "ParkingSlot[id=%d, x=%d, y=%d, w=%d, h=%d, customPolygon=%s, origin=%s, status=%s]",
            id, x, y, width, height, customPolygon, origin.name(), status);
    }
}
