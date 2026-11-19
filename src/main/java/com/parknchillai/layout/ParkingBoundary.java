package com.parknchillai.layout;

public final class ParkingBoundary {

    private final int x;
    private final int y;
    private final int width;
    private final int height;

    public ParkingBoundary(int x, int y, int width, int height) {
        if (x < 0 || y < 0) {
            throw new IllegalArgumentException("Boundary coordinates must be non-negative.");
        }
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Boundary dimensions must be positive.");
        }
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public int getX()      { return x; }
    public int getY()      { return y; }
    public int getWidth()  { return width; }
    public int getHeight() { return height; }
    public int getRight()  { return x + width; }
    public int getBottom() { return y + height; }

    @Override
    public String toString() {
        return String.format("ParkingBoundary[x=%d, y=%d, w=%d, h=%d]", x, y, width, height);
    }
}
