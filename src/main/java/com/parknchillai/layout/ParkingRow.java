package com.parknchillai.layout;

public final class ParkingRow {

    private final int rowIndex;
    private final int y;
    private final int height;
    private final int xStart;
    private final int width;

    public ParkingRow(int rowIndex, int y, int height, int xStart, int width) {
        if (rowIndex < 0) throw new IllegalArgumentException("rowIndex must be >= 0.");
        if (y < 0 || xStart < 0) throw new IllegalArgumentException("Row coordinates must be >= 0.");
        if (height <= 0 || width <= 0) throw new IllegalArgumentException("Row dimensions must be > 0.");
        this.rowIndex = rowIndex;
        this.y = y;
        this.height = height;
        this.xStart = xStart;
        this.width = width;
    }

    public int getRowIndex() { return rowIndex; }
    public int getY()        { return y; }
    public int getHeight()   { return height; }
    public int getXStart()   { return xStart; }
    public int getWidth()    { return width; }
    public int getBottom()   { return y + height; }
    public int getRight()    { return xStart + width; }

    @Override
    public String toString() {
        return String.format("ParkingRow[idx=%d, y=%d, h=%d, x=%d, w=%d]",
            rowIndex, y, height, xStart, width);
    }
}
