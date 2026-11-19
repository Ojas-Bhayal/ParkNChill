package com.parknchillai.layout;

import com.parknchillai.parking.ParkingSlot;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates candidate {@link ParkingSlot} boundaries from detected rows and
 * suppresses duplicate/heavily overlapping slot proposals (ParkNChill v2.3 Requirement 4).
 */
public final class SlotGenerator {

    public static final double DUPLICATE_PROPOSAL_MAX_IOU = 0.55;

    private SlotGenerator() { /* utility class */ }

    public static List<ParkingSlot> generateSlots(BufferedImage image,
                                                  ParkingBoundary boundary,
                                                  List<ParkingRow> rows) {
        if (image == null) throw new IllegalArgumentException("Image must not be null.");
        if (boundary == null) throw new IllegalArgumentException("Boundary must not be null.");
        if (rows == null || rows.isEmpty()) {
            return List.of();
        }

        int imgW = image.getWidth();
        int imgH = image.getHeight();
        List<ParkingSlot> rawProposals = new ArrayList<>();
        int nextId = 1;

        for (ParkingRow row : rows) {
            List<int[]> bayIntervals = SpacingEstimator.detectBayIntervals(image, row);
            int ry = Math.max(0, Math.min(imgH - 1, row.getY()));
            int rh = Math.min(row.getHeight(), imgH - ry);
            if (rh <= 0) continue;

            for (int[] bay : bayIntervals) {
                int bx = Math.max(0, Math.min(imgW - 1, bay[0]));
                int bw = Math.min(bay[1], imgW - bx);
                if (bw >= 8 && rh >= 8) {
                    rawProposals.add(new ParkingSlot(nextId++, bx, ry, bw, rh, ParkingSlot.GeometryOrigin.AUTO_DETECTED));
                }
            }
        }
        return suppressOverlappingProposals(rawProposals, DUPLICATE_PROPOSAL_MAX_IOU);
    }

    /**
     * Suppresses duplicate or heavily overlapping parking-space proposals ({@code IoU >= maxIou}),
     * re-indexing accepted slots sequentially from {@code 1..N}.
     */
    public static List<ParkingSlot> suppressOverlappingProposals(List<ParkingSlot> proposals, double maxIou) {
        if (proposals == null || proposals.isEmpty()) {
            return List.of();
        }
        List<ParkingSlot> kept = new ArrayList<>();
        int nextId = 1;
        for (ParkingSlot candidate : proposals) {
            if (candidate == null || candidate.getWidth() <= 0 || candidate.getHeight() <= 0) continue;
            boolean duplicate = false;
            for (ParkingSlot existing : kept) {
                if (computeSlotIoU(existing, candidate) >= maxIou) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                kept.add(new ParkingSlot(
                    nextId++,
                    candidate.getX(),
                    candidate.getY(),
                    candidate.getWidth(),
                    candidate.getHeight(),
                    candidate.getOrigin()
                ));
            }
        }
        return kept;
    }

    public static double computeSlotIoU(ParkingSlot a, ParkingSlot b) {
        if (a == null || b == null) return 0.0;
        int x1 = Math.max(a.getX(), b.getX());
        int y1 = Math.max(a.getY(), b.getY());
        int x2 = Math.min(a.getX() + a.getWidth(), b.getX() + b.getWidth());
        int y2 = Math.min(a.getY() + a.getHeight(), b.getY() + b.getHeight());
        int iw = Math.max(0, x2 - x1);
        int ih = Math.max(0, y2 - y1);
        double inter = (double) iw * ih;
        if (inter <= 0.0) return 0.0;
        double union = (double) a.getWidth() * a.getHeight() + (double) b.getWidth() * b.getHeight() - inter;
        return union > 0.0 ? inter / union : 0.0;
    }

    public static List<ParkingSlot> generateRowGrid(int startId,
                                                    int rx,
                                                    int ry,
                                                    int rWidth,
                                                    int rHeight,
                                                    int numBays,
                                                    int gapPx) {
        if (startId < 1) throw new IllegalArgumentException("startId must be >= 1.");
        if (rx < 0 || ry < 0) throw new IllegalArgumentException("Row coordinates must be >= 0.");
        if (rWidth <= 0 || rHeight <= 0) throw new IllegalArgumentException("Row dimensions must be > 0.");
        if (numBays < 1) throw new IllegalArgumentException("numBays must be >= 1.");
        if (gapPx < 0) throw new IllegalArgumentException("gapPx must be >= 0.");

        int totalGap = gapPx * (numBays - 1);
        if (rWidth - totalGap < numBays * 4) {
            gapPx = 0;
            totalGap = 0;
        }
        double exactBayW = (double) (rWidth - totalGap) / numBays;
        if (exactBayW < 4.0) {
            throw new IllegalArgumentException(
                "Row width (" + rWidth + "px) is too narrow for " + numBays + " bays.");
        }

        List<ParkingSlot> generated = new ArrayList<>(numBays);
        for (int i = 0; i < numBays; i++) {
            int x0 = rx + (int) Math.round(i * (exactBayW + gapPx));
            int x1 = (i == numBays - 1)
                ? (rx + rWidth)
                : (rx + (int) Math.round((i + 1) * exactBayW + i * gapPx));
            int w = Math.max(4, x1 - x0);
            generated.add(new ParkingSlot(startId + i, x0, ry, w, rHeight, ParkingSlot.GeometryOrigin.ROW_GRID));
        }
        return generated;
    }
}
