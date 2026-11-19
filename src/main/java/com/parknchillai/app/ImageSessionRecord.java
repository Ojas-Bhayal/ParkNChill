package com.parknchillai.app;

import com.parknchillai.ai.SlotTrainingRecord;
import com.parknchillai.imageprocessing.ImageEnhancer;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import com.parknchillai.utils.ImageLoader;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.lang.ref.SoftReference;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Encapsulates the complete state of a single analysed parking lot photograph (Problem 2 &amp; WS-4).
 *
 * <p>Each session maintains a stable identity derived from the image content hash (SHA-256)
 * and metadata, preserving:
 * <ul>
 *   <li>The immutable original photograph and the active working image.</li>
 *   <li>The enhancement settings currently applied.</li>
 *   <li>The parking slot layout, boundary geometry, and origin provenance.</li>
 *   <li>Per-image detection and analysis results.</li>
 *   <li>Provisional and human-verified slot occupancy labels.</li>
 * </ul>
 *
 * <p>Memory Optimization (WS-4): When sessions are inactive, full-resolution bitmaps
 * can be unloaded from direct strong references via {@link #unloadFullImages()}, retaining
 * only a {@link SoftReference} and metadata to reload on demand if reactivated.</p>
 */
public class ImageSessionRecord {

    private final String sessionId;
    private final String sourceImageSha256;
    private final ImageLoader.LoadedImageMetadata metadata;

    private BufferedImage directOriginalImage;
    private BufferedImage directWorkingImage;
    private SoftReference<BufferedImage> softOriginalImage;
    private SoftReference<BufferedImage> softWorkingImage;
    private String enhancementDescription;

    private double contrast = 1.0;
    private double brightness = 0.0;
    private double gamma = 1.0;
    private int noiseKernel = 0;

    private List<ParkingSlot> slots = new ArrayList<>();
    private boolean slotLocalizationReliable;
    private String slotLocalizationMessage;
    private AnalysisSummary analysisSummary;

    private final Map<Integer, SlotTrainingRecord> provisionalRecords = new LinkedHashMap<>();
    private final Map<Integer, SlotStatus> verifiedLabels = new LinkedHashMap<>();
    private final Set<Integer> geometryDirtySlotIds = new LinkedHashSet<>();

    public ImageSessionRecord(String sourceImageSha256,
                              ImageLoader.LoadedImageMetadata metadata,
                              BufferedImage originalImage,
                              BufferedImage workingImage,
                              String enhancementDescription,
                              List<ParkingSlot> slots,
                              boolean slotLocalizationReliable,
                              String slotLocalizationMessage) {
        this.sourceImageSha256 = (sourceImageSha256 == null || sourceImageSha256.isBlank())
            ? "UNHASHED_SESSION_" + System.nanoTime()
            : sourceImageSha256.trim();
        this.metadata = metadata != null ? metadata.withoutImage() : null;
        this.sessionId = this.sourceImageSha256;

        BufferedImage orig = Objects.requireNonNull(originalImage, "originalImage must not be null");
        BufferedImage work = workingImage != null ? workingImage : orig;
        this.directOriginalImage = orig;
        this.directWorkingImage = work;
        this.softOriginalImage = new SoftReference<>(orig);
        this.softWorkingImage = new SoftReference<>(work);

        this.enhancementDescription = (enhancementDescription == null || enhancementDescription.isBlank())
            ? "Original (Unenhanced)"
            : enhancementDescription;
        if (slots != null) {
            this.slots = new ArrayList<>(slots);
        }
        this.slotLocalizationReliable = slotLocalizationReliable;
        this.slotLocalizationMessage = slotLocalizationMessage != null ? slotLocalizationMessage : "";
    }

    public synchronized void loadOrKeepFullImages(BufferedImage orig, BufferedImage work) {
        if (orig != null) {
            this.directOriginalImage = orig;
            this.softOriginalImage = new SoftReference<>(orig);
        }
        if (work != null) {
            this.directWorkingImage = work;
            this.softWorkingImage = new SoftReference<>(work);
        }
    }

    public synchronized void unloadFullImages() {
        this.directOriginalImage = null;
        this.directWorkingImage = null;
        boolean hasRealBackingFile = false;
        if (metadata != null && metadata.filePath() != null && !metadata.filePath().isBlank()) {
            try {
                hasRealBackingFile = new File(metadata.filePath()).exists();
            } catch (Exception ignored) {}
        }
        if (hasRealBackingFile) {
            this.softOriginalImage = null;
            this.softWorkingImage = null;
        }
        if (this.analysisSummary != null) {
            this.analysisSummary.unloadFullImages();
        }
    }

    public synchronized boolean isImagesRetainedDirectly() {
        return directOriginalImage != null || directWorkingImage != null;
    }

    public String getSessionId()                        { return sessionId; }
    public String getSourceImageSha256()                { return sourceImageSha256; }
    public ImageLoader.LoadedImageMetadata getMetadata() { return metadata; }

    public synchronized BufferedImage getOriginalImage() {
        if (directOriginalImage != null) return directOriginalImage;
        BufferedImage soft = softOriginalImage != null ? softOriginalImage.get() : null;
        if (soft != null) return soft;

        if (metadata != null && metadata.filePath() != null && !metadata.filePath().isBlank()) {
            try {
                File file = new File(metadata.filePath());
                if (file.exists()) {
                    BufferedImage reloaded = ImageIO.read(file);
                    if (reloaded != null) {
                        this.softOriginalImage = new SoftReference<>(reloaded);
                        return reloaded;
                    }
                }
            } catch (Exception ignored) {}
        }
        return null;
    }

    public synchronized BufferedImage getWorkingImage() {
        if (directWorkingImage != null) return directWorkingImage;
        BufferedImage soft = softWorkingImage != null ? softWorkingImage.get() : null;
        if (soft != null) return soft;

        BufferedImage orig = getOriginalImage();
        if (orig != null) {
            if (isEnhanced()) {
                BufferedImage reEnhanced = ImageEnhancer.adjust(orig, contrast, brightness, gamma, noiseKernel);
                this.softWorkingImage = new SoftReference<>(reEnhanced);
                return reEnhanced;
            } else {
                this.softWorkingImage = new SoftReference<>(orig);
                return orig;
            }
        }
        return null;
    }

    public boolean isEnhanced() {
        return enhancementDescription != null && !enhancementDescription.equalsIgnoreCase("Original (Unenhanced)")
            && !enhancementDescription.equalsIgnoreCase("Original");
    }

    public String getEnhancementDescription()           { return enhancementDescription; }

    public synchronized void setWorkingImage(BufferedImage workingImage, String description) {
        setWorkingImage(workingImage, description, 1.0, 0.0, 1.0, 0);
    }

    public synchronized void setWorkingImage(BufferedImage workingImage, String description,
                                             double contrast, double brightness, double gamma, int noiseKernel) {
        BufferedImage orig = getOriginalImage();
        this.directWorkingImage = workingImage != null ? workingImage : orig;
        this.softWorkingImage = new SoftReference<>(this.directWorkingImage);
        this.enhancementDescription = description != null ? description : "Custom Enhancement";
        this.contrast = contrast;
        this.brightness = brightness;
        this.gamma = gamma;
        this.noiseKernel = noiseKernel;
    }

    public String getFileName() {
        if (metadata != null && metadata.fileName() != null && !metadata.fileName().isBlank()) {
            return metadata.fileName();
        }
        return "parking_image_" + (sourceImageSha256.length() > 8 ? sourceImageSha256.substring(0, 8) : sourceImageSha256) + ".png";
    }

    public List<ParkingSlot> getSlots() {
        return Collections.unmodifiableList(slots);
    }

    public void setSlots(List<ParkingSlot> newSlots) {
        this.slots = new ArrayList<>(newSlots != null ? newSlots : List.of());
    }

    public boolean isSlotLocalizationReliable() { return slotLocalizationReliable; }
    public void setSlotLocalizationReliable(boolean reliable) { this.slotLocalizationReliable = reliable; }

    public String getSlotLocalizationMessage() { return slotLocalizationMessage; }
    public void setSlotLocalizationMessage(String message) { this.slotLocalizationMessage = message; }

    public AnalysisSummary getAnalysisSummary() { return analysisSummary; }
    public synchronized void setAnalysisSummary(AnalysisSummary summary) {
        this.analysisSummary = summary;
        if (!isImagesRetainedDirectly() && summary != null) {
            summary.unloadFullImages();
        }
    }

    public Map<Integer, SlotTrainingRecord> getProvisionalRecords() {
        return provisionalRecords;
    }

    public Map<Integer, SlotStatus> getVerifiedLabels() {
        return verifiedLabels;
    }

    public Set<Integer> getGeometryDirtySlotIds() {
        return geometryDirtySlotIds;
    }

    public void markSlotGeometryModified(int slotId) {
        geometryDirtySlotIds.add(slotId);
        verifiedLabels.remove(slotId);
    }

    public boolean isSlotGeometryUnconfirmed(int slotId) {
        return geometryDirtySlotIds.contains(slotId);
    }

    public int getSlotCount() {
        return slots.size();
    }

    public int getConfirmedCount() {
        return verifiedLabels.size();
    }

    public int getOccupiedCount() {
        if (analysisSummary != null) {
            return analysisSummary.getOccupiedSlots();
        }
        int count = 0;
        for (SlotStatus s : verifiedLabels.values()) {
            if (s == SlotStatus.OCCUPIED) count++;
        }
        return count;
    }

    public int getAvailableCount() {
        if (analysisSummary != null) {
            return analysisSummary.getAvailableSlots();
        }
        int count = 0;
        for (SlotStatus s : verifiedLabels.values()) {
            if (s == SlotStatus.AVAILABLE) count++;
        }
        return count;
    }

    private String cachedFullPixelSha256 = null;

    /**
     * Returns the full-pixel-content SHA-256 for this session's original image,
     * computed once and cached on this session record without repeated re-hashing.
     */
    public synchronized String getOrComputeFullPixelSha256() {
        if (cachedFullPixelSha256 == null) {
            BufferedImage orig = getOriginalImage();
            if (orig != null) {
                cachedFullPixelSha256 = com.parknchillai.detection.OnnxVehicleDetectionService.computeFullPixelSha256(orig);
            }
        }
        return cachedFullPixelSha256;
    }

    /**
     * Resolves the vehicle detection cache identity using the documented three-tier policy:
     * 1. Verified source-file SHA-256 if the image pixels are unmodified from the loaded source file.
     * 2. Full-pixel-content SHA-256 computed once per image session and stored on this record.
     * 3. null (no caching) if neither is available.
     */
    public synchronized String resolveDetectionCacheIdentity() {
        if (metadata != null && !metadata.wasResized() && !metadata.perspectiveWarped()
                && sourceImageSha256 != null && !sourceImageSha256.startsWith("UNHASHED_")) {
            return sourceImageSha256;
        }
        return getOrComputeFullPixelSha256();
    }

    @Override
    public String toString() {
        return getFileName() + " (" + slots.size() + " slots, SHA: "
            + (sourceImageSha256.length() > 8 ? sourceImageSha256.substring(0, 8) : sourceImageSha256) + ")";
    }
}
