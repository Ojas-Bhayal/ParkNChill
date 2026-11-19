package com.parknchillai.app;

import com.parknchillai.ai.DatasetManifest;
import com.parknchillai.ai.DecisionTreeModel;
import com.parknchillai.ai.DecisionTreePredictor;
import com.parknchillai.ai.ModelEvaluator;
import com.parknchillai.ai.ModelSerializer;
import com.parknchillai.ai.PkLotDatasetImporter;
import com.parknchillai.ai.SlotTrainingRecord;
import com.parknchillai.detection.VehicleDetectionService;
import com.parknchillai.features.FeatureVector;
import com.parknchillai.parking.ParkingSlot;
import com.parknchillai.parking.SlotStatus;
import com.parknchillai.utils.ImageLoader;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Centralized shared-state model ({@code AppState}) serving as the single source of truth
 * across the ParkNChill v2.3 workflow. The UI groups internal phases into four simpler steps.
 *
 * <ul>
 *   <li><strong>Strict Separation of User-Reviewed vs. PKLot Datasets (Requirement 2)</strong>:
 *       Maintains separate pools for manually reviewed user-collected examples
 *       ({@link ActiveDatasetMode#USER_REVIEWED_SLOTS}) and imported PKLot datasets
 *       ({@link ActiveDatasetMode#PKLOT_IMPORTED_DATASET}) so datasets with different partition
 *       definitions or provenance are never silently mixed.</li>
 *   <li><strong>Partition-Aware PKLot Training (Requirement 2 &amp; 3)</strong>:
 *       When {@link ActiveDatasetMode#PKLOT_IMPORTED_DATASET} is active, routes
 *       {@link #trainTenFeatureDecisionTree(int, int)} through
 *       {@link ParkingAnalysisService#trainFromPartitionedPkLotRecords}, preserving the saved
 *       {@code TRAIN}, {@code VALIDATION}, and {@code TEST} camera/date partitions, selecting
 *       hyperparameters on {@code VALIDATION}, and evaluating once on the frozen {@code TEST} set.</li>
 * </ul>
 */
public class AppState {

    /**
     * The internal workflow stages in ParkNChill v2.3. The UI displays four grouped steps.
     */
    public enum WorkflowStage {
        LOAD_IMAGE(1, "1. Load Image", "Load Image"),
        DETECT_VEHICLES_AND_SLOTS(2, "2. Detect Vehicles & Slots", "Detect Vehicles and Slots"),
        REVIEW_OCCUPANCY(3, "3. Review Occupancy", "Review Occupancy"),
        TRAIN_DECISION_TREE(4, "4. Train Decision Tree", "Train Decision Tree"),
        EVALUATE_MODEL(5, "5. Evaluate Model", "Evaluate Model"),
        SAVE_AND_USE_MODEL(6, "6. Save & Use Model", "Save and Use Model");

        private final int stepNumber;
        private final String numberedTitle;
        private final String shortTitle;

        WorkflowStage(int stepNumber, String numberedTitle, String shortTitle) {
            this.stepNumber = stepNumber;
            this.numberedTitle = numberedTitle;
            this.shortTitle = shortTitle;
        }

        public int getStepNumber()       { return stepNumber; }
        public String getNumberedTitle() { return numberedTitle; }
        public String getShortTitle()    { return shortTitle; }

        public WorkflowStage previous() {
            WorkflowStage[] all = values();
            return ordinal() > 0 ? all[ordinal() - 1] : null;
        }

        public WorkflowStage next() {
            WorkflowStage[] all = values();
            return ordinal() < all.length - 1 ? all[ordinal() + 1] : null;
        }
    }

    /**
     * Visual and semantic state of each workflow stage in the top stepper.
     */
    public enum StageVisualState {
        COMPLETED("COMPLETED"),
        ACTIVE("ACTIVE"),
        BLOCKED("BLOCKED"),
        PENDING("PENDING");

        private final String badgeText;
        StageVisualState(String badgeText) { this.badgeText = badgeText; }
        public String getBadgeText() { return badgeText; }
    }

    /**
     * Clearly separated dataset workflow modes (Requirement 2: do not silently mix user-reviewed
     * interactive slots with partitioned PKLot datasets).
     */
    public enum ActiveDatasetMode {
        USER_REVIEWED_SLOTS("My Reviewed Slots"),
        PKLOT_IMPORTED_DATASET("PKLot Dataset");

        private final String displayName;
        ActiveDatasetMode(String displayName) { this.displayName = displayName; }
        public String getDisplayName() { return displayName; }
        @Override public String toString() { return displayName; }
    }

    private final ParkingAnalysisService analysisService;
    private WorkflowStage currentStage = WorkflowStage.LOAD_IMAGE;
    private ActiveDatasetMode activeDatasetMode = ActiveDatasetMode.USER_REVIEWED_SLOTS;

    // Image & Detection State
    private BufferedImage currentImage;
    private BufferedImage originalImage;
    private BufferedImage activeWorkingImage;
    private String activeEnhancementDescription = "Original";
    private ImageLoader.LoadedImageMetadata currentImageMetadata;
    private String currentImageSha256 = "UNHASHED_MEMORY_IMAGE";
    private List<ParkingSlot> currentSlots = new ArrayList<>();
    private boolean slotLocalizationReliable = false;
    private String slotLocalizationMessage = "No image loaded yet.";
    private AnalysisSummary currentSummary;

    // Multi-Image Sessions (Problem 2 & WS-4 Bounded Memory)
    public static final int MAX_ACTIVE_SESSION_IMAGES = 3;
    private final Map<String, ImageSessionRecord> imageSessions = new LinkedHashMap<>();
    private final LinkedHashSet<String> activeLruSessionIds = new LinkedHashSet<>();
    private String activeSessionId = null;

    // Separate Pools: (1) Interactive User-Reviewed Pool vs (2) Partitioned PKLot Pool
    private final Map<Integer, SlotTrainingRecord> currentImageProvisionalRecords = new LinkedHashMap<>();
    private final Map<Integer, SlotStatus> currentImageVerifiedLabels = new LinkedHashMap<>();
    private final Set<Integer> geometryDirtySlotIds = new LinkedHashSet<>();
    private final Map<String, SlotTrainingRecord> userReviewedTrainingPool = new LinkedHashMap<>();
    private final Map<String, SlotTrainingRecord> pkLotPartitionedPool = new LinkedHashMap<>();

    // Dataset & PKLot State
    private String datasetSourceDescription = "Interactive User-Reviewed Slots";
    private String datasetSchemaVersion = FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION;
    private PkLotDatasetImporter.PkLotImportResult lastPkLotImportResult;
    private PkLotDatasetImporter.AnnotationInspectionReport lastPkLotInspectionReport;
    private Path frozenPkLotTestMetaPath;

    // Model & Evaluation State
    private boolean compatibleTenFeatureModelActive = false;
    private String activeModelSourceDescription = "Bundled 8-Feature Legacy Model (v2.2-8feat)";
    private ModelEvaluator.EvaluationReport lastEvaluationReport;
    private String reviewerId = "";

    private final List<Runnable> stateChangeListeners = new ArrayList<>();

    public AppState(ParkingAnalysisService analysisService) {
        if (analysisService == null) {
            throw new IllegalArgumentException("ParkingAnalysisService must not be null.");
        }
        this.analysisService = analysisService;
        syncModelCompatibilityFromService();
    }

    public void syncModelCompatibilityFromService() {
        DecisionTreeModel m = (analysisService.getPredictor() != null)
            ? analysisService.getPredictor().getModel()
            : null;
        if (m != null && m.getFeatureCount() == FeatureVector.DETECTION_AUGMENTED_FEATURE_COUNT) {
            this.compatibleTenFeatureModelActive = true;
            this.datasetSchemaVersion = FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION;
        } else {
            this.compatibleTenFeatureModelActive = false;
        }
    }

    public void addStateChangeListener(Runnable listener) {
        if (listener != null) stateChangeListeners.add(listener);
    }

    public void notifyStateChanged() {
        for (Runnable r : new ArrayList<>(stateChangeListeners)) {
            r.run();
        }
    }

    // -------------------------------------------------------------------------
    // Workflow Stage & Prerequisite Logic (Requirement 1)
    // -------------------------------------------------------------------------

    public WorkflowStage getCurrentStage() {
        return currentStage;
    }

    public void setCurrentStage(WorkflowStage stage) {
        if (stage == null) throw new IllegalArgumentException("WorkflowStage must not be null.");
        this.currentStage = stage;
        notifyStateChanged();
    }

    public ActiveDatasetMode getActiveDatasetMode() {
        return activeDatasetMode;
    }

    public void setActiveDatasetMode(ActiveDatasetMode mode) {
        if (mode == null) throw new IllegalArgumentException("ActiveDatasetMode must not be null.");
        if (this.activeDatasetMode != mode) {
            this.activeDatasetMode = mode;
            this.datasetSourceDescription = mode.getDisplayName();
            // Metrics belong to the dataset that produced them. Do not show old test results
            // after the user switches to a different training pool.
            this.lastEvaluationReport = null;
        }
        notifyStateChanged();
    }

    public boolean isStageCompleted(WorkflowStage stage) {
        return switch (stage) {
            case LOAD_IMAGE -> currentImage != null;
            case DETECT_VEHICLES_AND_SLOTS -> currentImage != null && currentSummary != null;
            case REVIEW_OCCUPANCY -> !getEligibleReviewedRecords().isEmpty()
                || (currentSummary != null && !currentImageVerifiedLabels.isEmpty() && geometryDirtySlotIds.isEmpty());
            case TRAIN_DECISION_TREE -> compatibleTenFeatureModelActive;
            case EVALUATE_MODEL -> compatibleTenFeatureModelActive && lastEvaluationReport != null;
            case SAVE_AND_USE_MODEL -> compatibleTenFeatureModelActive && lastEvaluationReport != null;
        };
    }

    public boolean isStageBlocked(WorkflowStage stage) {
        return switch (stage) {
            case LOAD_IMAGE -> false;
            case DETECT_VEHICLES_AND_SLOTS -> currentImage == null;
            case REVIEW_OCCUPANCY -> currentImage == null || currentSummary == null;
            case TRAIN_DECISION_TREE -> !canTrainDecisionTree() && !compatibleTenFeatureModelActive;
            case EVALUATE_MODEL -> !compatibleTenFeatureModelActive;
            case SAVE_AND_USE_MODEL -> false;
        };
    }

    public StageVisualState getStageVisualState(WorkflowStage stage) {
        if (stage == currentStage) {
            return StageVisualState.ACTIVE;
        }
        if (isStageCompleted(stage)) {
            return StageVisualState.COMPLETED;
        }
        if (isStageBlocked(stage)) {
            return StageVisualState.BLOCKED;
        }
        return StageVisualState.PENDING;
    }

    public String getStagePrerequisiteExplanation(WorkflowStage stage) {
        return switch (stage) {
            case LOAD_IMAGE -> currentImage != null
                ? "Image loaded (" + currentImage.getWidth() + "×" + currentImage.getHeight() + " px). Proceed to Step 2."
                : "Prerequisite: Open a parking lot image or click 'Load Demo Image' to begin.";
            case DETECT_VEHICLES_AND_SLOTS -> {
                if (currentImage == null) {
                    yield "Prerequisite: Load a parking image in Step 1 before running vehicle and slot detection.";
                }
                if (currentSummary == null) {
                    yield "Prerequisite: Click 'Detect and Analyse' to detect vehicles and localize parking spaces.";
                }
                yield "Detection complete (" + currentSummary.getTotalSlots() + " slots analyzed). Proceed to Step 3 to review occupancy.";
            }
            case REVIEW_OCCUPANCY -> {
                if (currentImage == null) {
                    yield "Prerequisite: Load a parking image in Step 1 first.";
                }
                if (currentSummary == null) {
                    yield "Prerequisite: Run 'Detect and Analyse' in Step 2 to generate provisional slot predictions before reviewing.";
                }
                if (!geometryDirtySlotIds.isEmpty()) {
                    yield "Action Required: Slot boundary geometry was modified for Slot(s) " + geometryDirtySlotIds
                        + ". Select each modified slot and click 'Mark Occupied' or 'Mark Available' to confirm its label before training.";
                }
                yield "Select any slot in the table or image to confirm its occupancy label (Mark Occupied / Mark Available).";
            }
            case TRAIN_DECISION_TREE -> {
                String disabledReason = getTrainingDisabledReason();
                if (disabledReason != null) {
                    yield disabledReason;
                }
                yield String.format(Locale.ROOT,
                    "Ready to train 10-feature CART Decision Tree [%s]: %d eligible records (%d Imported Ground Truth, %d Human-Verified).",
                    activeDatasetMode.getDisplayName(),
                    getEligibleReviewedRecords().size(),
                    getEligibleImportedGroundTruthCount(),
                    getEligibleHumanVerifiedCount());
            }
            case EVALUATE_MODEL -> {
                if (!compatibleTenFeatureModelActive) {
                    yield "Prerequisite: Train or load a compatible 10-feature Decision Tree in Step 4 or Step 6 before viewing held-out test metrics.";
                }
                yield "Evaluating the trained 10-feature Decision Tree on the frozen Held-Out Test Partition.";
            }
            case SAVE_AND_USE_MODEL -> compatibleTenFeatureModelActive
                ? "Compatible 10-feature Decision Tree is active. Save it to disk or return to image analysis."
                : "No trained 10-feature model is active yet. Load a saved 10-feature model file or train one in Step 4.";
        };
    }

    // -------------------------------------------------------------------------
    // Reviewer ID Validation (Requirement 2.A)
    // -------------------------------------------------------------------------

    public String getReviewerId() {
        return reviewerId == null ? "" : reviewerId.trim();
    }

    public void setReviewerId(String id) {
        this.reviewerId = id == null ? "" : id.trim();
        notifyStateChanged();
    }

    public boolean isReviewerIdValid() {
        String clean = getReviewerId();
        return !clean.isEmpty()
            && clean.length() >= 2
            && !"UNKNOWN".equalsIgnoreCase(clean)
            && !"PKLOT_ANNOTATION".equalsIgnoreCase(clean);
    }

    public String getReviewerValidationMessage() {
        if (isReviewerIdValid()) {
            return "Reviewer ID ready (" + getReviewerId() + ")";
        }
        return "Enter a valid human Reviewer ID (at least 2 characters, e.g. OPERATOR_1) to confirm labels.";
    }

    // -------------------------------------------------------------------------
    // Image, Detection & Slot Review State (Requirements 2.B, 2.C, 2.D)
    // -------------------------------------------------------------------------

    public void setLoadedImageAndSlots(ImageLoader.LoadedImageMetadata metadata,
                                       String imageSha256,
                                       List<ParkingSlot> slots,
                                       boolean reliable,
                                       String localizationMessage) {
        this.currentImageMetadata = metadata;
        this.originalImage = metadata != null ? metadata.image() : null;
        this.activeWorkingImage = this.originalImage;
        this.currentImage = this.originalImage;
        this.activeEnhancementDescription = "Original";
        String resolvedSha = (imageSha256 != null && !imageSha256.isBlank()) ? imageSha256.trim() : null;
        if (resolvedSha == null && metadata != null && metadata.filePath() != null && !metadata.filePath().isBlank()) {
            resolvedSha = DatasetManifest.computeFileSha256OrEmpty(Path.of(metadata.filePath()));
        }
        this.currentImageSha256 = (resolvedSha != null && !resolvedSha.isBlank()) ? resolvedSha : "UNHASHED_MEMORY_IMAGE";
        this.currentSlots = slots != null ? new ArrayList<>(slots) : new ArrayList<>();
        this.slotLocalizationReliable = reliable;
        this.slotLocalizationMessage = localizationMessage != null ? localizationMessage : "";
        this.activeSessionId = this.currentImageSha256;

        ImageSessionRecord existingSession = imageSessions.get(this.activeSessionId);
        if (existingSession != null) {
            existingSession.setSlots(this.currentSlots);
            existingSession.setSlotLocalizationReliable(reliable);
            existingSession.setSlotLocalizationMessage(localizationMessage);
            this.activeWorkingImage = existingSession.getWorkingImage();
            this.currentImage = existingSession.getWorkingImage();
            this.activeEnhancementDescription = existingSession.getEnhancementDescription();
            this.currentSummary = existingSession.getAnalysisSummary();
            this.currentImageProvisionalRecords.clear();
            this.currentImageProvisionalRecords.putAll(existingSession.getProvisionalRecords());
            this.currentImageVerifiedLabels.clear();
            this.currentImageVerifiedLabels.putAll(existingSession.getVerifiedLabels());
            this.geometryDirtySlotIds.clear();
            this.geometryDirtySlotIds.addAll(existingSession.getGeometryDirtySlotIds());
        } else {
            this.currentSummary = null;
            this.currentImageProvisionalRecords.clear();
            this.currentImageVerifiedLabels.clear();
            this.geometryDirtySlotIds.clear();
            if (this.originalImage != null) {
                ImageSessionRecord newSession = new ImageSessionRecord(
                    this.activeSessionId, metadata, this.originalImage, this.activeWorkingImage,
                    this.activeEnhancementDescription, this.currentSlots, reliable, localizationMessage
                );
                imageSessions.put(this.activeSessionId, newSession);
            }
        }
        touchSession(this.activeSessionId, this.originalImage, this.activeWorkingImage);
        notifyStateChanged();
    }

    public void clearImageState() {
        this.currentImageMetadata = null;
        this.originalImage = null;
        this.activeWorkingImage = null;
        this.currentImage = null;
        this.activeEnhancementDescription = "Original";
        this.currentImageSha256 = "UNHASHED_MEMORY_IMAGE";
        this.currentSlots.clear();
        this.slotLocalizationReliable = false;
        this.slotLocalizationMessage = "No image loaded.";
        this.currentSummary = null;
        this.currentImageProvisionalRecords.clear();
        this.currentImageVerifiedLabels.clear();
        this.geometryDirtySlotIds.clear();
        this.currentStage = WorkflowStage.LOAD_IMAGE;
        notifyStateChanged();
    }

    public void updateAnalysisSummary(AnalysisSummary summary) {
        this.currentSummary = summary;
        this.currentImageProvisionalRecords.clear();
        if (summary != null) {
            String srcName = currentImageMetadata != null ? currentImageMetadata.fileName() : "loaded_parking_lot.png";
            String capGrp = "LOT_SESSION_" + srcName.replaceAll("[^a-zA-Z0-9]", "_");
            List<SlotTrainingRecord> provList = analysisService.generateProvisionalSlotRecords(
                srcName, currentImageSha256, capGrp, summary
            );
            for (SlotTrainingRecord rec : provList) {
                currentImageProvisionalRecords.put(rec.slotId(), rec);
            }
        }
        if (activeSessionId != null && imageSessions.containsKey(activeSessionId)) {
            ImageSessionRecord sess = imageSessions.get(activeSessionId);
            sess.setAnalysisSummary(summary);
            sess.getProvisionalRecords().clear();
            sess.getProvisionalRecords().putAll(currentImageProvisionalRecords);
        }
        notifyStateChanged();
    }

    public void applyWorkingImageEnhancement(BufferedImage enhancedImage, String description) {
        if (enhancedImage == null) return;
        this.activeWorkingImage = enhancedImage;
        this.currentImage = enhancedImage;
        this.activeEnhancementDescription = (description != null && !description.isBlank())
            ? description : "Custom Enhancement";

        if (activeSessionId != null && imageSessions.containsKey(activeSessionId)) {
            imageSessions.get(activeSessionId).setWorkingImage(enhancedImage, this.activeEnhancementDescription);
        }
        touchSession(this.activeSessionId, this.originalImage, this.activeWorkingImage);

        if (this.currentSummary != null && this.originalImage != null && !this.currentSlots.isEmpty()) {
            AnalysisSummary recomputed = analysisService.analyze(
                this.originalImage,
                this.activeWorkingImage,
                this.currentSlots,
                analysisService.isUseAdaptiveThreshold(),
                this.currentSummary.getAnalysisMode(),
                this.slotLocalizationReliable,
                this.slotLocalizationMessage
            );
            updateAnalysisSummary(recomputed);
        } else {
            notifyStateChanged();
        }
    }

    public void resetWorkingImageToOriginal() {
        if (this.originalImage == null) return;
        this.activeWorkingImage = this.originalImage;
        this.currentImage = this.originalImage;
        this.activeEnhancementDescription = "Original (Unenhanced)";

        if (activeSessionId != null && imageSessions.containsKey(activeSessionId)) {
            imageSessions.get(activeSessionId).setWorkingImage(this.originalImage, this.activeEnhancementDescription);
        }

        if (this.currentSummary != null && !this.currentSlots.isEmpty()) {
            AnalysisSummary recomputed = analysisService.analyze(
                this.originalImage,
                this.originalImage,
                this.currentSlots,
                analysisService.isUseAdaptiveThreshold(),
                this.currentSummary.getAnalysisMode(),
                this.slotLocalizationReliable,
                this.slotLocalizationMessage
            );
            updateAnalysisSummary(recomputed);
        } else {
            notifyStateChanged();
        }
    }

    public boolean isWorkingImageEnhanced() {
        return originalImage != null && activeWorkingImage != null && originalImage != activeWorkingImage;
    }

    public BufferedImage getOriginalImage() {
        return originalImage != null ? originalImage : currentImage;
    }

    public BufferedImage getActiveWorkingImage() {
        return activeWorkingImage != null ? activeWorkingImage : currentImage;
    }

    public String getActiveEnhancementDescription() {
        return activeEnhancementDescription != null ? activeEnhancementDescription : "Original";
    }

    public void switchToSession(String sessionId) {
        if (sessionId == null || !imageSessions.containsKey(sessionId)) return;
        ImageSessionRecord session = imageSessions.get(sessionId);
        this.activeSessionId = sessionId;
        this.currentImageMetadata = session.getMetadata();
        this.originalImage = session.getOriginalImage();
        this.activeWorkingImage = session.getWorkingImage();
        this.currentImage = session.getWorkingImage();
        this.activeEnhancementDescription = session.getEnhancementDescription();
        this.currentImageSha256 = session.getSourceImageSha256();
        this.currentSlots = new ArrayList<>(session.getSlots());
        this.slotLocalizationReliable = session.isSlotLocalizationReliable();
        this.slotLocalizationMessage = session.getSlotLocalizationMessage();
        this.currentSummary = session.getAnalysisSummary();
        this.currentImageProvisionalRecords.clear();
        this.currentImageProvisionalRecords.putAll(session.getProvisionalRecords());
        this.currentImageVerifiedLabels.clear();
        this.currentImageVerifiedLabels.putAll(session.getVerifiedLabels());
        this.geometryDirtySlotIds.clear();
        this.geometryDirtySlotIds.addAll(session.getGeometryDirtySlotIds());
        touchSession(sessionId, this.originalImage, this.activeWorkingImage);
        notifyStateChanged();
    }

    private synchronized void touchSession(String sessionId, BufferedImage orig, BufferedImage work) {
        if (sessionId == null) return;
        activeLruSessionIds.remove(sessionId);
        activeLruSessionIds.add(sessionId);

        ImageSessionRecord session = imageSessions.get(sessionId);
        if (session != null && orig != null) {
            session.loadOrKeepFullImages(orig, work != null ? work : orig);
        }

        // Bounded multi-image session cache (WS-4):
        // Cap retained full-resolution images to MAX_ACTIVE_SESSION_IMAGES (default 3)
        while (activeLruSessionIds.size() > MAX_ACTIVE_SESSION_IMAGES) {
            String oldestId = activeLruSessionIds.iterator().next();
            if (Objects.equals(oldestId, this.activeSessionId)) {
                break;
            }
            activeLruSessionIds.remove(oldestId);
            ImageSessionRecord toEvict = imageSessions.get(oldestId);
            if (toEvict != null) {
                toEvict.unloadFullImages();
            }
        }
    }

    private Runnable onSessionClosedCallback;

    public void setOnSessionClosedCallback(Runnable callback) {
        this.onSessionClosedCallback = callback;
    }

    public synchronized boolean closeSession(String sessionId) {
        if (sessionId == null || !imageSessions.containsKey(sessionId)) {
            return false;
        }
        ImageSessionRecord removed = imageSessions.remove(sessionId);
        activeLruSessionIds.remove(sessionId);
        if (removed != null) {
            removed.unloadFullImages();
        }

        if (Objects.equals(activeSessionId, sessionId)) {
            if (!imageSessions.isEmpty()) {
                String nextId = imageSessions.keySet().iterator().next();
                switchToSession(nextId);
            } else {
                clearImageState();
                activeSessionId = null;
            }
        }
        if (onSessionClosedCallback != null) {
            try { onSessionClosedCallback.run(); } catch (Exception ignored) {}
        }
        notifyStateChanged();
        return true;
    }

    public synchronized void clearAllSessions(boolean preserveReviewedTrainingData) {
        for (ImageSessionRecord s : imageSessions.values()) {
            s.unloadFullImages();
        }
        imageSessions.clear();
        activeLruSessionIds.clear();
        activeSessionId = null;
        clearImageState();
        if (!preserveReviewedTrainingData) {
            userReviewedTrainingPool.clear();
        }
        if (onSessionClosedCallback != null) {
            try { onSessionClosedCallback.run(); } catch (Exception ignored) {}
        }
        notifyStateChanged();
    }

    public void startFreshDataset(boolean confirm) {
        if (!confirm) return;
        clearAllSessions(false);
    }

    public String getActiveSessionId() {
        return activeSessionId;
    }

    public int getDirectlyRetainedSessionImageCount() {
        return (int) imageSessions.values().stream()
            .filter(ImageSessionRecord::isImagesRetainedDirectly)
            .count();
    }

    public int getTotalImagesAnalysed() {
        return (int) imageSessions.values().stream()
            .filter(s -> s.getAnalysisSummary() != null)
            .count();
    }

    public int getTotalSlotsCollected() {
        return imageSessions.values().stream()
            .mapToInt(ImageSessionRecord::getSlotCount)
            .sum();
    }

    public int getTotalOccupiedSlots() {
        return imageSessions.values().stream()
            .mapToInt(ImageSessionRecord::getOccupiedCount)
            .sum();
    }

    public int getTotalAvailableSlots() {
        return imageSessions.values().stream()
            .mapToInt(ImageSessionRecord::getAvailableCount)
            .sum();
    }

    public List<ImageSessionRecord> getAllImageSessions() {
        return Collections.unmodifiableList(new ArrayList<>(imageSessions.values()));
    }

    public ImageSessionRecord getActiveSession() {
        return activeSessionId != null ? imageSessions.get(activeSessionId) : null;
    }

    public void markSlotGeometryModified(int slotId) {
        geometryDirtySlotIds.add(slotId);
        SlotTrainingRecord prev = currentImageProvisionalRecords.get(slotId);
        if (prev != null) {
            userReviewedTrainingPool.remove(prev.sampleId());
        }
        currentImageVerifiedLabels.remove(slotId);
        notifyStateChanged();
    }

    public boolean isSlotGeometryUnconfirmed(int slotId) {
        return geometryDirtySlotIds.contains(slotId);
    }

    public Set<Integer> getGeometryDirtySlotIds() {
        return Collections.unmodifiableSet(geometryDirtySlotIds);
    }

    /**
     * Confirms a slot's occupancy label ({@code OCCUPIED} or {@code AVAILABLE}) in the
     * user-reviewed workflow ({@link ActiveDatasetMode#USER_REVIEWED_SLOTS}).
     */
    public SlotTrainingRecord reviewSlot(int slotId, SlotStatus status) {
        if (status == null) {
            throw new IllegalArgumentException("SlotStatus must not be null.");
        }
        SlotTrainingRecord prov = currentImageProvisionalRecords.get(slotId);
        if (prov == null) {
            throw new IllegalStateException("Slot #" + slotId + " has not been analyzed yet. Run 'Detect and Analyse' first.");
        }

        if (status == SlotStatus.NEEDS_REVIEW) {
            currentImageVerifiedLabels.remove(slotId);
            userReviewedTrainingPool.remove(prov.sampleId());
            if (activeSessionId != null && imageSessions.containsKey(activeSessionId)) {
                ImageSessionRecord sess = imageSessions.get(activeSessionId);
                sess.getVerifiedLabels().remove(slotId);
            }
            if (currentSummary != null) {
                for (SlotAnalysisResult r : currentSummary.getSlotResults()) {
                    if (r.getId() == slotId) {
                        r.getSlot().setStatus(SlotStatus.NEEDS_REVIEW);
                    }
                }
            }
            notifyStateChanged();
            return prov;
        }

        if (!isReviewerIdValid()) {
            throw new IllegalArgumentException(getReviewerValidationMessage());
        }

        SlotTrainingRecord verified = prov.promoteToHumanVerified(getReviewerId(), status);
        currentImageVerifiedLabels.put(slotId, status);
        geometryDirtySlotIds.remove(slotId);
        userReviewedTrainingPool.put(verified.sampleId(), verified);

        if (activeSessionId != null && imageSessions.containsKey(activeSessionId)) {
            ImageSessionRecord sess = imageSessions.get(activeSessionId);
            sess.getVerifiedLabels().put(slotId, status);
            sess.getGeometryDirtySlotIds().remove(slotId);
        }

        if (pkLotPartitionedPool.isEmpty()) {
            this.activeDatasetMode = ActiveDatasetMode.USER_REVIEWED_SLOTS;
            this.datasetSourceDescription = "Interactive User-Reviewed Slots (" + userReviewedTrainingPool.size() + " records)";
        }

        if (currentSummary != null) {
            for (SlotAnalysisResult r : currentSummary.getSlotResults()) {
                if (r.getId() == slotId) {
                    r.getSlot().setStatus(status);
                }
            }
        }
        notifyStateChanged();
        return verified;
    }

    /**
     * Applies an auditable user correction to an imported PKLot ground-truth record without
     * overwriting its {@link SlotTrainingRecord#originalSourceLabel()} (Requirement 3).
     */
    public SlotTrainingRecord applyAuditedCorrectionToPkLotRecord(String sampleId,
                                                                  SlotStatus correctedLabel,
                                                                  String reason) {
        if (!isReviewerIdValid()) {
            throw new IllegalArgumentException(getReviewerValidationMessage());
        }
        SlotTrainingRecord existing = pkLotPartitionedPool.get(sampleId);
        if (existing == null) {
            throw new IllegalArgumentException("PKLot record not found: " + sampleId);
        }
        if ("TEST".equalsIgnoreCase(existing.partition())) {
            throw new IllegalStateException(
                "Record '" + sampleId + "' belongs to the frozen TEST partition and cannot be modified without an explicit evaluation version bump.");
        }
        SlotTrainingRecord corrected = existing.withAuditedUserCorrection(getReviewerId(), correctedLabel, reason);
        pkLotPartitionedPool.put(sampleId, corrected);
        notifyStateChanged();
        return corrected;
    }

    public SlotStatus getVerifiedLabelForSlot(int slotId) {
        return currentImageVerifiedLabels.get(slotId);
    }

    public String getReviewStatusForSlot(int slotId) {
        if (geometryDirtySlotIds.contains(slotId)) {
            return "Geometry Modified — Re-Confirm Label";
        }
        SlotStatus ver = currentImageVerifiedLabels.get(slotId);
        if (ver != null) {
            return "Confirmed (" + getReviewerId() + ")";
        }
        SlotTrainingRecord prov = currentImageProvisionalRecords.get(slotId);
        if (prov != null) {
            return "Provisional (Awaiting Review)";
        }
        return "Unanalyzed";
    }

    public int getConfirmedSlotCountForCurrentImage() {
        return currentImageVerifiedLabels.size();
    }

    // -------------------------------------------------------------------------
    // Separate Training Pools, PKLot Partition Routing & Model Training (Requirements 2 & 3)
    // -------------------------------------------------------------------------

    /**
     * Atomically replaces the PKLot dataset pool only when a valid import completes.
     * If {@code importResult} has 0 valid records (e.g., due to detector failure or invalid directory),
     * preserves any previously valid dataset in {@code pkLotPartitionedPool} (Requirement 4).
     */
    public void registerPkLotImportResult(PkLotDatasetImporter.PkLotImportResult importResult) {
        this.lastPkLotImportResult = importResult;
        if (importResult != null && importResult.eligibleRecords() != null && !importResult.eligibleRecords().isEmpty()) {
            // A new imported dataset invalidates evaluation metrics from the previous dataset.
            this.lastEvaluationReport = null;
            pkLotPartitionedPool.clear();
            for (SlotTrainingRecord rec : importResult.eligibleRecords()) {
                pkLotPartitionedPool.put(rec.sampleId(), rec);
            }
            this.activeDatasetMode = ActiveDatasetMode.PKLOT_IMPORTED_DATASET;
            this.datasetSourceDescription = "PKLot Imported Dataset (" + importResult.rootDirectory().getFileName() + ")";
            this.datasetSchemaVersion = FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION;
            this.frozenPkLotTestMetaPath = importResult.rootDirectory().resolve("pklot_frozen_test.meta");
        }
        notifyStateChanged();
    }

    public void setLastPkLotInspectionReport(PkLotDatasetImporter.AnnotationInspectionReport report) {
        this.lastPkLotInspectionReport = report;
        notifyStateChanged();
    }

    public PkLotDatasetImporter.PkLotImportResult getLastPkLotImportResult() {
        return lastPkLotImportResult;
    }

    public PkLotDatasetImporter.AnnotationInspectionReport getLastPkLotInspectionReport() {
        return lastPkLotInspectionReport;
    }

    /**
     * Returns the eligible 10-feature training records for the currently active dataset mode
     * ({@link ActiveDatasetMode#PKLOT_IMPORTED_DATASET} vs {@link ActiveDatasetMode#USER_REVIEWED_SLOTS}).
     */
    public List<SlotTrainingRecord> getEligibleReviewedRecords() {
        if (activeDatasetMode == ActiveDatasetMode.PKLOT_IMPORTED_DATASET && !pkLotPartitionedPool.isEmpty()) {
            List<SlotTrainingRecord> out = new ArrayList<>();
            for (SlotTrainingRecord rec : pkLotPartitionedPool.values()) {
                if (rec != null
                        && rec.isEligibleForTraining(SlotTrainingRecord.TrainingEligibilityPolicy.HUMAN_VERIFIED_AND_IMPORTED_GROUND_TRUTH)
                        && FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION.equals(rec.featureSchemaVersion())) {
                    out.add(rec);
                }
            }
            return Collections.unmodifiableList(out);
        }

        List<SlotTrainingRecord> out = new ArrayList<>();
        for (SlotTrainingRecord rec : userReviewedTrainingPool.values()) {
            if (rec != null
                    && rec.isEligibleForTraining(SlotTrainingRecord.TrainingEligibilityPolicy.HUMAN_VERIFIED_ONLY)
                    && FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION.equals(rec.featureSchemaVersion())) {
                out.add(rec);
            }
        }
        return Collections.unmodifiableList(out);
    }

    public int getEligibleImportedGroundTruthCount() {
        return (int) getEligibleReviewedRecords().stream()
            .filter(r -> r.isImportedGroundTruth() && !r.isHumanVerified())
            .count();
    }

    public int getEligibleHumanVerifiedCount() {
        return (int) getEligibleReviewedRecords().stream()
            .filter(SlotTrainingRecord::isHumanVerified)
            .count();
    }

    public int getEligibleOccupiedCount() {
        return (int) getEligibleReviewedRecords().stream()
            .filter(r -> r.occupancyLabel() == SlotStatus.OCCUPIED)
            .count();
    }

    public int getEligibleAvailableCount() {
        return (int) getEligibleReviewedRecords().stream()
            .filter(r -> r.occupancyLabel() == SlotStatus.AVAILABLE)
            .count();
    }

    public boolean canTrainDecisionTree() {
        return getTrainingDisabledReason() == null;
    }

    public String getTrainingDisabledReason() {
        if (activeDatasetMode == ActiveDatasetMode.USER_REVIEWED_SLOTS && !geometryDirtySlotIds.isEmpty()) {
            return "Training is disabled because Slot(s) " + geometryDirtySlotIds
                + " have unconfirmed boundary geometry changes. Confirm their labels in Step 3 first.";
        }
        List<SlotTrainingRecord> eligible = getEligibleReviewedRecords();
        int occ = getEligibleOccupiedCount();
        int avail = getEligibleAvailableCount();

        if (activeDatasetMode == ActiveDatasetMode.PKLOT_IMPORTED_DATASET && !pkLotPartitionedPool.isEmpty()) {
            Set<String> trainGrps = new LinkedHashSet<>();
            Set<String> valGrps = new LinkedHashSet<>();
            Set<String> testGrps = new LinkedHashSet<>();
            int trainOcc = 0, trainAvail = 0;
            for (SlotTrainingRecord r : eligible) {
                String p = r.partition();
                if ("TRAIN".equalsIgnoreCase(p)) {
                    trainGrps.add(r.captureGroup());
                    if (r.occupancyLabel() == SlotStatus.OCCUPIED) trainOcc++;
                    else if (r.occupancyLabel() == SlotStatus.AVAILABLE) trainAvail++;
                } else if ("VALIDATION".equalsIgnoreCase(p) || "VAL".equalsIgnoreCase(p)) {
                    valGrps.add(r.captureGroup());
                } else if ("TEST".equalsIgnoreCase(p)) {
                    testGrps.add(r.captureGroup());
                }
            }
            if (trainGrps.isEmpty() || valGrps.isEmpty() || testGrps.isEmpty()) {
                return String.format(Locale.ROOT,
                    "PKLot is not ready for training: independent camera/date groups are TRAIN=%d, VALIDATION=%d, TEST=%d. "
                    + "In Import PKLot, select the TOP-LEVEL extracted PKLot folder—not a single weather/date folder. "
                    + "Include multiple dates and camera folders (for example PUCPR, UFPR04, and UFPR05), then re-import.",
                    trainGrps.size(), valGrps.size(), testGrps.size());
            }
            if (trainOcc == 0 || trainAvail == 0) {
                return String.format(Locale.ROOT,
                    "PKLot TRAIN currently has Occupied=%d and Available=%d. Select a broader top-level PKLot folder with more camera/date groups and re-import so the TRAIN partition contains both classes.",
                    trainOcc, trainAvail);
            }
            return null;
        }

        if (eligible.size() < 4 || occ == 0 || avail == 0) {
            return String.format(Locale.ROOT,
                "Training requires at least 4 confirmed 10-feature records including both Occupied and Available examples "
                + "(currently %d total: %d Occupied, %d Available). Review slots in Step 3 or click 'Import PKLot...'.",
                eligible.size(), occ, avail);
        }
        return null;
    }

    /**
     * Routes the main {@code Train Decision Tree} action through:
     * <ul>
     *   <li>{@link ParkingAnalysisService#trainFromPartitionedPkLotRecords} when a PKLot dataset is active
     *       (preserving saved {@code TRAIN}, {@code VALIDATION}, and frozen {@code TEST} partitions,
     *       selecting hyperparameters on {@code VALIDATION}, and evaluating once on {@code TEST}).</li>
     *   <li>{@link ParkingAnalysisService#trainFromReviewedSlotRecords} when interactive user-reviewed
     *       slot records are active.</li>
     * </ul>
     */
    public ModelEvaluator.EvaluationReport trainTenFeatureDecisionTree(int maxDepth, int minSamplesPerLeaf) {
        String disabledReason = getTrainingDisabledReason();
        if (disabledReason != null) {
            throw new IllegalStateException(disabledReason);
        }
        List<SlotTrainingRecord> records = getEligibleReviewedRecords();

        if (activeDatasetMode == ActiveDatasetMode.PKLOT_IMPORTED_DATASET && !pkLotPartitionedPool.isEmpty()) {
            ParkingAnalysisService.PkLotTrainingSummary summary =
                analysisService.trainFromPartitionedPkLotRecords(
                    records, maxDepth, minSamplesPerLeaf, frozenPkLotTestMetaPath
                );
            this.compatibleTenFeatureModelActive = true;
            this.datasetSchemaVersion = FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION;
            this.activeModelSourceDescription = String.format(Locale.ROOT,
                "Trained 10-Feature CART Model [PKLot: Train=%d, Val=%d (depth=%d,minLeaf=%d), Frozen Test=%d | %d Imported, %d User-Verified]",
                summary.trainCount(), summary.validationCount(),
                summary.selectedMaxDepth(), summary.selectedMinSamplesPerLeaf(),
                summary.testCount(),
                summary.importedGroundTruthCount(), summary.humanVerifiedOrCorrectedCount()
            );
            this.lastEvaluationReport = summary.heldOutTestReport();
            notifyStateChanged();
            return summary.heldOutTestReport();
        }

        ModelEvaluator.EvaluationReport report = analysisService.trainFromReviewedSlotRecords(
            records, maxDepth, minSamplesPerLeaf
        );
        this.compatibleTenFeatureModelActive = true;
        this.datasetSchemaVersion = FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION;
        this.activeModelSourceDescription = String.format(Locale.ROOT,
            "Trained 10-Feature CART Model [User-Reviewed: %d records (%d Occupied, %d Available)]",
            records.size(), getEligibleOccupiedCount(), getEligibleAvailableCount());
        this.lastEvaluationReport = report;
        notifyStateChanged();
        return report;
    }

    public void applyLoadedModelBundle(ModelSerializer.LoadedModelBundle bundle) {
        if (bundle == null || bundle.model() == null) {
            throw new IllegalArgumentException("LoadedModelBundle must not be null.");
        }
        if (!bundle.compatibleTenFeatureModel()) {
            if (bundle.featureCount() == FeatureVector.FEATURE_COUNT) {
                // A newly loaded legacy model invalidates any report from the previous model.
                this.lastEvaluationReport = null;
                // Kept strictly in legacy container, isolated from enhanced training
                this.analysisService.setPredictor(new DecisionTreePredictor(bundle.model()));
                this.compatibleTenFeatureModelActive = false;
                this.datasetSchemaVersion = bundle.schemaVersion();
                this.activeModelSourceDescription = bundle.statusSummary() + " [" + bundle.datasetSource() + "]";
                notifyStateChanged();
                return;
            }
            throw new com.parknchillai.ai.SchemaMismatchException(
                com.parknchillai.imageprocessing.ProcessingPolicy.CANONICAL_SCHEMA_ID, bundle.schemaVersion(),
                com.parknchillai.imageprocessing.ProcessingPolicy.CANONICAL_POLICY_ID, bundle.policyId(),
                "Incompatible model cannot be activated for enhanced prediction: " + bundle.compatibilityMessage());
        }
        // A newly loaded model invalidates any report produced by a different in-memory model.
        this.lastEvaluationReport = null;
        this.analysisService.setPredictor(new DecisionTreePredictor(bundle.model()));
        this.compatibleTenFeatureModelActive = true;
        this.datasetSchemaVersion = bundle.schemaVersion();
        this.activeModelSourceDescription = bundle.statusSummary() + " [" + bundle.datasetSource() + "]";
        notifyStateChanged();
    }

    public boolean isLegacyEightFeatureModelPresent() {
        DecisionTreeModel m = analysisService.getPredictor() != null
            ? analysisService.getPredictor().getModel()
            : null;
        return !compatibleTenFeatureModelActive && m != null && m.getFeatureCount() == FeatureVector.FEATURE_COUNT;
    }

    public String getLegacyModelExplanation() {
        if (isLegacyEightFeatureModelPresent()) {
            return ModelSerializer.LEGACY_EIGHT_FEATURE_MESSAGE;
        }
        return "";
    }

    public boolean isCompatibleTenFeatureModelActive() {
        return compatibleTenFeatureModelActive;
    }

    public String getActiveModelStatusSummary() {
        DecisionTreeModel m = analysisService.getPredictor() != null
            ? analysisService.getPredictor().getModel()
            : null;
        if (m == null) {
            return "Untrained Model";
        }
        if (compatibleTenFeatureModelActive) {
            return String.format(Locale.ROOT, "Ready — 10-Feature CART Decision Tree (Schema %s, maxDepth=%d)",
                FeatureVector.DETECTION_AUGMENTED_SCHEMA_VERSION, m.getMaxDepth());
        }
        return String.format(Locale.ROOT, "Legacy %d-Feature Model (%s) — 10-Feature Upgrade Required",
            m.getFeatureCount(), ModelSerializer.schemaForFeatureCount(m.getFeatureCount()));
    }

    public ParkingAnalysisService getAnalysisService()               { return analysisService; }
    public BufferedImage getCurrentImage()                           { return currentImage; }
    public ImageLoader.LoadedImageMetadata getCurrentImageMetadata() { return currentImageMetadata; }
    public String getCurrentImageSha256()                            { return currentImageSha256; }
    public List<ParkingSlot> getCurrentSlots()                       { return Collections.unmodifiableList(currentSlots); }
    public void setCurrentSlots(List<ParkingSlot> slots)             { this.currentSlots = new ArrayList<>(slots != null ? slots : List.of()); }
    public boolean isSlotLocalizationReliable()                      { return slotLocalizationReliable; }
    public String getSlotLocalizationMessage()                       { return slotLocalizationMessage; }
    public AnalysisSummary getCurrentSummary()                       { return currentSummary; }
    public String getDatasetSourceDescription()                      { return datasetSourceDescription; }
    public void setDatasetSourceDescription(String desc)             { this.datasetSourceDescription = desc; }
    public String getDatasetSchemaVersion()                          { return datasetSchemaVersion; }
    public String getActiveModelSourceDescription()                  { return activeModelSourceDescription; }
    public ModelEvaluator.EvaluationReport getLastEvaluationReport() { return lastEvaluationReport; }
    public void setLastEvaluationReport(ModelEvaluator.EvaluationReport report) {
        this.lastEvaluationReport = report;
        notifyStateChanged();
    }

    public VehicleDetectionService.DetectorStatus getDetectorStatus() {
        if (currentSummary != null && currentSummary.getDetectionBatchResult() != null) {
            return currentSummary.getDetectionBatchResult().status();
        }
        return analysisService.getVehicleDetectionService().getStatus();
    }
}
