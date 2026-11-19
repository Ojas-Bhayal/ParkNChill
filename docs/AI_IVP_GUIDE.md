# Showing AI, IVP, and the Decision Tree in ParkNChill

Use this guide for your project demonstration.

## A. Vehicle AI (ONNX/YOLOX)

1. Load a real parking photograph.
2. Click **Detect and Analyse**.
3. Show the vehicle bounding boxes and the detected vehicle count.
4. Explain that the pretrained ONNX detector locates vehicles; it does not learn parking-slot labels in this project.
5. If `Detector: READY` appears but the count is zero, the model loaded but detection did not succeed on that image. Do not claim successful detection until boxes align with visible vehicles.

## B. Image and Video Processing (IVP)

1. Click **View IVP Stages**. This is available even before any parking slot is selected.
2. With no slot selected, the app computes a downsized full-image overview to show the processing steps; the numeric values are not suitable as slot-level training features.
3. For a real slot-level demonstration, first load the correct PKLot `.xml` using **Load Slot XML** (or use the image/XML pair), then select a slot.
4. Click **View IVP Stages** again. The source region is cropped/rectified from that slot and passed through the actual `FeatureExtractor`.
5. Show the processed previews and eight computed IVP features:
   - Mean intensity
   - Foreground ratio
   - Dark-pixel ratio
   - Edge density
   - Object area
   - Mean saturation
   - Saturation ratio
   - Largest blob ratio

The stages are actual outputs from the Java `FeatureExtractor`, not decorative pictures. The two extra v2.3 values—vehicle-overlap ratio and vehicle-match flag—are computed from vehicle detections against the slot boundary.

## C. Decision Tree AI

1. Import PKLot or prepare a user-reviewed dataset.
2. Confirm valid records and examples for both OCCUPIED and AVAILABLE.
3. Ensure each record contains all ten features and a valid label.
4. Train the CART Decision Tree. If the button is disabled, read the reason displayed in the app and correct the missing prerequisite.
5. Show the trained tree/rule path and test metrics.
6. For a credible demonstration, analyse an independent image that was not used in training.

Do not conflate these outputs:
- **Vehicle detector:** detected object boxes.
- **PKLot/user verified label:** source ground truth.
- **IVP:** intermediate processed images and eight image features.
- **Decision Tree:** a prediction produced by a trained model from its ten features.

## Known limitation

The full-scene IVP preview demonstrates the pixel-processing algorithm independently of slot detection. It does not by itself establish parking-space geometry. For correct slot-specific features, reliable slot boundaries are required.
