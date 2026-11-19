Fix for ParkNChill test compilation error.

Replace:
src/test/java/com/parknchillai/detection/VehicleDetectionAndOverlapTest.java

with the file in this archive, preserving the same path.

The fix changes image.setRGB(0, 0, Color.RED) to image.setRGB(0, 0, Color.RED.getRGB()), because BufferedImage.setRGB expects an int RGB value, not a java.awt.Color object.

Then run: mvn clean test
