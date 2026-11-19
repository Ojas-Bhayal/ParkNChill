package com.parknchillai.ui;

import com.parknchillai.app.ParkingAnalysisService;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;

/**
 * Main JavaFX Entry Point for ParkNChill v2.3.
 */
public class MainApp extends Application {

    private ParkingAnalysisService analysisService;
    private MainWindow root;

    @Override
    public void start(Stage primaryStage) {
        this.analysisService = ParkingAnalysisService.createWithDefaultModel();

        this.root = new MainWindow(analysisService, false);
        Scene scene = new Scene(root, 1366, 768);

        var cssUrl = getClass().getResource("/styles/style.css");
        if (cssUrl != null) {
            scene.getStylesheets().add(cssUrl.toExternalForm());
        }

        primaryStage.setTitle("ParkNChill — Smart Parking Occupancy Detection");
        primaryStage.setMinWidth(1024);
        primaryStage.setMinHeight(680);
        primaryStage.setScene(scene);

        primaryStage.setOnCloseRequest(e -> shutdownResources());

        primaryStage.show();
    }

    @Override
    public void stop() {
        shutdownResources();
    }

    private void shutdownResources() {
        if (root != null) {
            try {
                root.getModelDetailsView().cancelActiveImport();
            } catch (Throwable ignored) {
            }
        }
        if (analysisService != null && analysisService.getVehicleDetectionService() instanceof AutoCloseable ac) {
            try {
                ac.close();
            } catch (Throwable ignored) {
            }
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
