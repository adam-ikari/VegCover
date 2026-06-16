"""VegCover - PyQt6 Desktop Application.

Launch with: python main.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vegcover.detector import VegetationDetector
from vegcover.exporter import ResultExporter
from vegcover.models import AnalysisResult
from vegcover.pipeline import AnalysisPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    """Background worker for running the analysis pipeline."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(AnalysisResult)
    error = pyqtSignal(str)

    def __init__(self, images, confidence, iou, gsd):
        super().__init__()
        self.images = images
        self.confidence = confidence
        self.iou = iou
        self.gsd = gsd

    def run(self):
        try:
            self.progress.emit("Stitching photos...")
            detector = VegetationDetector(
                confidence_threshold=self.confidence,
                iou_threshold=self.iou,
            )
            pipeline = AnalysisPipeline(detector=detector)

            self.progress.emit("Detecting vegetation...")
            result = pipeline.run(self.images, gsd=self.gsd)

            self.progress.emit("Exporting results...")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class VegCoverApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VegCover - Aerial Vegetation Analysis")
        self.setMinimumSize(1200, 800)

        self.image_paths = []
        self.current_result = None
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        """Build the user interface."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Splitter for left/right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: Inputs and parameters
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Photo selection
        left_layout.addWidget(QLabel("<h2>📁 Photos</h2>"))
        self.photo_label = QLabel("No photos selected")
        self.photo_label.setWordWrap(True)
        left_layout.addWidget(self.photo_label)

        select_btn = QPushButton("Select Photos...")
        select_btn.clicked.connect(self._select_photos)
        left_layout.addWidget(select_btn)

        # Parameters
        left_layout.addWidget(QLabel("<h2>⚙️ Parameters</h2>"))

        # Confidence threshold
        left_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(25)
        self.confidence_label = QLabel("0.25")
        self.confidence_slider.valueChanged.connect(self._update_confidence_label)
        left_layout.addWidget(self.confidence_label)
        left_layout.addWidget(self.confidence_slider)

        # IoU threshold
        left_layout.addWidget(QLabel("IoU Threshold:"))
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setMinimum(0)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(45)
        self.iou_label = QLabel("0.45")
        self.iou_slider.valueChanged.connect(self._update_iou_label)
        left_layout.addWidget(self.iou_label)
        left_layout.addWidget(self.iou_slider)

        # GSD
        left_layout.addWidget(QLabel("GSD (m/pixel, optional):"))
        self.gsd_input = QLineEdit()
        self.gsd_input.setPlaceholderText("Leave empty for pixel ratios only")
        left_layout.addWidget(self.gsd_input)

        # Analyze button
        self.analyze_btn = QPushButton("▶ Start Analysis")
        self.analyze_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        self.analyze_btn.clicked.connect(self._start_analysis)
        left_layout.addWidget(self.analyze_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # Right panel: Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.tabs = QTabWidget()

        # Tab 1: Annotated Panorama
        self.annotated_scroll = QScrollArea()
        self.annotated_scroll.setWidgetResizable(True)
        self.annotated_label = QLabel("Analysis results will appear here")
        self.annotated_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.annotated_scroll.setWidget(self.annotated_label)
        self.tabs.addTab(self.annotated_scroll, "Annotated Panorama")

        # Tab 2: Vegetation Heatmap
        self.heatmap_scroll = QScrollArea()
        self.heatmap_scroll.setWidgetResizable(True)
        self.heatmap_label = QLabel("Analysis results will appear here")
        self.heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heatmap_scroll.setWidget(self.heatmap_label)
        self.tabs.addTab(self.heatmap_scroll, "Vegetation Heatmap")

        # Tab 3: Coverage Charts
        self.charts_scroll = QScrollArea()
        self.charts_scroll.setWidgetResizable(True)
        self.charts_label = QLabel("Analysis results will appear here")
        self.charts_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.charts_scroll.setWidget(self.charts_label)
        self.tabs.addTab(self.charts_scroll, "Coverage Charts")

        # Tab 4: Export Data
        export_widget = QWidget()
        export_layout = QVBoxLayout(export_widget)
        self.export_info = QLabel("Export analysis results:")
        export_layout.addWidget(self.export_info)

        self.export_csv_btn = QPushButton("Export CSV Report")
        self.export_csv_btn.clicked.connect(lambda: self._export_file("csv"))
        export_layout.addWidget(self.export_csv_btn)

        self.export_json_btn = QPushButton("Export JSON Report")
        self.export_json_btn.clicked.connect(lambda: self._export_file("json"))
        export_layout.addWidget(self.export_json_btn)

        self.export_img_btn = QPushButton("Export Images")
        self.export_img_btn.clicked.connect(lambda: self._export_file("images"))
        export_layout.addWidget(self.export_img_btn)

        export_layout.addStretch()
        self.tabs.addTab(export_widget, "Export Data")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([300, 900])

    def _update_confidence_label(self):
        value = self.confidence_slider.value() / 100
        self.confidence_label.setText(f"{value:.2f}")

    def _update_iou_label(self):
        value = self.iou_slider.value() / 100
        self.iou_label.setText(f"{value:.2f}")

    def _select_photos(self):
        """Open file dialog to select aerial photos."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Aerial Photos",
            "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff)",
        )
        if file_paths:
            self.image_paths = file_paths
            self.photo_label.setText(f"Selected {len(file_paths)} photos:\n" + "\n".join(Path(p).name for p in file_paths[:5]))
            if len(file_paths) > 5:
                self.photo_label.setText(self.photo_label.text() + f"\n... and {len(file_paths) - 5} more")

    def _start_analysis(self):
        """Start the analysis pipeline."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Photos", "Please select at least one photo.")
            return

        # Load images
        images = []
        for path in self.image_paths:
            img = Image.open(path).convert("RGB")
            images.append(np.array(img))

        # Get parameters
        confidence = self.confidence_slider.value() / 100
        iou = self.iou_slider.value() / 100
        gsd_text = self.gsd_input.text().strip()
        gsd = float(gsd_text) if gsd_text else None

        # Start worker thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.analyze_btn.setEnabled(False)
        self.status_label.setText("Analysis in progress...")

        self.worker = AnalysisWorker(images, confidence, iou, gsd)
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._analysis_finished)
        self.worker.error.connect(self._analysis_error)
        self.worker.start()

    def _update_progress(self, message):
        self.status_label.setText(message)

    def _analysis_finished(self, result: AnalysisResult):
        """Handle analysis completion."""
        self.current_result = result
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.status_label.setText(f"Analysis complete! Total vegetation coverage: {result.coverage.total_veg_ratio:.1%}")

        # Export results
        exporter = ResultExporter(output_dir="output")
        paths = exporter.export(result)

        # Display results
        self._display_image(paths.get("annotated_image"), self.annotated_label)
        self._display_image(paths.get("heatmap"), self.heatmap_label)
        if "charts" in paths:
            self._display_image(paths["charts"], self.charts_label)
        else:
            self.charts_label.setText("No charts available (no vegetation detected)")

    def _analysis_error(self, error_msg):
        """Handle analysis error."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Analysis Error", error_msg)

    def _display_image(self, image_path: str, label: QLabel):
        """Display an image in a QLabel."""
        if image_path and Path(image_path).exists():
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                # Scale to fit while maintaining aspect ratio
                scaled = pixmap.scaled(
                    800, 600,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                label.setPixmap(scaled)
            else:
                label.setText(f"Failed to load image: {image_path}")
        else:
            label.setText("Image not available")

    def _export_file(self, format_type: str):
        """Export results to a file."""
        if not self.current_result:
            QMessageBox.warning(self, "No Results", "Please run analysis first.")
            return

        exporter = ResultExporter(output_dir="output")
        paths = exporter.export(self.current_result)

        if format_type == "csv":
            path = paths.get("stats_csv")
            if path:
                QMessageBox.information(self, "CSV Exported", f"CSV report saved to:\n{path}")
        elif format_type == "json":
            path = paths.get("stats_json")
            if path:
                QMessageBox.information(self, "JSON Exported", f"JSON report saved to:\n{path}")
        elif format_type == "images":
            msg = "Images exported to:\n"
            for key in ["annotated_image", "heatmap"]:
                if key in paths:
                    msg += f"- {paths[key]}\n"
            QMessageBox.information(self, "Images Exported", msg)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VegCover")
    app.setApplicationDisplayName("VegCover - Aerial Vegetation Analysis")

    window = VegCoverApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
