"""Export module for charts and data files."""

import csv
import json
import logging
import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from vegcover.models import AnalysisResult, DetectResult

logger = logging.getLogger(__name__)

# Use non-interactive backend for headless environments
matplotlib.use("Agg")


class ResultExporter:
    """Exports analysis results to images, charts, and data files."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, result: AnalysisResult) -> dict[str, str]:
        """Export all analysis results.

        Returns:
            Dictionary mapping output type to file path.
        """
        paths = {}

        # Export annotated image
        annotated = self._create_annotated_image(result.stitch.panorama, result.detect)
        annotated_path = self.output_dir / "annotated_panorama.png"
        cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        paths["annotated_image"] = str(annotated_path)

        # Export heatmap
        heatmap = self._create_heatmap(result.veg_index.veg_mask)
        heatmap_path = self.output_dir / "vegetation_heatmap.png"
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        paths["heatmap"] = str(heatmap_path)

        # Export charts
        if result.coverage.categories:
            chart_path = self.output_dir / "coverage_charts.png"
            self._create_coverage_charts(result.coverage, chart_path)
            paths["charts"] = str(chart_path)

        # Export CSV
        csv_path = self.output_dir / "coverage_stats.csv"
        self._export_csv(result.coverage, csv_path)
        paths["stats_csv"] = str(csv_path)

        # Export JSON
        json_path = self.output_dir / "coverage_stats.json"
        self._export_json(result, json_path)
        paths["stats_json"] = str(json_path)

        return paths

    def _create_annotated_image(self, image: np.ndarray, detect: DetectResult) -> np.ndarray:
        """Draw bounding boxes on the image."""
        annotated = image.copy()
        colors = {"tree": (0, 255, 0), "plant": (0, 200, 0), "unclassified vegetation": (0, 150, 0)}

        for box in detect.boxes:
            color = colors.get(box.class_name, (0, 255, 0))
            cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, 2)
            label = f"{box.class_name}: {box.confidence:.2f}"
            cv2.putText(annotated, label, (box.x1, box.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return annotated

    def _create_heatmap(self, veg_mask: np.ndarray) -> np.ndarray:
        """Create a color heatmap from vegetation mask."""
        heatmap = cv2.applyColorMap(veg_mask * 255, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    def _create_coverage_charts(self, coverage, output_path: Path):
        """Create pie and bar charts for coverage statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart
        labels = [c.name for c in coverage.categories]
        sizes = [c.ratio for c in coverage.categories]
        if coverage.non_veg_ratio > 0:
            labels.append("non-vegetation")
            sizes.append(coverage.non_veg_ratio)
        axes[0].pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        axes[0].set_title("Vegetation Coverage")

        # Bar chart
        names = [c.name for c in coverage.categories]
        values = [c.pixels for c in coverage.categories]
        axes[1].bar(names, values)
        axes[1].set_title("Pixel Count by Category")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    def _export_csv(self, coverage, output_path: Path):
        """Export statistics to CSV."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Pixels", "Ratio", "Area (m2)"])
            for cat in coverage.categories:
                writer.writerow([cat.name, cat.pixels, f"{cat.ratio:.4f}", cat.area_m2 or "N/A"])
            writer.writerow(["Total Vegetation", "", f"{coverage.total_veg_ratio:.4f}", ""])
            writer.writerow(["Non-Vegetation", "", f"{coverage.non_veg_ratio:.4f}", ""])

    def _export_json(self, result: AnalysisResult, output_path: Path):
        """Export complete results to JSON."""
        data = {
            "stitch": {"success": result.stitch.success, "error_msg": result.stitch.error_msg},
            "detection": {
                "model": result.detect.model_name,
                "classes": result.detect.classes,
                "boxes": [
                    {
                        "x1": b.x1,
                        "y1": b.y1,
                        "x2": b.x2,
                        "y2": b.y2,
                        "class": b.class_name,
                        "confidence": b.confidence,
                    }
                    for b in result.detect.boxes
                ],
            },
            "vegetation_index": {"veg_ratio": result.veg_index.veg_ratio},
            "coverage": {
                "total_pixels": result.coverage.total_pixels,
                "total_veg_ratio": result.coverage.total_veg_ratio,
                "non_veg_ratio": result.coverage.non_veg_ratio,
                "gsd": result.coverage.gsd,
                "categories": [
                    {
                        "name": c.name,
                        "pixels": c.pixels,
                        "ratio": c.ratio,
                        "area_m2": c.area_m2,
                    }
                    for c in result.coverage.categories
                ],
            },
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
