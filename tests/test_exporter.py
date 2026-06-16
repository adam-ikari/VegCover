import os
import tempfile

import numpy as np
import pytest

from vegcover.exporter import ResultExporter
from vegcover.models import AnalysisResult, Box, CategoryStat, CoverageStat, DetectResult, StitchResult, VegIndexMap


class TestResultExporter:
    def test_init(self):
        exporter = ResultExporter(output_dir="/tmp/test_output")
        assert str(exporter.output_dir) == "/tmp/test_output"

    def test_export_creates_files(self):
        """export() should create output directory and files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ResultExporter(output_dir=tmpdir)

            # Create minimal analysis result
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            stitch = StitchResult(panorama=img, success=True, error_msg=None, transform_matrices=[])
            detect = DetectResult(boxes=[], classes=[], model_name="yolov8n")
            veg_index = VegIndexMap(
                exg_map=np.zeros((50, 50)),
                vari_map=np.zeros((50, 50)),
                veg_mask=np.zeros((50, 50), dtype=np.uint8),
                veg_ratio=0.0,
            )
            coverage = CoverageStat(
                total_pixels=2500,
                categories=[],
                total_veg_ratio=0.0,
                non_veg_ratio=1.0,
                gsd=None,
            )
            result = AnalysisResult(stitch=stitch, detect=detect, veg_index=veg_index, coverage=coverage)

            paths = exporter.export(result)

            assert "annotated_image" in paths
            assert "heatmap" in paths
            assert "stats_csv" in paths
            assert "stats_json" in paths
            assert os.path.exists(paths["annotated_image"])
            assert os.path.exists(paths["stats_csv"])
            assert os.path.exists(paths["stats_json"])

    def test_create_annotated_image(self):
        """Should create an image with bounding boxes drawn."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ResultExporter(output_dir=tmpdir)

            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            box = Box(x1=10, y1=10, x2=30, y2=30, class_name="tree", confidence=0.9)
            detect = DetectResult(boxes=[box], classes=["tree"], model_name="yolov8n")

            annotated = exporter._create_annotated_image(img, detect)
            assert annotated is not None
            assert annotated.shape == img.shape

    def test_create_heatmap(self):
        """Should create a heatmap from vegetation mask."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ResultExporter(output_dir=tmpdir)

            veg_mask = np.zeros((50, 50), dtype=np.uint8)
            veg_mask[:25, :] = 255

            heatmap = exporter._create_heatmap(veg_mask)
            assert heatmap is not None
            assert heatmap.shape[:2] == (50, 50)
