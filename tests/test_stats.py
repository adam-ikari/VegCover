import numpy as np
import pytest
from vegcover.stats import CoverageCalculator
from vegcover.models import DetectResult, VegIndexMap, Box


class TestCoverageCalculator:
    def test_init(self):
        calc = CoverageCalculator()
        assert calc is not None

    def test_calculate_with_empty_detection(self):
        """Test with no detections - should still compute from veg mask."""
        calc = CoverageCalculator()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        detect = DetectResult(boxes=[], classes=[], model_name="yolov8n")
        veg_index = VegIndexMap(
            exg_map=np.zeros((100, 100)),
            vari_map=np.zeros((100, 100)),
            veg_mask=np.ones((100, 100), dtype=np.uint8),
            veg_ratio=1.0,
        )

        result = calc.calculate(img, detect, veg_index)

        assert result.total_veg_ratio == pytest.approx(1.0)
        assert result.non_veg_ratio == pytest.approx(0.0)
        assert len(result.categories) == 1  # "unclassified vegetation"

    def test_calculate_with_detections(self):
        """Test with vegetation detections."""
        calc = CoverageCalculator()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Box covering 10x10 area = 100 pixels out of 10000 = 1%
        box = Box(x1=0, y1=0, x2=10, y2=10, class_name="tree", confidence=0.9)
        detect = DetectResult(boxes=[box], classes=["tree"], model_name="yolov8n")
        veg_index = VegIndexMap(
            exg_map=np.zeros((100, 100)),
            vari_map=np.zeros((100, 100)),
            veg_mask=np.zeros((100, 100), dtype=np.uint8),
            veg_ratio=0.0,
        )

        result = calc.calculate(img, detect, veg_index, gsd=0.1)

        assert result.total_pixels == 10000
        assert len(result.categories) >= 1
        tree_stat = next(c for c in result.categories if c.name == "tree")
        assert tree_stat.pixels == 100
        assert tree_stat.ratio == pytest.approx(0.01)
        assert tree_stat.area_m2 == pytest.approx(1.0)  # 100 pixels * 0.01 m2/pixel

    def test_calculate_without_gsd(self):
        """Without GSD, area_m2 should be None."""
        calc = CoverageCalculator()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        box = Box(x1=0, y1=0, x2=10, y2=10, class_name="tree", confidence=0.9)
        detect = DetectResult(boxes=[box], classes=["tree"], model_name="yolov8n")
        veg_index = VegIndexMap(
            exg_map=np.zeros((100, 100)),
            vari_map=np.zeros((100, 100)),
            veg_mask=np.zeros((100, 100), dtype=np.uint8),
            veg_ratio=0.0,
        )

        result = calc.calculate(img, detect, veg_index, gsd=None)

        tree_stat = next(c for c in result.categories if c.name == "tree")
        assert tree_stat.area_m2 is None
