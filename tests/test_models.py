import numpy as np
import pytest
from vegcover.models import Box, StitchResult, DetectResult, VegIndexMap, CategoryStat, CoverageStat, AnalysisResult


def test_box_creation():
    box = Box(x1=10, y1=20, x2=30, y2=40, class_name="tree", confidence=0.95)
    assert box.x1 == 10
    assert box.class_name == "tree"
    assert box.confidence == pytest.approx(0.95)


def test_stitch_result_creation():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = StitchResult(panorama=img, success=True, error_msg=None, transform_matrices=[])
    assert result.success is True
    assert result.panorama.shape == (100, 100, 3)


def test_detect_result_creation():
    box = Box(x1=10, y1=20, x2=30, y2=40, class_name="tree", confidence=0.95)
    result = DetectResult(boxes=[box], classes=["tree"], model_name="yolov8n")
    assert len(result.boxes) == 1
    assert result.classes == ["tree"]


def test_veg_index_map_creation():
    mask = np.ones((50, 50), dtype=np.uint8)
    exg = np.zeros((50, 50), dtype=np.float32)
    vari = np.zeros((50, 50), dtype=np.float32)
    result = VegIndexMap(exg_map=exg, vari_map=vari, veg_mask=mask, veg_ratio=0.5)
    assert result.veg_ratio == pytest.approx(0.5)


def test_coverage_stat_creation():
    cat = CategoryStat(name="tree", pixels=1000, ratio=0.25, area_m2=500.0)
    stat = CoverageStat(
        total_pixels=4000,
        categories=[cat],
        total_veg_ratio=0.25,
        non_veg_ratio=0.75,
        gsd=0.1,
    )
    assert stat.total_veg_ratio == pytest.approx(0.25)
    assert stat.categories[0].name == "tree"


def test_analysis_result_creation():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    box = Box(x1=10, y1=20, x2=30, y2=40, class_name="tree", confidence=0.95)
    stitch = StitchResult(panorama=img, success=True, error_msg=None, transform_matrices=[])
    detect = DetectResult(boxes=[box], classes=["tree"], model_name="yolov8n")
    veg_index = VegIndexMap(
        exg_map=np.zeros((100, 100), dtype=np.float32),
        vari_map=np.zeros((100, 100), dtype=np.float32),
        veg_mask=np.ones((100, 100), dtype=np.uint8),
        veg_ratio=0.5,
    )
    cat = CategoryStat(name="tree", pixels=1000, ratio=0.25, area_m2=500.0)
    coverage = CoverageStat(
        total_pixels=4000,
        categories=[cat],
        total_veg_ratio=0.25,
        non_veg_ratio=0.75,
        gsd=0.1,
    )
    result = AnalysisResult(stitch=stitch, detect=detect, veg_index=veg_index, coverage=coverage)
    assert result.stitch.success is True
    assert len(result.detect.boxes) == 1
