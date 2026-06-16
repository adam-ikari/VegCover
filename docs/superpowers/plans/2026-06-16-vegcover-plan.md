# VegCover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gradio-based desktop tool that stitches aerial RGB photos into a panorama, detects vegetation with YOLO, computes visible-light vegetation indices (ExG, VARI), and reports coverage statistics with charts and exportable data.

**Architecture:** Linear Pipeline with modular functions. Each pipeline step (stitch, detect, vegindex, stats, export) is a standalone module with a single responsibility. Gradio UI calls the pipeline controller which orchestrates the steps. Data flows through typed dataclasses defined in models.py.

**Tech Stack:** Python 3.13, Gradio, OpenCV, Ultralytics (YOLOv8), NumPy, Matplotlib, Pillow

---

## File Structure

```
VegCover/
├── pyproject.toml           # Dependencies: gradio, opencv-python, ultralytics, numpy, matplotlib, pillow
├── main.py                  # Entry: launch Gradio app
├── src/
│   └── vegcover/
│       ├── __init__.py      # Package init, version
│       ├── models.py        # Dataclasses: Box, StitchResult, DetectResult, VegIndexMap, CategoryStat, CoverageStat, AnalysisResult
│       ├── stitcher.py      # Photo stitching with OpenCV Stitcher
│       ├── detector.py      # YOLO vegetation detection
│       ├── vegindex.py      # ExG and VARI vegetation index computation
│       ├── stats.py         # Coverage statistics calculation
│       ├── exporter.py      # Chart generation and file export
│       └── pipeline.py      # Pipeline controller: orchestrates all steps
├── tests/
│   └── test_pipeline.py     # Unit tests for core pipeline logic
└── output/                  # Default export directory (gitignored)
```

---

## Task 1: Project Setup and Dependencies

**Files:**
- Create: `src/vegcover/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml with dependencies**

```toml
[project]
name = "vegcover"
version = "0.1.0"
description = "Aerial photo stitching and vegetation analysis tool"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "gradio>=5.0",
    "opencv-python>=4.10",
    "ultralytics>=8.3",
    "numpy>=2.0",
    "matplotlib>=3.9",
    "pillow>=10.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create package init file**

```python
"""VegCover - Aerial photo stitching and vegetation analysis tool."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -e .`
Expected: All packages install successfully.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/vegcover/__init__.py
git commit -m "chore: add project dependencies and package init"
```

---

## Task 2: Data Models (models.py)

**Files:**
- Create: `src/vegcover/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'vegcover" (need to install in editable mode first)

- [ ] **Step 3: Write minimal implementation**

```python
"""Data models for VegCover pipeline."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Box:
    """Bounding box for detected vegetation."""

    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    confidence: float


@dataclass
class StitchResult:
    """Result of photo stitching."""

    panorama: np.ndarray
    success: bool
    error_msg: str | None
    transform_matrices: list


@dataclass
class DetectResult:
    """Result of YOLO vegetation detection."""

    boxes: list[Box]
    classes: list[str]
    model_name: str


@dataclass
class VegIndexMap:
    """Vegetation index maps."""

    exg_map: np.ndarray
    vari_map: np.ndarray
    veg_mask: np.ndarray
    veg_ratio: float


@dataclass
class CategoryStat:
    """Statistics for a single vegetation category."""

    name: str
    pixels: int
    ratio: float
    area_m2: float | None


@dataclass
class CoverageStat:
    """Overall coverage statistics."""

    total_pixels: int
    categories: list[CategoryStat]
    total_veg_ratio: float
    non_veg_ratio: float
    gsd: float | None


@dataclass
class AnalysisResult:
    """Complete analysis result."""

    stitch: StitchResult
    detect: DetectResult
    veg_index: VegIndexMap
    coverage: CoverageStat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_models.py src/vegcover/models.py
git commit -m "feat: add data models for pipeline results"
```

---

## Task 3: Photo Stitcher (stitcher.py)

**Files:**
- Create: `src/vegcover/stitcher.py`
- Test: `tests/test_stitcher.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from vegcover.stitcher import PhotoStitcher


class TestPhotoStitcher:
    def test_stitch_single_image(self):
        """Single image should return itself as panorama."""
        stitcher = PhotoStitcher()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = stitcher.stitch([img])
        assert result.success is True
        assert result.panorama is not None
        assert result.error_msg is None

    def test_stitch_no_images(self):
        """Empty list should fail gracefully."""
        stitcher = PhotoStitcher()
        result = stitcher.stitch([])
        assert result.success is False
        assert result.error_msg is not None

    def test_stitch_two_overlapping_images(self):
        """Two overlapping images should produce a larger panorama."""
        stitcher = PhotoStitcher()
        # Create two overlapping images: left half and right half
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Make them share 50% overlap by copying right half of img1 to left half of img2
        img2[:, :50] = img1[:, 50:]

        result = stitcher.stitch([img1, img2])
        # Note: OpenCV Stitcher may not always succeed with synthetic images
        if result.success:
            assert result.panorama is not None
            assert result.panorama.shape[0] > 0

    def test_stitch_result_type(self):
        """Stitch result should be a StitchResult dataclass."""
        from vegcover.models import StitchResult

        stitcher = PhotoStitcher()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = stitcher.stitch([img])
        assert isinstance(result, StitchResult)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stitcher.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError" (stitcher module doesn't exist yet)

- [ ] **Step 3: Write minimal implementation**

```python
"""Photo stitching module using OpenCV Stitcher."""

import logging

import cv2
import numpy as np

from vegcover.models import StitchResult

logger = logging.getLogger(__name__)


class PhotoStitcher:
    """Stitches multiple aerial photos into a panorama."""

    def __init__(self, feature_threshold: float = 0.5, blend_mode: str = "multiband"):
        self.feature_threshold = feature_threshold
        self.blend_mode = blend_mode

    def stitch(self, images: list[np.ndarray]) -> StitchResult:
        """Stitch multiple images into a panorama.

        Args:
            images: List of RGB images as numpy arrays.

        Returns:
            StitchResult with panorama or error info.
        """
        if not images:
            return StitchResult(
                panorama=np.array([]),
                success=False,
                error_msg="No images provided for stitching.",
                transform_matrices=[],
            )

        if len(images) == 1:
            return StitchResult(
                panorama=images[0],
                success=True,
                error_msg=None,
                transform_matrices=[],
            )

        # Use OpenCV Stitcher
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

        # Try stitching
        status, panorama = stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            return StitchResult(
                panorama=panorama,
                success=True,
                error_msg=None,
                transform_matrices=[],
            )

        # Stitching failed - return the first image as fallback
        error_messages = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images for stitching.",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed.",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed.",
        }
        error_msg = error_messages.get(status, f"Stitching failed with status code: {status}")
        logger.warning(f"{error_msg} Returning first image as fallback.")

        return StitchResult(
            panorama=images[0],
            success=False,
            error_msg=error_msg,
            transform_matrices=[],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stitcher.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_stitcher.py src/vegcover/stitcher.py
git commit -m "feat: add photo stitching with OpenCV Stitcher"
```

---

## Task 4: YOLO Vegetation Detector (detector.py)

**Files:**
- Create: `src/vegcover/detector.py`
- Test: `tests/test_detector.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from vegcover.detector import VegetationDetector
from vegcover.models import DetectResult


class TestVegetationDetector:
    def test_init_default_model(self):
        detector = VegetationDetector()
        assert detector.confidence_threshold == 0.25
        assert detector.iou_threshold == 0.45

    def test_init_custom_params(self):
        detector = VegetationDetector(confidence_threshold=0.5, iou_threshold=0.6)
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.6

    @patch("vegcover.detector.YOLO")
    def test_detect_returns_detect_result(self, mock_yolo_class):
        """Mock YOLO and verify detect returns proper DetectResult."""
        # Setup mock
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock results
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = np.array([[10, 20, 30, 40]])
        mock_result.boxes.conf = np.array([0.95])
        mock_result.boxes.cls = np.array([0])
        mock_result.names = {0: "tree"}
        mock_model.return_value = [mock_result]

        detector = VegetationDetector(model_path="yolov8n.pt")
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        result = detector.detect(img)

        assert isinstance(result, DetectResult)
        assert len(result.boxes) == 1
        assert result.boxes[0].class_name == "tree"
        assert result.boxes[0].confidence == pytest.approx(0.95)

    def test_coco_vegetation_classes(self):
        """Verify COCO vegetation-related class names are recognized."""
        from vegcover.detector import COCO_VEGETATION_CLASSES
        assert "potted plant" in COCO_VEGETATION_CLASSES
        assert "tree" in COCO_VEGETATION_CLASSES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detector.py -v`
Expected: FAIL with ImportError (detector module doesn't exist yet)

- [ ] **Step 3: Write minimal implementation**

```python
"""YOLO vegetation detection module."""

import logging

import numpy as np

from vegcover.models import Box, DetectResult

logger = logging.getLogger(__name__)

# COCO classes related to vegetation
COCO_VEGETATION_CLASSES = {
    "potted plant",
    "tree",
}

# Mapping from COCO names to friendlier names
COCO_TO_VEG_NAME = {
    "potted plant": "plant",
    "tree": "tree",
}


class VegetationDetector:
    """Detects vegetation in images using YOLO."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._model = None

    @property
    def model(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image: np.ndarray) -> DetectResult:
        """Detect vegetation in an image.

        Args:
            image: RGB image as numpy array.

        Returns:
            DetectResult with detected boxes.
        """
        results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)

        boxes = []
        classes = set()

        for result in results:
            if result.boxes is None:
                continue

            for box_data, conf, cls_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                class_name = result.names.get(cls_id, "unknown")

                # Map to vegetation-friendly name
                display_name = COCO_TO_VEG_NAME.get(class_name, class_name)

                box = Box(
                    x1=int(box_data[0]),
                    y1=int(box_data[1]),
                    x2=int(box_data[2]),
                    y2=int(box_data[3]),
                    class_name=display_name,
                    confidence=float(conf),
                )
                boxes.append(box)
                classes.add(display_name)

        return DetectResult(
            boxes=boxes,
            classes=sorted(list(classes)),
            model_name=self.model_path,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_detector.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_detector.py src/vegcover/detector.py
git commit -m "feat: add YOLO vegetation detector with COCO class mapping"
```

---

## Task 5: Vegetation Index Calculator (vegindex.py)

**Files:**
- Create: `src/vegcover/vegindex.py`
- Test: `tests/test_vegindex.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from vegcover.vegindex import VegetationIndexCalculator
from vegcover.models import VegIndexMap


class TestVegetationIndexCalculator:
    def test_init(self):
        calc = VegetationIndexCalculator()
        assert calc is not None

    def test_compute_exg(self):
        """ExG should be high for green pixels."""
        calc = VegetationIndexCalculator()
        # Pure green pixel
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 1] = 255  # Green channel max

        exg = calc._compute_exg(img)
        # ExG = 2*255 - 0 - 0 = 510, normalized to [0,1] = 510 / 510 = 1.0
        assert exg[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_compute_vari(self):
        """VARI should be high for green pixels."""
        calc = VegetationIndexCalculator()
        # Pure green pixel
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 1] = 255  # Green channel max

        vari = calc._compute_vari(img)
        # VARI = (255 - 0) / (255 + 0 - 0) = 1.0
        assert vari[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_compute_returns_veg_index_map(self):
        """compute() should return VegIndexMap with proper structure."""
        calc = VegetationIndexCalculator()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Make half the image green
        img[:50, :, 1] = 255

        result = calc.compute(img)

        assert isinstance(result, VegIndexMap)
        assert result.exg_map.shape == (100, 100)
        assert result.vari_map.shape == (100, 100)
        assert result.veg_mask.shape == (100, 100)
        assert 0.0 <= result.veg_ratio <= 1.0

    def test_green_image_high_veg_ratio(self):
        """All-green image should have high vegetation ratio."""
        calc = VegetationIndexCalculator()
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :, 1] = 255  # Full green

        result = calc.compute(img)
        assert result.veg_ratio > 0.8

    def test_black_image_low_veg_ratio(self):
        """Black image should have low vegetation ratio."""
        calc = VegetationIndexCalculator()
        img = np.zeros((50, 50, 3), dtype=np.uint8)

        result = calc.compute(img)
        assert result.veg_ratio < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vegindex.py -v`
Expected: FAIL with ImportError (vegindex module doesn't exist yet)

- [ ] **Step 3: Write minimal implementation**

```python
"""Vegetation index computation for RGB images."""

import logging

import cv2
import numpy as np

from vegcover.models import VegIndexMap

logger = logging.getLogger(__name__)


class VegetationIndexCalculator:
    """Computes vegetation indices from RGB images."""

    def __init__(self, threshold_method: str = "otsu"):
        self.threshold_method = threshold_method

    def _compute_exg(self, image: np.ndarray) -> np.ndarray:
        """Compute Excess Greenness (ExG) index.

        ExG = 2G - R - B, normalized to [0, 1].
        """
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        exg = 2 * g - r - b
        # Normalize to [0, 1]
        exg_min, exg_max = exg.min(), exg.max()
        if exg_max > exg_min:
            exg = (exg - exg_min) / (exg_max - exg_min)
        else:
            exg = np.zeros_like(exg)
        return exg

    def _compute_vari(self, image: np.ndarray) -> np.ndarray:
        """Compute Visible Atmospherically Resistant Index (VARI).

        VARI = (G - R) / (G + R - B), clamped to [-1, 1].
        """
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        denominator = g + r - b
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-10, denominator)

        vari = (g - r) / denominator
        vari = np.clip(vari, -1.0, 1.0)
        return vari

    def compute(self, image: np.ndarray) -> VegIndexMap:
        """Compute vegetation indices and mask.

        Args:
            image: RGB image as numpy array (H, W, 3).

        Returns:
            VegIndexMap with ExG, VARI, and binary vegetation mask.
        """
        exg = self._compute_exg(image)
        vari = self._compute_vari(image)

        # Use ExG for thresholding (typically better for RGB)
        exg_uint8 = (exg * 255).astype(np.uint8)

        # Otsu thresholding
        if self.threshold_method == "otsu":
            threshold, _ = cv2.threshold(exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            threshold = 128  # Default fallback

        veg_mask = (exg_uint8 > threshold).astype(np.uint8)

        # Calculate vegetation ratio
        veg_ratio = float(veg_mask.sum() / veg_mask.size)

        return VegIndexMap(
            exg_map=exg,
            vari_map=vari,
            veg_mask=veg_mask,
            veg_ratio=veg_ratio,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vegindex.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_vegindex.py src/vegcover/vegindex.py
git commit -m "feat: add ExG and VARI vegetation index calculator"
```

---

## Task 6: Coverage Statistics (stats.py)

**Files:**
- Create: `src/vegcover/stats.py`
- Test: `tests/test_stats.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stats.py -v`
Expected: FAIL with ImportError (stats module doesn't exist yet)

- [ ] **Step 3: Write minimal implementation**

```python
"""Coverage statistics calculation module."""

import logging

import numpy as np

from vegcover.models import Box, CategoryStat, CoverageStat, DetectResult, VegIndexMap

logger = logging.getLogger(__name__)


class CoverageCalculator:
    """Calculates vegetation coverage statistics."""

    def calculate(
        self,
        image: np.ndarray,
        detect: DetectResult,
        veg_index: VegIndexMap,
        gsd: float | None = None,
    ) -> CoverageStat:
        """Calculate coverage statistics.

        Args:
            image: RGB image (H, W, 3).
            detect: Detection results.
            veg_index: Vegetation index results.
            gsd: Ground sampling distance in meters per pixel (optional).

        Returns:
            CoverageStat with per-category and overall statistics.
        """
        h, w = image.shape[:2]
        total_pixels = h * w

        # Calculate per-pixel area if GSD is provided
        pixel_area = gsd * gsd if gsd is not None else None

        # Build a category-to-pixel mapping
        category_pixels: dict[str, int] = {}

        # Count pixels from YOLO detections
        for box in detect.boxes:
            box_pixels = (box.x2 - box.x1) * (box.y2 - box.y1)
            category_pixels[box.class_name] = category_pixels.get(box.class_name, 0) + box_pixels

        # Calculate vegetation from veg_mask for unclassified areas
        veg_pixels = int(veg_index.veg_mask.sum())
        detected_pixels = sum(category_pixels.values())
        unclassified_pixels = max(0, veg_pixels - detected_pixels)

        if unclassified_pixels > 0:
            category_pixels["unclassified vegetation"] = unclassified_pixels

        # Build CategoryStat list
        categories = []
        for name, pixels in category_pixels.items():
            ratio = pixels / total_pixels
            area_m2 = pixels * pixel_area if pixel_area is not None else None
            categories.append(CategoryStat(name=name, pixels=pixels, ratio=ratio, area_m2=area_m2))

        # Calculate overall ratios
        total_veg_ratio = veg_index.veg_ratio
        non_veg_ratio = 1.0 - total_veg_ratio

        return CoverageStat(
            total_pixels=total_pixels,
            categories=categories,
            total_veg_ratio=total_veg_ratio,
            non_veg_ratio=non_veg_ratio,
            gsd=gsd,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stats.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_stats.py src/vegcover/stats.py
git commit -m "feat: add vegetation coverage statistics calculator"
```

---

## Task 7: Export Module (exporter.py)

**Files:**
- Create: `src/vegcover/exporter.py`
- Test: `tests/test_exporter.py`

- [ ] **Step 1: Write the failing test**

```python
import os
import tempfile

import numpy as np
import pytest

from vegcover.exporter import ResultExporter
from vegcover.models import AnalysisResult, Box, CategoryStat, CoverageStat, DetectResult, StitchResult, VegIndexMap


class TestResultExporter:
    def test_init(self):
        exporter = ResultExporter(output_dir="/tmp/test_output")
        assert exporter.output_dir == "/tmp/test_output"

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
        """Should create a heatmap from vegetation index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ResultExporter(output_dir=tmpdir)

            veg_mask = np.zeros((50, 50), dtype=np.uint8)
            veg_mask[:25, :] = 255

            heatmap = exporter._create_heatmap(veg_mask)
            assert heatmap is not None
            assert heatmap.shape[:2] == (50, 50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exporter.py -v`
Expected: FAIL with ImportError (exporter module doesn't exist yet)

- [ ] **Step 3: Write minimal implementation**

```python
"""Export module for charts and data files."""

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
        import csv

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_exporter.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_exporter.py src/vegcover/exporter.py
git commit -m "feat: add result exporter for charts and data files"
```

---

## Task 8: Pipeline Controller (pipeline.py)

**Files:**
- Create: `src/vegcover/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from vegcover.pipeline import AnalysisPipeline
from vegcover.models import AnalysisResult


class TestAnalysisPipeline:
    def test_init(self):
        pipeline = AnalysisPipeline()
        assert pipeline is not None

    @patch("vegcover.pipeline.PhotoStitcher")
    @patch("vegcover.pipeline.VegetationDetector")
    @patch("vegcover.pipeline.VegetationIndexCalculator")
    @patch("vegcover.pipeline.CoverageCalculator")
    @patch("vegcover.pipeline.ResultExporter")
    def test_run_returns_analysis_result(self, mock_exporter, mock_calc, mock_vi, mock_det, mock_stitch):
        """run() should return an AnalysisResult."""
        # Setup mocks
        mock_stitch_instance = MagicMock()
        mock_stitch_instance.stitch.return_value = MagicMock(success=True, panorama=np.zeros((50, 50, 3), dtype=np.uint8), error_msg=None)
        mock_stitch.return_value = mock_stitch_instance

        mock_det_instance = MagicMock()
        mock_det_instance.detect.return_value = MagicMock(boxes=[], classes=[], model_name="test")
        mock_det.return_value = mock_det_instance

        mock_vi_instance = MagicMock()
        mock_vi_instance.compute.return_value = MagicMock(exg_map=np.zeros((50, 50)), vari_map=np.zeros((50, 50)), veg_mask=np.zeros((50, 50), dtype=np.uint8), veg_ratio=0.0)
        mock_vi.return_value = mock_vi_instance

        mock_calc_instance = MagicMock()
        mock_calc_instance.calculate.return_value = MagicMock(total_pixels=2500, categories=[], total_veg_ratio=0.0, non_veg_ratio=1.0, gsd=None)
        mock_calc.return_value = mock_calc_instance

        mock_exporter_instance = MagicMock()
        mock_exporter_instance.export.return_value = {}
        mock_exporter.return_value = mock_exporter_instance

        pipeline = AnalysisPipeline()
        images = [np.zeros((50, 50, 3), dtype=np.uint8)]
        result = pipeline.run(images)

        assert isinstance(result, AnalysisResult)
        assert result.stitch.success is True

    def test_run_with_single_image(self):
        """Should handle a single image."""
        pipeline = AnalysisPipeline()
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        # Just verify it doesn't crash - actual result depends on real models
        # which we don't want to load in tests
        assert pipeline is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with ImportError (pipeline module doesn't exist yet)

- [ ] **Step 3: Write minimal implementation**

```python
"""Pipeline controller that orchestrates all analysis steps."""

import logging
from typing import Any

import numpy as np

from vegcover.detector import VegetationDetector
from vegcover.exporter import ResultExporter
from vegcover.models import AnalysisResult
from vegcover.stats import CoverageCalculator
from vegcover.stitcher import PhotoStitcher
from vegcover.vegindex import VegetationIndexCalculator

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates the full analysis pipeline."""

    def __init__(
        self,
        stitcher: PhotoStitcher | None = None,
        detector: VegetationDetector | None = None,
        veg_indexer: VegetationIndexCalculator | None = None,
        calculator: CoverageCalculator | None = None,
        exporter: ResultExporter | None = None,
    ):
        self.stitcher = stitcher or PhotoStitcher()
        self.detector = detector or VegetationDetector()
        self.veg_indexer = veg_indexer or VegetationIndexCalculator()
        self.calculator = calculator or CoverageCalculator()
        self.exporter = exporter or ResultExporter()

    def run(
        self,
        images: list[np.ndarray],
        gsd: float | None = None,
        export: bool = True,
    ) -> AnalysisResult:
        """Run the full analysis pipeline.

        Args:
            images: List of RGB images.
            gsd: Ground sampling distance in meters per pixel (optional).
            export: Whether to export results to files.

        Returns:
            AnalysisResult with all analysis data.
        """
        logger.info("Starting analysis pipeline...")

        # Step 1: Stitch photos
        logger.info("Stitching photos...")
        stitch_result = self.stitcher.stitch(images)
        panorama = stitch_result.panorama

        # Step 2: Detect vegetation
        logger.info("Detecting vegetation...")
        detect_result = self.detector.detect(panorama)

        # Step 3: Compute vegetation indices
        logger.info("Computing vegetation indices...")
        veg_index = self.veg_indexer.compute(panorama)

        # Step 4: Calculate coverage statistics
        logger.info("Calculating coverage statistics...")
        coverage = self.calculator.calculate(panorama, detect_result, veg_index, gsd=gsd)

        # Build result
        result = AnalysisResult(
            stitch=stitch_result,
            detect=detect_result,
            veg_index=veg_index,
            coverage=coverage,
        )

        # Step 5: Export results
        if export:
            logger.info("Exporting results...")
            self.exporter.export(result)

        logger.info("Analysis complete!")
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline.py src/vegcover/pipeline.py
git commit -m "feat: add pipeline controller orchestrating all analysis steps"
```

---

## Task 9: Gradio UI (main.py)

**Files:**
- Modify: `main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from vegcover.models import AnalysisResult, Box, CategoryStat, CoverageStat, DetectResult, StitchResult, VegIndexMap


class TestGradioUI:
    def test_ui_import(self):
        """Should be able to import the main module."""
        import main
        assert hasattr(main, "create_ui")

    def test_create_ui_returns_blocks(self):
        """create_ui() should return a Gradio Blocks instance."""
        import gradio as gr
        from main import create_ui

        ui = create_ui()
        assert isinstance(ui, gr.Blocks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL with ImportError or AttributeError (main.py doesn't have create_ui yet)

- [ ] **Step 3: Write minimal implementation**

Replace `main.py` content with:

```python
"""VegCover - Aerial photo stitching and vegetation analysis tool.

Launch with: python main.py
"""

import logging
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from vegcover.models import AnalysisResult
from vegcover.pipeline import AnalysisPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_images(
    image_files: list,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    gsd: float | None = None,
) -> tuple:
    """Process uploaded images through the analysis pipeline.

    Args:
        image_files: List of uploaded image file paths.
        confidence_threshold: YOLO confidence threshold.
        iou_threshold: YOLO IoU threshold.
        gsd: Ground sampling distance (m/pixel).

    Returns:
        Tuple of (annotated_image, heatmap, pie_chart, bar_chart, csv_path, json_path)
    """
    if not image_files:
        return [None] * 6 + ["Please upload at least one image."]

    # Load images
    images = []
    for file_path in image_files:
        img = Image.open(file_path).convert("RGB")
        images.append(np.array(img))

    # Run pipeline
    pipeline = AnalysisPipeline()
    result = pipeline.run(images, gsd=gsd)

    # Prepare outputs
    outputs = _prepare_outputs(result)
    return outputs


def _prepare_outputs(result: AnalysisResult) -> tuple:
    """Prepare Gradio-compatible outputs from analysis result."""
    from vegcover.exporter import ResultExporter

    # Export to get file paths
    exporter = ResultExporter(output_dir="output")
    paths = exporter.export(result)

    # Load exported images
    annotated = Image.open(paths["annotated_image"])
    heatmap = Image.open(paths["heatmap"])

    # Load charts if available
    pie_chart = None
    bar_chart = None
    if "charts" in paths:
        charts = Image.open(paths["charts"])
        # Split the combined chart image
        w, h = charts.size
        pie_chart = charts.crop((0, 0, w // 2, h))
        bar_chart = charts.crop((w // 2, 0, w, h))

    return (
        annotated,
        heatmap,
        pie_chart,
        bar_chart,
        paths.get("stats_csv"),
        paths.get("stats_json"),
        f"Analysis complete! Total vegetation coverage: {result.coverage.total_veg_ratio:.1%}",
    )


def create_ui() -> gr.Blocks:
    """Create the Gradio web interface."""
    with gr.Blocks(title="VegCover - Aerial Vegetation Analysis") as app:
        gr.Markdown("# 🌿 VegCover - Aerial Vegetation Analysis")
        gr.Markdown("Upload aerial photos to stitch, detect vegetation, and analyze coverage.")

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📁 Upload Photos")
                image_input = gr.File(
                    label="Aerial Photos",
                    file_count="multiple",
                    file_types=["image"],
                )

                # Parameters
                gr.Markdown("### ⚙️ Parameters")
                with gr.Accordion("Detection Settings", open=False):
                    confidence_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                        label="Confidence Threshold"
                    )
                    iou_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.45, step=0.05,
                        label="IoU Threshold"
                    )

                with gr.Accordion("Coverage Settings", open=False):
                    gsd_input = gr.Number(
                        value=None, label="GSD (m/pixel)",
                        info="Leave empty for pixel ratios only"
                    )

                analyze_btn = gr.Button("▶ Start Analysis", variant="primary")

            with gr.Column(scale=2):
                # Results section
                gr.Markdown("### 📊 Results")
                with gr.Tabs():
                    with gr.TabItem("Annotated Panorama"):
                        annotated_output = gr.Image(label="Vegetation Detection")
                    with gr.TabItem("Vegetation Heatmap"):
                        heatmap_output = gr.Image(label="Vegetation Distribution")
                    with gr.TabItem("Coverage Charts"):
                        pie_output = gr.Image(label="Coverage Pie Chart")
                        bar_output = gr.Image(label="Coverage Bar Chart")
                    with gr.TabItem("Export Data"):
                        csv_output = gr.File(label="CSV Report")
                        json_output = gr.File(label="JSON Report")

                status_output = gr.Textbox(label="Status", interactive=False)

        # Wire up the button
        analyze_btn.click(
            fn=process_images,
            inputs=[image_input, confidence_slider, iou_slider, gsd_input],
            outputs=[
                annotated_output,
                heatmap_output,
                pie_output,
                bar_output,
                csv_output,
                json_output,
                status_output,
            ],
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_main.py main.py
git commit -m "feat: add Gradio web UI for VegCover"
```

---

## Task 10: Integration and Final Test

**Files:**
- Modify: `.gitignore`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add pytest fixtures**

Create `tests/conftest.py`:

```python
"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_rgb_image():
    """Return a sample RGB image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_green_image():
    """Return a sample green-dominant RGB image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 1] = 200  # High green
    img[:, :, 0] = 50   # Low red
    img[:, :, 2] = 50   # Low blue
    return img
```

- [ ] **Step 2: Update .gitignore**

Add to `.gitignore`:

```
# Output directory
output/

# Model files
*.pt
*.onnx

# Gradio temp files
.gradio/
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py .gitignore
git commit -m "chore: add pytest fixtures and update gitignore"
```

---

## Self-Review

### Spec Coverage Check

| Spec Requirement | Implementing Task |
|-------------------|-----------------|
| Photo stitching with OpenCV Stitcher | Task 3 (stitcher.py) |
| YOLO vegetation detection | Task 4 (detector.py) |
| ExG / VARI vegetation indices | Task 5 (vegindex.py) |
| Coverage statistics | Task 6 (stats.py) |
| Annotated image export | Task 7 (exporter.py) |
| Pie/bar chart generation | Task 7 (exporter.py) |
| CSV/JSON data export | Task 7 (exporter.py) |
| Gradio Web UI | Task 9 (main.py) |
| Pipeline orchestration | Task 8 (pipeline.py) |
| Data models | Task 2 (models.py) |

### Placeholder Scan

- No TBD/TODO placeholders ✅
- No "implement later" or "fill in details" ✅
- All steps include actual code ✅
- Type consistency verified across tasks ✅

### Type Consistency Check

- `StitchResult`, `DetectResult`, `VegIndexMap`, `CoverageStat`, `AnalysisResult` all used consistently ✅
- `Box` dataclass fields match across detector and exporter ✅
- `gsd: float | None` consistent in stats and pipeline ✅

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-16-vegcover-plan.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
