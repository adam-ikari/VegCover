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
