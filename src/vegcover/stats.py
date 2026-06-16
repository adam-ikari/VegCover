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
