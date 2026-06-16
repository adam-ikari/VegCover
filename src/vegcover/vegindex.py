"""Vegetation index computation for RGB images.

Supports detection of both green vegetation and yellow/senescent vegetation
using multiple vegetation indices combined.
"""

import logging

import cv2
import numpy as np

from vegcover.models import VegIndexMap

logger = logging.getLogger(__name__)


class VegetationIndexCalculator:
    """Computes vegetation indices from RGB images.

    Uses multiple indices to detect both green and yellow/senescent vegetation:
    - ExG (Excess Greenness): Detects green vegetation
    - VARI (Visible Atmospherically Resistant Index): Detects general vegetation
    - ExR (Excess Redness): Detects yellow/senescent vegetation
    - Combined approach: Green OR Yellow vegetation = vegetation
    """

    def __init__(self, threshold_method: str = "adaptive"):
        self.threshold_method = threshold_method

    def _compute_exg(self, image: np.ndarray) -> np.ndarray:
        """Compute Excess Greenness (ExG) index.

        ExG = 2G - R - B, normalized to [0, 1].
        Higher values indicate more green vegetation.
        """
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        exg = 2 * g - r - b
        # Normalize to [0, 1]
        exg = (exg + 510) / 1020
        return exg

    def _compute_exr(self, image: np.ndarray) -> np.ndarray:
        """Compute Excess Redness (ExR) index.

        ExR = 2R - G - B, normalized to [0, 1].
        Higher values indicate more red/yellow vegetation (senescent).
        """
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        exr = 2 * r - g - b
        exr = (exr + 510) / 1020
        return exr

    def _compute_vari(self, image: np.ndarray) -> np.ndarray:
        """Compute Visible Atmospherically Resistant Index (VARI).

        VARI = (G - R) / (G + R - B), clamped to [-1, 1].
        General vegetation index, less sensitive to atmospheric effects.
        """
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        denominator = g + r - b
        denominator = np.where(denominator == 0, 1e-10, denominator)

        vari = (g - r) / denominator
        vari = np.clip(vari, -1.0, 1.0)
        return vari

    def _is_green_vegetation(self, image: np.ndarray) -> np.ndarray:
        """Detect green vegetation using ExG and color ratio."""
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        # ExG > threshold AND green is dominant channel
        exg = 2 * g - r - b
        green_dominant = (g > r) & (g > b)
        exg_positive = exg > 20  # ExG threshold for green vegetation

        return green_dominant & exg_positive

    def _is_yellow_vegetation(self, image: np.ndarray) -> np.ndarray:
        """Detect yellow/senescent vegetation.

        Yellow vegetation: R high, G medium-high, B low
        Characteristics: R > G > B, but G is still significant
        """
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        # Yellow: R and G are both high, B is low
        # R > B and G > B (yellow is red+green)
        # R and G are similar (yellow, not orange/red)
        # But not too green (green vegetation is handled separately)
        yellowish = (r > b + 30) & (g > b + 30)  # Both R and G higher than B
        not_too_green = g < r + 50  # Not strongly green dominant
        not_brown = (r + g) > (2 * b + 40)  # Not just brown soil
        has_color = (r + g) > 100  # Has some color (not dark shadow)

        return yellowish & not_too_green & not_brown & has_color

    def _is_senescent_vegetation(self, image: np.ndarray) -> np.ndarray:
        """Detect senescent (brown/yellow-brown) vegetation.

        Senescent vegetation: R medium-high, G medium, B low
        Brownish color with some green remaining.
        """
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        # Brown/yellow-brown: R > G > B, moderate saturation
        r_greater_than_g = r > g
        g_greater_than_b = g > b + 10
        not_too_dark = (r + g + b) > 150  # Not too dark
        has_some_green = g > 50  # Still has some green (distinguishes from bare soil)

        return r_greater_than_g & g_greater_than_b & not_too_dark & has_some_green

    def compute(self, image: np.ndarray) -> VegIndexMap:
        """Compute vegetation indices and mask.

        Detects three types of vegetation:
        - Green vegetation (healthy)
        - Yellow vegetation (stressed/senescent)
        - Brown/senescent vegetation (dry)

        Args:
            image: RGB image as numpy array (H, W, 3).

        Returns:
            VegIndexMap with ExG, VARI, and binary vegetation mask.
        """
        exg = self._compute_exg(image)
        vari = self._compute_vari(image)

        # Detect different vegetation types
        green_mask = self._is_green_vegetation(image)
        yellow_mask = self._is_yellow_vegetation(image)
        senescent_mask = self._is_senescent_vegetation(image)

        # Combine: vegetation = green OR yellow OR senescent
        # But exclude pure soil (very low saturation)
        veg_mask = green_mask | yellow_mask | senescent_mask

        # Calculate vegetation ratio
        veg_ratio = float(veg_mask.sum() / veg_mask.size)

        logger.info(
            f"Vegetation detection: green={green_mask.sum()}, "
            f"yellow={yellow_mask.sum()}, senescent={senescent_mask.sum()}, "
            f"total={veg_mask.sum()}"
        )

        return VegIndexMap(
            exg_map=exg,
            vari_map=vari,
            veg_mask=veg_mask.astype(np.uint8),
            veg_ratio=veg_ratio,
        )
