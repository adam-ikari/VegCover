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

        ExG = 2G - R - B, normalized to [0, 1] using fixed scaling
        based on 8-bit RGB range ([-510, 510]).
        """
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        exg = 2 * g - r - b
        # Fixed normalization: ExG range for 8-bit RGB is [-510, 510]
        exg = (exg + 510) / 1020
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
        # With fixed normalization, ExG=0.5 is the neutral point (R=G=B)
        # Pixels with ExG > 0.5 have more green than red+blue, indicating vegetation
        veg_mask = (exg > 0.5).astype(np.uint8)

        # Calculate vegetation ratio
        veg_ratio = float(veg_mask.sum() / veg_mask.size)

        return VegIndexMap(
            exg_map=exg,
            vari_map=vari,
            veg_mask=veg_mask,
            veg_ratio=veg_ratio,
        )
