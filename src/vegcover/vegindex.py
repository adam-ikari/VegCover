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
        elif exg_max > 0:
            exg = np.ones_like(exg)
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
