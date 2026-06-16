"""Photo stitching module using OpenCV Stitcher."""

import logging

import cv2
import numpy as np

from vegcover.models import StitchResult

logger = logging.getLogger(__name__)


class PhotoStitcher:
    """Stitches multiple aerial photos into a panorama."""

    def __init__(self):
        pass

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
                # transform_matrices reserved for future homography matrix extraction
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
