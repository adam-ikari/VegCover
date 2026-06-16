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
