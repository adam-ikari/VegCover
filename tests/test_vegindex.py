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
