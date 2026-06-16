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
