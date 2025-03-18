import re

import numpy as np
import pytest

import qim3d

def test_remove_background():
    # Volume with noisy, dark background and white cube in the center
    vol = np.zeros((10, 10, 10), dtype=np.uint8)
    noise = np.random.default_rng(seed = 0).uniform(0, 50, size=vol.shape).astype(np.uint8)
    vol += noise
    vol[3:7, 3:7, 3:7] = 255

    # Apply the remove_background function
    vol_filtered = qim3d.operations.remove_background(vol, min_object_radius=1, background='dark')

    # Check if the background has been removed (blurred) and that the white cube is still there
    assert vol_filtered[0:3, 0:3, 0:3].std() < vol[0:3, 0:3, 0:3].std(), "Background standard deviation did not decrease"
    assert vol_filtered[5, 5, 5] == 255, "Foreground object intensity changed"

def test_fade_mask():
    pass

def test_overlay_rgb_images():
    pass