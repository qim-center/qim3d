import re

import numpy as np
import pytest

import qim3d

def test_remove_background():
    # Volume with a noisy, dark background and a white cube in the middle
    vol = np.zeros((10, 10, 10), dtype=np.uint8)
    noise = np.random.default_rng(seed = 0).uniform(0, 50, size=vol.shape).astype(np.uint8)
    vol += noise
    vol[3:7, 3:7, 3:7] = 255

    # Apply the remove_background function
    vol_filtered = qim3d.operations.remove_background(vol, min_object_radius=1, background='dark')

    # Check the volume shape and intensities
    assert vol.shape == vol_filtered.shape, 'Volume shape changed'
    assert not np.array_equal(vol, vol_filtered), 'Volume intensities did not change'

    # Check if the background has been removed (blurred) and that the white cube is still there
    assert vol_filtered[0:3, 0:3, 0:3].std() < vol[0:3, 0:3, 0:3].std(), 'Background standard deviation did not decrease'
    assert vol_filtered[5, 5, 5] == 255, 'Foreground object intensity changed'

def test_fade_mask():
    # Volume with a noisy background and a white cylinder in the middle
    vol = np.zeros((10, 10, 10), dtype=np.uint8)
    noise = np.random.default_rng(seed=0).uniform(0, 50, size=vol.shape).astype(np.uint8)
    vol += noise
    vol[:, 4:6, 4:6] = 255

    # Apply the fade_mask function
    vol_faded = qim3d.operations.fade_mask(vol, decay_rate=0.1, ratio=0.5, geometry='cylindrical', axis=0)

    # Check the volume shape and intensities
    assert vol.shape == vol_faded.shape, 'Volume shape changed'
    assert not np.array_equal(vol, vol_faded), 'Volume intensities did not change'

    # The background should be faded (lower intensity) at the edges
    assert vol_faded[0, 0, 0] < vol[0, 0, 0], 'Background at the edge did not fade'
    assert vol_faded[9, 9, 9] < vol[9, 9, 9], 'Background at the edge did not fade'

    # The white cylinder should still be visible
    assert vol_faded[:, 4:6, 4:6].max() == 255, 'White cylinder should still be visible'

def test_overlay_rgb_images():
    # Volume with noisy background and white cylinder in the center
    vol = np.zeros((10, 10, 10), dtype=np.uint8)
    noise = np.random.default_rng(seed = 0).uniform(0, 50, size=vol.shape).astype(np.uint8)
    vol += noise
    vol[:, 3:7, 3:7] = 255

    # Slice of the volume
    slice = vol[5, :, :]

    # RGB mask of the slice
    mask = slice > 50
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[:, :, 0] = mask

    # Apply the overlay_rgb_images function
    composite = qim3d.operations.overlay_rgb_images(slice, mask_rgb, alpha=0.5, hide_black=True)

    # Check the composite image shape
    assert composite.shape == (slice.shape[0], slice.shape[1], 3), 'Composite image shape is incorrect'

    # Check that the background color remains unchanged where there is no mask
    background_indices = np.where(mask == 0)
    assert np.all(composite[background_indices] == slice[background_indices][:, None]), 'Background color changed where there is no mask'

    # Check that the foreground color is correctly blended with the background where the mask is applied
    foreground_indices = np.where(mask == 1)
    expected_foreground_color = (slice[foreground_indices] * 0.5).astype(np.uint8)
    assert np.all(composite[foreground_indices][:, 1] == expected_foreground_color), 'Foreground color not blended correctly'