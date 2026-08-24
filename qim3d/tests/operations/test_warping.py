import numpy as np

import qim3d


def test_pad():
    """Create volume and pad it. Asserts padded size."""
    vol = np.zeros((10, 10, 10))
    vol = qim3d.operations.pad(vol, x_axis=2, y_axis=2, z_axis=2)
    assert vol.shape == (14, 14, 14)


def test_pad_to():
    """Create volume and pad it. Asserts padded size."""
    vol = np.zeros((10, 10, 10))
    vol = qim3d.operations.pad_to(vol, (20, 20, 20))
    assert vol.shape == (20, 20, 20)


def test_trim():
    """Create volume and trim it. Asserts trimmed size."""
    vol = np.zeros((10, 10, 10))
    vol[2:8, 2:8, 2:8] = 1
    vol = qim3d.operations.trim(vol)
    assert vol.shape == (6, 6, 6)


def test_shear3d():
    """Create volume and shear it. No assertions, just tests function."""
    vol = np.zeros((60, 100, 100))
    vol[:, 20:80, 20:80] = 1
    factor = 0.2
    shift = int(vol.shape[0] * factor)
    sheared_vol = qim3d.operations.shear3d(vol, x_shift_z=shift, order=1)


def test_curve_warp():
    """Create volume and curve it. No assertions, just tests function."""
    vol = np.zeros((100, 100, 100))
    vol[:, 40:60, 40:60] = 1
    warped_volume = qim3d.operations.curve_warp(vol, x_amp=10, x_periods=4)


def test_stretch():
    """Create volume and stretch it. Asserts stretched and squeezed size."""
    vol = np.zeros((100, 100, 100))
    vol[:, 20:80, 20:80] = 1

    stretched_volume = qim3d.operations.stretch(vol, x_stretch=20)
    assert stretched_volume.shape == (100, 100, 140)

    squeezed_volume = qim3d.operations.stretch(vol, x_stretch=-20)
    assert squeezed_volume.shape == (100, 100, 60)


def test_center_twist():
    """Create volume and twists it. No assertions, just tests function."""
    vol = np.zeros((100, 100, 100))
    vol[:, 20:80, 20:80] = 1

    twisted_volume = qim3d.operations.center_twist(
        vol, rotation_angle=180, axis='z', order=1
    )
