import numpy as np
import pytest
import qim3d

@pytest.fixture
def volume():
    # simple 3D ramp volume, shape (10,10,10)
    x = np.arange(10)
    y = np.arange(10)[:, None]
    z = np.arange(10)[:, None, None]
    return x + 10*y + 100*z

def test_random_slice_shape(volume):
    """Output shape matches requested width and length."""
    sl = qim3d.operations._slicing_operations.get_random_slice(volume, width=5, length=8, seed=0)
    assert sl.shape == (5, 8)

def test_random_slice_reproducible(volume):
    """Same seed produces identical slices."""
    s1 = qim3d.operations._slicing_operations.get_random_slice(volume, width=6, length=4, seed=42)
    s2 = qim3d.operations._slicing_operations.get_random_slice(volume, width=6, length=4, seed=42)
    np.testing.assert_array_equal(s1, s2)

def test_random_slice_differs_with_different_seed(volume):
    """Different seeds produce different slices."""
    s1 = qim3d.operations._slicing_operations.get_random_slice(volume, width=6, length=4, seed=1)
    s2 = qim3d.operations._slicing_operations.get_random_slice(volume, width=6, length=4, seed=2)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(s1, s2)

def test_random_slice_within_bounds(volume):
    """Slice values are within volume min/max and contain no NaNs."""
    sl = qim3d.operations._slicing_operations.get_random_slice(volume, width=7, length=7, seed=7)
    assert not np.isnan(sl).any()
    assert sl.min() >= volume.min()
    assert sl.max() <= volume.max()

def test_slicer_get_slice_shape(volume):
    """_Slicer.get_slice returns array with expected shape."""
    slicer = qim3d.operations._slicing_operations._Slicer(volume.shape)
    slicer.randomize(sampling_mode='grid', sampling_axis='z')
    sl = slicer.get_slice(volume, width=5, length=8, axis=0, order=1)
    assert sl.shape == (5, 8)

def test_get_random_slice_sanity(volume):
    """Quick check: shape, no NaNs, within volume bounds."""
    sl = qim3d.operations._slicing_operations.get_random_slice(volume, width=7, length=4, seed=42)
    assert sl.shape == (7, 4)
    assert not np.isnan(sl).any()
    assert sl.min() >= volume.min() and sl.max() <= volume.max()
