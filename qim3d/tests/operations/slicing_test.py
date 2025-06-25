import numpy as np
import pytest

from qim3d.operations import RectangularSlicer, get_random_slice
from interactive_unet.slicer import Slicer
from scipy.ndimage import map_coordinates

@pytest.fixture
def volume():
    # simple ramp: shape (10,10,10) with unique values
    x = np.arange(10)
    y = np.arange(10)[:, None]
    z = np.arange(10)[:, None, None]
    return x + 10*y + 100*z

def test_random_slice_shape(volume):
    """Requested width/length must show up in the output shape."""
    sl = get_random_slice(volume, width=5, length=8, seed=0)
    assert sl.shape == (5, 8)

def test_random_slice_reproducible(volume):
    """Same seed → identical slices."""
    s1 = get_random_slice(volume, width=6, length=4, seed=42)
    s2 = get_random_slice(volume, width=6, length=4, seed=42)
    np.testing.assert_array_equal(s1, s2)

def test_random_slice_differs_with_different_seed(volume):
    """Different seeds → (very likely) different slices."""
    s1 = get_random_slice(volume, width=6, length=4, seed=1)
    s2 = get_random_slice(volume, width=6, length=4, seed=2)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(s1, s2)

def test_random_slice_within_bounds(volume):
    """No NaNs and values within the original volume’s min/max."""
    sl = get_random_slice(volume, width=7, length=7, seed=7)
    assert not np.isnan(sl).any()
    assert sl.min() >= volume.min()
    assert sl.max() <= volume.max()

def test_rectangular_slicer_shape(volume):
    """
    If I force a grid sampling on the z-axis, 
    I should get a slice of exactly (width, length).
    """
    slicer = RectangularSlicer(volume.shape)
    slicer.randomize(sampling_mode='grid', sampling_axis='z')
    sl = slicer.get_slice(volume, width=5, length=8, axis=0, order=1)
    assert sl.shape == (5, 8)

def test_rectangular_slicer_square_matches_base(volume):
    """
    When width==length, RectangularSlicer must behave identically
    to the upstream Slicer for the same random seed.
    """
    seed = 999
    np.random.seed(seed)
    base = Slicer(volume.shape)
    base.randomize(sampling_mode='random')
    # original square slice
    orig = base.get_slice(volume, axis=0, slice_width=6, order=1)

    # same seed for our RectangularSlicer
    np.random.seed(seed)
    rect = RectangularSlicer(volume.shape)
    rect.randomize(sampling_mode='random')
    patch = rect.get_slice(volume, width=6, length=6, axis=0, order=1)

    np.testing.assert_allclose(patch, orig)

def test_get_random_slice_shape_and_bounds(volume):
    """
    Quick sanity for the wrapper: shape, no NaNs, within [min,max].
    """
    sl = get_random_slice(volume, width=7, length=4, seed=42)
    assert sl.shape == (7, 4)
    assert not np.isnan(sl).any()
    assert sl.min() >= volume.min() and sl.max() <= volume.max()