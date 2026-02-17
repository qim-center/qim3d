import numpy as np
import pytest

from qim3d.generate._shapes import (
    _generate_berry_positions,
    _generate_twisted_thread,
    _overlapping_placement,
    berry,
    rope,
)

# =============================================================================
# Tests for _generate_twisted_thread
# =============================================================================


def test_generate_twisted_thread_basic():
    """Test basic thread generation."""
    length, thickness = 100, 10
    thread = _generate_twisted_thread(
        length=length,
        thickness=thickness,
        twist_rate=2.0,
        phase_offset=0.0,
        noise_scale=0.03,
        seed=42,
    )
    
    assert thread.shape == (length, thickness, thickness)
    assert thread.dtype == np.uint8
    assert np.any(thread > 0), 'Thread should have non-zero values'


def test_generate_twisted_thread_reproducibility():
    """Test that same seed produces identical results."""
    params = {
        'length': 50,
        'thickness': 8,
        'twist_rate': 1.5,
        'phase_offset': 45.0,
        'noise_scale': 0.02,
        'seed': 123,
    }
    
    thread1 = _generate_twisted_thread(**params)
    thread2 = _generate_twisted_thread(**params)
    
    np.testing.assert_array_equal(thread1, thread2)


def test_generate_twisted_thread_different_seeds():
    """Test that different seeds produce different results."""
    params = {
        'length': 50,
        'thickness': 8,
        'twist_rate': 1.5,
        'phase_offset': 45.0,
        'noise_scale': 0.02,
    }
    
    thread1 = _generate_twisted_thread(**params, seed=1)
    thread2 = _generate_twisted_thread(**params, seed=2)
    
    assert not np.array_equal(thread1, thread2)


def test_generate_twisted_thread_twist_rate():
    """Test different twist rates affect the output."""
    params = {
        'length': 50,
        'thickness': 8,
        'phase_offset': 0.0,
        'noise_scale': 0.02,
        'seed': 42,
    }
    
    thread_low_twist = _generate_twisted_thread(**params, twist_rate=0.5)
    thread_high_twist = _generate_twisted_thread(**params, twist_rate=5.0)
    
    assert not np.array_equal(thread_low_twist, thread_high_twist)


# =============================================================================
# Tests for _generate_berry_positions
# =============================================================================



def test_generate_berry_positions_reproducibility():
    """Test reproducibility with same seed."""
    params = {
        'core_radius': 15,
        'drupelet_radius': 8,
        'num_drupelets': 20,
        'seed': 123,
    }
    
    pos1, rad1 = _generate_berry_positions(**params)
    pos2, rad2 = _generate_berry_positions(**params)
    
    assert pos1 == pos2
    assert rad1 == rad2


def test_generate_berry_positions_with_jitter():
    """Test position and radius jitter."""
    positions, radii = _generate_berry_positions(
        core_radius=20,
        drupelet_radius=10,
        num_drupelets=30,
        drupelet_radius_jitter=3,
        position_jitter=5,
        seed=42,
    )
    
    assert len(positions) == 30
    assert len(radii) == 30
    # Check that radii vary due to jitter
    assert len(set(radii)) > 1


# =============================================================================
# Tests for _overlapping_placement
# =============================================================================


def test_overlapping_placement_no_overlap():
    """Test placing blob without overlap."""
    collection = np.zeros((50, 50, 50), dtype=np.uint8)
    blob = np.ones((10, 10, 10), dtype=np.uint8) * 100
    position = (25, 25, 25)
    
    result, placed = _overlapping_placement(collection, blob, position)
    
    assert placed is True
    assert np.any(result > 0)


def test_overlapping_placement_with_labels():
    """Test placement with label tracking."""
    collection = np.zeros((50, 50, 50), dtype=np.uint8)
    labels = np.zeros((50, 50, 50), dtype=np.uint8)
    blob = np.ones((10, 10, 10), dtype=np.uint8) * 100
    position = (25, 25, 25)
    
    result, placed = _overlapping_placement(collection, blob, position, labels, label_id=1)
    
    assert placed is True
    assert np.any(labels == 1)


def test_overlapping_placement_out_of_bounds():
    """Test placement outside bounds."""
    collection = np.zeros((50, 50, 50), dtype=np.uint8)
    blob = np.ones((10, 10, 10), dtype=np.uint8) * 100
    position = (4, 4, 4)  # Too close to edge
    
    result, placed = _overlapping_placement(collection, blob, position)
    
    assert placed is False


# =============================================================================
# Tests for berry function
# =============================================================================


def test_berry_basic():
    """Test basic berry generation."""
    berry_vol = berry(
        shape=(100, 100, 100),
        num_drupelets=30,
        seed=42,
    )
    
    assert berry_vol.shape == (100, 100, 100)
    assert berry_vol.dtype == np.uint8
    assert np.any(berry_vol > 0), 'Berry should have non-zero values'


def test_berry_with_labels():
    """Test berry generation with labels."""
    berry_vol, labels = berry(
        shape=(100, 100, 100),
        num_drupelets=30,
        seed=42,
        return_labels=True,
    )
    
    assert berry_vol.shape == (100, 100, 100)
    assert labels.shape == (100, 100, 100)
    assert labels.dtype == np.uint8
    assert np.max(labels) > 0, 'Labels should be assigned'


def test_berry_reproducibility():
    """Test berry reproducibility with same seed."""
    params = {
        'shape': (80, 80, 80),
        'num_drupelets': 20,
        'seed': 123,
    }
    
    berry1 = berry(**params)
    berry2 = berry(**params)
    
    np.testing.assert_array_equal(berry1, berry2)


def test_berry_different_seeds():
    """Test different seeds produce different berries."""
    params = {
        'shape': (80, 80, 80),
        'num_drupelets': 20,
    }
    
    berry1 = berry(**params, seed=1)
    berry2 = berry(**params, seed=2)
    
    assert not np.array_equal(berry1, berry2)


def test_berry_parameter_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError, match='num_drupelets must be at least 1'):
        berry(num_drupelets=0)
    
    with pytest.raises(ValueError, match='core_radius must be positive'):
        berry(core_radius=-5)
    
    with pytest.raises(ValueError, match='drupelet_radius must be positive'):
        berry(drupelet_radius=0)
    
    with pytest.raises(ValueError, match='All shape dimensions must be positive'):
        berry(shape=(100, 0, 100))
    
    with pytest.raises(ValueError, match='threshold must be between 0 and 1'):
        berry(threshold=1.5)
    
    with pytest.raises(ValueError, match='top_opening_threshold must be between 0 and 1'):
        berry(top_opening_threshold=1.2)
    
    with pytest.raises(ValueError, match='rim_thickness must be between 0 and 1'):
        berry(rim_thickness=0)
    
    with pytest.raises(ValueError, match='gamma must be positive'):
        berry(gamma=-0.5)


def test_berry_different_configurations():
    """Test berry with various parameter configurations."""
    # Small berry
    small_berry = berry(shape=(50, 50, 50), num_drupelets=10, seed=42)
    assert small_berry.shape == (50, 50, 50)
    
    # Large berry
    large_berry = berry(shape=(150, 150, 150), num_drupelets=80, seed=42)
    assert large_berry.shape == (150, 150, 150)
    
    # Different opening threshold
    open_berry = berry(top_opening_threshold=0.7, seed=42)
    assert open_berry.shape == (200, 200, 200)


# =============================================================================
# Tests for rope function
# =============================================================================


def test_rope_basic():
    """Test basic rope generation."""
    rope_vol = rope(
        shape=(200, 60, 60),
        num_threads=12,
        seed=42,
    )
    
    assert rope_vol.shape == (200, 60, 60)
    assert rope_vol.dtype == np.uint8
    assert np.any(rope_vol > 0), 'Rope should have non-zero values'


def test_rope_with_labels():
    """Test rope generation with labels."""
    rope_vol, labels = rope(
        shape=(200, 60, 60),
        num_threads=12,
        seed=42,
        return_labels=True,
    )
    
    assert rope_vol.shape == (200, 60, 60)
    assert labels.shape == (200, 60, 60)
    assert labels.dtype == np.uint8
    assert np.max(labels) > 0, 'Labels should be assigned'


def test_rope_reproducibility():
    """Test rope reproducibility with same seed."""
    params = {
        'shape': (150, 50, 50),
        'num_threads': 10,
        'seed': 123,
    }
    
    rope1 = rope(**params)
    rope2 = rope(**params)
    
    np.testing.assert_array_equal(rope1, rope2)


def test_rope_different_seeds():
    """Test different seeds produce different ropes."""
    params = {
        'shape': (150, 50, 50),
        'num_threads': 10,
    }
    
    rope1 = rope(**params, seed=1)
    rope2 = rope(**params, seed=2)
    
    assert not np.array_equal(rope1, rope2)


def test_rope_parameter_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError, match='num_threads must be at least 1'):
        rope(num_threads=0)
    
    with pytest.raises(ValueError, match='thread_thickness must be positive'):
        rope(thread_thickness=-5)
    
    with pytest.raises(ValueError, match='All shape dimensions must be positive'):
        rope(shape=(100, 0, 50))
    
    with pytest.raises(ValueError, match='Rope cross-section must be square'):
        rope(shape=(100, 50, 60))
    
    with pytest.raises(ValueError, match='twist_rate must be non-negative'):
        rope(twist_rate=-1.0)
    
    with pytest.raises(ValueError, match='compression_factor must be between 0 and 1'):
        rope(compression_factor=1.5)
    
    with pytest.raises(ValueError, match='thread_spacing must be positive'):
        rope(thread_spacing=-0.5)


def test_rope_different_thread_counts():
    """Test rope with different thread counts."""
    # Single thread
    single_thread = rope(shape=(100, 40, 40), num_threads=1, seed=42)
    assert single_thread.shape == (100, 40, 40)
    
    # Many threads
    many_threads = rope(shape=(100, 80, 80), num_threads=24, seed=42)
    assert many_threads.shape == (100, 80, 80)


def test_rope_different_twist_rates():
    """Test rope with different twist rates."""
    params = {
        'shape': (150, 50, 50),
        'num_threads': 10,
        'seed': 42,
    }
    
    low_twist = rope(**params, twist_rate=0.5)
    high_twist = rope(**params, twist_rate=5.0)
    
    assert not np.array_equal(low_twist, high_twist)


def test_rope_with_compression():
    """Test rope with different compression factors."""
    params = {
        'shape': (150, 50, 50),
        'num_threads': 10,
        'seed': 42,
    }
    
    no_compression = rope(**params, compression_factor=0.0)
    with_compression = rope(**params, compression_factor=0.5)
    
    assert not np.array_equal(no_compression, with_compression)