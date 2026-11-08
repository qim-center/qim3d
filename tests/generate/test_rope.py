import numpy as np
import pytest

from qim3d.generate._shapes import _generate_twisted_thread, rope


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
    assert np.any(thread > 0), "Thread should have non-zero values"


def test_generate_twisted_thread_reproducibility():
    """Test that same seed produces identical results."""
    params = {
        "length": 50,
        "thickness": 8,
        "twist_rate": 1.5,
        "phase_offset": 45.0,
        "noise_scale": 0.02,
        "seed": 123,
    }

    thread1 = _generate_twisted_thread(**params)
    thread2 = _generate_twisted_thread(**params)

    np.testing.assert_array_equal(thread1, thread2)


def test_generate_twisted_thread_different_seeds():
    """Test that different seeds produce different results."""
    params = {
        "length": 50,
        "thickness": 8,
        "twist_rate": 1.5,
        "phase_offset": 45.0,
        "noise_scale": 0.02,
    }

    thread1 = _generate_twisted_thread(**params, seed=1)
    thread2 = _generate_twisted_thread(**params, seed=2)

    assert not np.array_equal(thread1, thread2)


def test_generate_twisted_thread_twist_rate():
    """Test different twist rates affect the output."""
    params = {
        "length": 50,
        "thickness": 8,
        "phase_offset": 0.0,
        "noise_scale": 0.02,
        "seed": 42,
    }

    thread_low_twist = _generate_twisted_thread(**params, twist_rate=0.5)
    thread_high_twist = _generate_twisted_thread(**params, twist_rate=5.0)

    assert not np.array_equal(thread_low_twist, thread_high_twist)


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
    assert np.any(rope_vol > 0), "Rope should have non-zero values"


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
    assert np.max(labels) > 0, "Labels should be assigned"


def test_rope_reproducibility():
    """Test rope reproducibility with same seed."""
    params = {
        "shape": (150, 50, 50),
        "num_threads": 10,
        "seed": 123,
    }

    rope1 = rope(**params)
    rope2 = rope(**params)

    np.testing.assert_array_equal(rope1, rope2)


def test_rope_different_seeds():
    """Test different seeds produce different ropes."""
    params = {
        "shape": (150, 50, 50),
        "num_threads": 10,
    }

    rope1 = rope(**params, seed=1)
    rope2 = rope(**params, seed=2)

    assert not np.array_equal(rope1, rope2)


def test_rope_parameter_validation():
    """Test parameter validation."""
    with pytest.raises(ValueError, match="num_threads must be at least 1"):
        rope(num_threads=0)

    with pytest.raises(ValueError, match="thread_thickness must be positive"):
        rope(thread_thickness=-5)

    with pytest.raises(ValueError, match="All shape dimensions must be positive"):
        rope(shape=(100, 0, 50))

    with pytest.raises(ValueError, match="Rope cross-section must be square"):
        rope(shape=(100, 50, 60))

    with pytest.raises(ValueError, match="twist_rate must be non-negative"):
        rope(twist_rate=-1.0)

    with pytest.raises(ValueError, match="compression_factor must be between 0 and 1"):
        rope(compression_factor=1.5)

    with pytest.raises(ValueError, match="thread_spacing must be positive"):
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
        "shape": (150, 50, 50),
        "num_threads": 10,
        "seed": 42,
    }

    low_twist = rope(**params, twist_rate=0.5)
    high_twist = rope(**params, twist_rate=5.0)

    assert not np.array_equal(low_twist, high_twist)


def test_rope_with_compression():
    """Test rope with different compression factors."""
    params = {
        "shape": (150, 50, 50),
        "num_threads": 10,
        "seed": 42,
    }

    no_compression = rope(**params, compression_factor=0.0)
    with_compression = rope(**params, compression_factor=0.5)

    assert not np.array_equal(no_compression, with_compression)
