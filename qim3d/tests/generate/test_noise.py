"""Checks for the internal noise backend and its public volume integration."""

import numpy as np
import pytest

from qim3d.generate import volume, volume_collection
from qim3d.generate._noise import _noise_field, _perlin, _simplex


@pytest.mark.parametrize("algorithm", ["perlin", "simplex"])
def test_seed_and_chunk_independence(algorithm):
    """The seed controls the pattern; splitting work into batches must not."""
    field = _noise_field((11, 13, 15), 0.17, algorithm, 42)
    # Compare one batch with batches that split rows and leave a partial batch.
    np.testing.assert_array_equal(
        field, _noise_field(field.shape, 0.17, algorithm, 42, chunk_size=37)
    )
    assert not np.array_equal(field, _noise_field(field.shape, 0.17, algorithm, 43))
    assert np.isfinite(field).all()
    assert np.ptp(field) > 0


@pytest.mark.parametrize("evaluate", [_perlin, _simplex])
def test_nearby_points_have_similar_values(evaluate):
    """Check small movements at selected positions, not continuity everywhere."""
    permutation = np.tile(np.random.default_rng(42).permutation(256), 2)
    # Sample positive/negative grid vertices, tied coordinates, and an interior
    # point. Coordinates are already in noise space, after scaling.
    points = np.array(
        [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.4, 0.4, 0.4], [0.1, 0.7, 0.2]]
    )
    center = evaluate(points, permutation)
    # Move a small distance in each axis direction; allow normal slope variation.
    for axis in np.eye(3):
        np.testing.assert_allclose(
            evaluate(points + axis * 1e-7, permutation), center, atol=1e-5
        )
        np.testing.assert_allclose(
            evaluate(points - axis * 1e-7, permutation), center, atol=1e-5
        )


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize("axes", [(0, 1, 2), (0, 2, 1), (2, 0, 1)])
def test_simplex_tetrahedron_boundary(seed, axes):
    """Switching the selected corner must not introduce a jump at an axis tie."""
    permutation = np.tile(np.random.default_rng(seed).permutation(256), 2)
    # In noise coordinates, x = y > z is a boundary where the selected corner
    # changes between (1, 0, 0) and (0, 1, 0). Permute axes to test each pair.
    boundary = np.array([0.4, 0.4, 0.0])[list(axes)]
    direction = np.array([1.0, -1.0, 0.0])[list(axes)]
    # Repeat the boundary check at positive and negative locations.
    points = boundary + np.array([[0, 0, 0], [-2, -2, -2], [3, 3, 3]])
    center = _simplex(points, permutation)
    # Approach from both sides: value differences should shrink with step size.
    for epsilon in (1e-7, 1e-9):
        for sign in (-1, 1):
            nearby = _simplex(points + sign * epsilon * direction, permutation)
            np.testing.assert_allclose(nearby, center, rtol=0, atol=20 * epsilon)


@pytest.mark.parametrize("algorithm", ["perlin", "simplex"])
def test_volume_seed(algorithm):
    kwargs = dict(base_shape=(16, 18, 20), noise_scale=0.12, noise_type=algorithm)
    first = volume(**kwargs, seed=42)
    np.testing.assert_array_equal(first, volume(**kwargs, seed=42))
    assert not np.array_equal(first, volume(**kwargs, seed=43))
    assert first.dtype == np.uint8
    assert first.shape == kwargs["base_shape"]
    assert first.max() > 0


def test_constant_perlin_field():
    """A constant noise field must normalize safely to an untextured shape."""
    # Noise coordinates = integer voxel coordinates * noise_scale. At scale 1,
    # every sample lies on a Perlin grid vertex, where its value is zero.
    # Normalization must avoid dividing by the resulting zero range and produce
    # the same shape as scale 0, which explicitly disables texture.
    with np.errstate(divide="raise", invalid="raise"):
        result = volume(base_shape=(12, 12, 12), noise_scale=1, dtype="float32")
        smooth = volume(base_shape=(12, 12, 12), noise_scale=0, dtype="float32")
    np.testing.assert_array_equal(result, smooth)
    assert np.isfinite(result).all()
    assert result.max() > 0
