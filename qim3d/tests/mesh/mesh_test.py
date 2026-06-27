import numpy as np
import pytest
import scipy
from pygel3d import hmesh

import qim3d


def test_from_volume_valid_input():
    """Test that from_volume returns a SurfaceMesh object for a valid 3D input."""
    volume = np.random.rand(50, 50, 50).astype(np.float32)
    mesh = qim3d.mesh.from_volume(volume)
    assert isinstance(mesh, qim3d.mesh.SurfaceMesh)


def test_from_volume_invalid_input():
    """Test that from_volume raises ValueError for non-3D input."""
    volume = np.random.rand(50, 50)  # A 2D array
    with pytest.raises(ValueError, match='The input volume must be a 3D numpy array.'):
        qim3d.mesh.from_volume(volume)


def test_from_volume_mesh_precision():
    """Test that from_volume correctly applies mesh_precision."""
    volume = np.random.rand(50, 50, 50).astype(np.float32)

    # Check if downscaling correctly affects shape
    mesh_precision = 0.5
    scaled_volume = scipy.ndimage.zoom(volume, zoom=mesh_precision, order=0)
    assert scaled_volume.shape == (25, 25, 25)  # Expected downscaled shape

    # Check if invalid precision values raise ValueError
    with pytest.raises(ValueError, match='The mesh precision must be between 0 and 1.'):
        qim3d.mesh.from_volume(volume, mesh_precision=-0.1)

    with pytest.raises(ValueError, match='The mesh precision must be between 0 and 1.'):
        qim3d.mesh.from_volume(volume, mesh_precision=1.1)


def test_from_volume_empty_array():
    """Test how from_volume handles an empty 3D array."""
    volume = np.empty((0, 0, 0))  # Empty 3D array
    with pytest.raises(
        ValueError
    ):  # It should fail because it doesn't make sense to generate a mesh from empty data
        qim3d.mesh.from_volume(volume)


def test_from_volume_with_kwargs():
    """Test that from_volume correctly passes kwargs."""
    volume = np.random.rand(50, 50, 50).astype(np.float32)

    # Mock volumetric_isocontour to check if kwargs are passed
    def mock_volumetric_isocontour(vol, **kwargs):
        assert 'isovalue' in kwargs
        assert kwargs['isovalue'] == 0.5
        return hmesh.Manifold()

    # Replace the function temporarily
    original_function = hmesh.volumetric_isocontour
    hmesh.volumetric_isocontour = mock_volumetric_isocontour

    try:
        qim3d.mesh.from_volume(volume, isovalue=0.5)
    finally:
        hmesh.volumetric_isocontour = original_function  # Restore original function

def test_from_volume_pyvista_return_pygel3D():
    volume = np.random.rand(50, 50, 50).astype(np.float32)
    mesh = qim3d.mesh.from_volume(volume, backend='pyvista', return_pygel3D=True)
    assert isinstance(mesh, hmesh.Manifold)

def test_from_volume_pyvista_return_pyvista():
    volume = np.random.rand(50, 50, 50).astype(np.float32)
    mesh = qim3d.mesh.from_volume(volume, backend='pyvista', return_pygel3D=False)
    assert isinstance(mesh, qim3d.mesh.SurfaceMesh)

