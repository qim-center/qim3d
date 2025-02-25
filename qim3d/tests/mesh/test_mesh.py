import pytest
import numpy as np
from pygel3d import hmesh
import qim3d

def test_from_volume_valid_input():
    """Test that from_volume returns a hmesh.Manifold object for a valid 3D input."""
    volume = np.random.rand(50, 50, 50).astype(np.float32)  # Generate a random 3D volume
    mesh = qim3d.mesh.from_volume(volume)
    assert isinstance(mesh, hmesh.Manifold)  # Check if output is a Manifold object

def test_from_volume_invalid_input():
    """Test that from_volume raises ValueError for non-3D input."""
    volume = np.random.rand(50, 50)  # A 2D array
    with pytest.raises(ValueError, match="The input volume must be a 3D numpy array."):
        qim3d.mesh.from_volume(volume)

def test_from_volume_empty_array():
    """Test how from_volume handles an empty 3D array."""
    volume = np.empty((0, 0, 0))  # Empty 3D array
    with pytest.raises(ValueError):  # It should fail because it doesn't make sense to generate a mesh from empty data
        qim3d.mesh.from_volume(volume)

def test_from_volume_with_kwargs():
    """Test that from_volume correctly passes kwargs."""
    volume = np.random.rand(50, 50, 50).astype(np.float32)

    # Mock volumetric_isocontour to check if kwargs are passed
    def mock_volumetric_isocontour(vol, **kwargs):
        assert "isovalue" in kwargs
        assert kwargs["isovalue"] == 0.5
        return hmesh.Manifold()

    # Replace the function temporarily
    original_function = hmesh.volumetric_isocontour
    hmesh.volumetric_isocontour = mock_volumetric_isocontour

    try:
        qim3d.mesh.from_volume(volume, isovalue=0.5)
    finally:
        hmesh.volumetric_isocontour = original_function  # Restore original function

