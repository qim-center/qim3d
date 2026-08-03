import numpy as np
import pytest
import scipy
from pygel3d import hmesh

import qim3d

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

def test_from_volume_pyvista_return_surfacemesh():
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    volume[6:14, 6:14, 6:14] = 1.0

    mesh = qim3d.mesh.from_volume(
        volume,
        backend='pyvista',
        method='marching_cubes',
        isovalue=0.5,
        return_pygel3d=False,
    )

    assert isinstance(mesh, qim3d.mesh.SurfaceMesh)
    assert mesh.n_points > 0
    assert mesh.n_faces_strict > 0


def test_from_volume_pyvista_return_pygel3d():
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    volume[6:14, 6:14, 6:14] = 1.0

    mesh = qim3d.mesh.from_volume(
        volume,
        backend='pyvista',
        method='marching_cubes',
        isovalue=0.5,
        return_pygel3d=True,
    )

    assert isinstance(mesh, hmesh.Manifold)
    assert len(list(mesh.vertices())) > 0
    assert len(list(mesh.faces())) > 0


def test_from_volume_pygel_return_pygel3d():
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    volume[6:14, 6:14, 6:14] = 1.0

    mesh = qim3d.mesh.from_volume(
        volume,
        backend='pygel',
        isovalue=0.5,
        return_pygel3d=True,
    )

    assert isinstance(mesh, hmesh.Manifold)
    assert len(list(mesh.vertices())) > 0
    assert len(list(mesh.faces())) > 0


def test_from_volume_pygel_return_surfacemesh():
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    volume[6:14, 6:14, 6:14] = 1.0

    mesh = qim3d.mesh.from_volume(
        volume,
        backend='pygel',
        isovalue=0.5,
        return_pygel3d=False,
    )

    assert isinstance(mesh, qim3d.mesh.SurfaceMesh)
    assert mesh.n_points > 0
    assert mesh.n_faces_strict > 0

def test_from_volume_pyvista_to_pygel3d_conversion_counts_match():
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    volume[6:14, 6:14, 6:14] = 1.0

    pv_mesh = qim3d.mesh.from_volume(
        volume,
        backend='pyvista',
        method='marching_cubes',
        isovalue=0.5,
        return_pygel3d=False,
    ).triangulate()

    pygel_mesh = qim3d.mesh.from_volume(
        volume,
        backend='pyvista',
        method='marching_cubes',
        isovalue=0.5,
        return_pygel3d=True,
    )

    assert len(list(pygel_mesh.vertices())) == pv_mesh.n_points
    assert len(list(pygel_mesh.faces())) == pv_mesh.n_faces_strict

def test_from_volume_pygel_to_surfacemesh_conversion_counts_match():
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    volume[6:14, 6:14, 6:14] = 1.0

    pygel_mesh = qim3d.mesh.from_volume(
        volume,
        backend='pygel',
        isovalue=0.5,
        return_pygel3d=True,
    )

    surface_mesh = qim3d.mesh.from_volume(
        volume,
        backend='pygel',
        isovalue=0.5,
        return_pygel3d=False,
    )

    assert surface_mesh.n_points == len(list(pygel_mesh.vertices()))
    assert surface_mesh.n_faces_strict == len(list(pygel_mesh.faces()))