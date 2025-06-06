import numpy as np
from pygel3d import hmesh

import qim3d
from qim3d.utils._logger import log


def volume(obj: np.ndarray | hmesh.Manifold) -> float:
    """
    Compute the volume of a 3D mesh using the Pygel3D library.

    Args:
        obj (numpy.ndarray or pygel3d.hmesh.Manifold): Either a np.ndarray volume or a mesh object of type pygel3d.hmesh.Manifold.

    Returns:
        volume (float): The volume of the object.

    Example:
        Compute volume from a mesh:
        ```python
        import qim3d

        # Generate a synthetic 3D object
        synthetic_object = qim3d.generate.volume()

        # Convert into a mesh
        mesh = qim3d.mesh.from_volume(synthetic_object, mesh_precision=0.5)
        
        # Compute the volume of the mesh
        volume = qim3d.features.volume(mesh)
        ```

        Compute volume from a `np.ndarray`:
        ```python
        import qim3d

        # Generate a 3D object
        synthetic_object = qim3d.generate.volume()

        # Compute the volume of the object
        volume = qim3d.features.volume(synthetic_object)
        ```

    """

    if isinstance(obj, np.ndarray):
        log.info('Converting volume to mesh.')
        obj = qim3d.mesh.from_volume(obj)

    return hmesh.volume(obj)


def area(obj: np.ndarray | hmesh.Manifold) -> float:
    """
    Compute the surface area of a 3D mesh using the Pygel3D library.

    Args:
        obj (numpy.ndarray or pygel3d.hmesh.Manifold): Either a np.ndarray volume or a mesh object of type pygel3d.hmesh.Manifold.

    Returns:
        area (float): The surface area of the object.

    Example:
        Compute area from a mesh:
        ```python
        import qim3d

        # Generate a synthetic 3D object
        synthetic_object = qim3d.generate.volume()

        # Convert into a mesh
        mesh = qim3d.mesh.from_volume(synthetic_object, mesh_precision=0.5)

        # Compute the surface area of the mesh
        area = qim3d.features.area(mesh)
        ```

        Compute area from a `np.ndarray`:
        ```python
        import qim3d

        # Generate a synthetic 3D object
        synthetic_object = qim3d.generate.volume(noise_scale = 0.015)

        # Compute the surface area of the object
        area = qim3d.features.area(synthetic_object)
        ```

    """

    if isinstance(obj, np.ndarray):
        log.info('Converting volume to mesh.')
        obj = qim3d.mesh.from_volume(obj)

    return hmesh.area(obj)


def sphericity(obj: np.ndarray | hmesh.Manifold) -> float:
    """
    Compute the sphericity of a 3D mesh using the Pygel3D library.

    Args:
        obj (numpy.ndarray or pygel3d.hmesh.Manifold): Either a np.ndarray volume or a mesh object of type pygel3d.hmesh.Manifold.

    Returns:
        sphericity (float): The sphericity of the object.

    Example:
        Compute sphericity from a mesh:
        ```python
        import qim3d

        # Generate a synthetic 3D object
        synthetic_object = qim3d.generate.volume()

        # Convert into a mesh
        mesh = qim3d.mesh.from_volume(synthetic_object, mesh_precision=0.5)

        # Compute the sphericity of the mesh
        sphericity = qim3d.features.sphericity(mesh)
        ```

        Compute sphericity from a `np.ndarray`:
        ```python
        import qim3d

        # Generate a 3D object
        synthetic_object = qim3d.generate.volume(noise_scale = 0.015)

        # Compute the sphericity of the object
        sphericity = qim3d.features.sphericity(synthetic_object)
        ```
    """

    if isinstance(obj, np.ndarray):
        log.info('Converting volume to mesh.')
        obj = qim3d.mesh.from_volume(obj)

    volume = qim3d.features.volume(obj)
    area = qim3d.features.area(obj)

    if area == 0:
        log.warning('Surface area is zero, sphericity is undefined.')
        return np.nan

    sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / area
    return sphericity
