import numpy as np
import qim3d
from qim3d.utils._logger import log
import qim3d
from pygel3d import hmesh

def volume(obj: np.ndarray|hmesh.Manifold) -> float:
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

        # Load a mesh from a file
        mesh = qim3d.io.load_mesh('path/to/mesh.obj')

        # Compute the volume of the mesh
        volume = qim3d.features.volume(mesh)
        print(f'Volume: {volume}')
        ```

        Compute volume from a np.ndarray:
        ```python
        import qim3d

        # Generate a 3D blob
        synthetic_blob = qim3d.generate.noise_object(noise_scale = 0.015)

        # Compute the volume of the blob
        volume = qim3d.features.volume(synthetic_blob)
        print(f'Volume: {volume}')
        ```

    """

    if isinstance(obj, np.ndarray):
        log.info("Converting volume to mesh.")
        obj = qim3d.mesh.from_volume(obj)

    return hmesh.volume(obj)

def area(obj: np.ndarray|hmesh.Manifold) -> float:
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

        # Load a mesh from a file
        mesh = qim3d.io.load_mesh('path/to/mesh.obj')

        # Compute the surface area of the mesh
        area = qim3d.features.area(mesh)
        print(f'Area: {area}')
        ```

        Compute area from a np.ndarray:
        ```python
        import qim3d

        # Generate a 3D blob
        synthetic_blob = qim3d.generate.noise_object(noise_scale = 0.015)

        # Compute the surface area of the blob
        area = qim3d.features.area(synthetic_blob)
        print(f'Area: {area}')
        ```
    
    """

    if isinstance(obj, np.ndarray):
        log.info("Converting volume to mesh.")
        obj = qim3d.mesh.from_volume(obj)

    return hmesh.area(obj)

def sphericity(obj: np.ndarray|hmesh.Manifold) -> float:
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

        # Load a mesh from a file
        mesh = qim3d.io.load_mesh('path/to/mesh.obj')

        # Compute the sphericity of the mesh
        sphericity = qim3d.features.sphericity(mesh)
        print(f'Sphericity: {sphericity}')
        ```

        Compute sphericity from a np.ndarray:
        ```python
        import qim3d

        # Generate a 3D blob
        synthetic_blob = qim3d.generate.noise_object(noise_scale = 0.015)

        # Compute the sphericity of the blob
        sphericity = qim3d.features.sphericity(synthetic_blob)
        print(f'Sphericity: {sphericity}')
        ```

    """

    if isinstance(obj, np.ndarray):
        log.info("Converting volume to mesh.")
        obj = qim3d.mesh.from_volume(obj)

    volume = qim3d.features.volume(obj)
    area = qim3d.features.area(obj)

    if area == 0:
        log.warning("Surface area is zero, sphericity is undefined.")
        return np.nan

    sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / area
    # log.info(f"Sphericity: {sphericity}")
    return sphericity