import numpy as np
import qim3d.processing
from qim3d.utils.logger import log
import trimesh
import qim3d


def volume(obj, **mesh_kwargs) -> float:
    """
    Compute the volume of a 3D volume or mesh.

    Args:
        obj: Either a np.ndarray volume or a mesh object of type trimesh.Trimesh.
        **mesh_kwargs: Additional arguments for mesh creation if the input is a volume.

    Returns:
        volume: The volume of the object.

    Example:
        Compute volume from a mesh:
        ```python
        import qim3d

        # Load a mesh from a file
        mesh = qim3d.io.load_mesh('path/to/mesh.obj')

        # Compute the volume of the mesh
        volume = qim3d.processing.volume(mesh)
        print('Volume:', volume)
        ```

        Compute volume from a np.ndarray:
        ```python
        import qim3d

        # Generate a 3D blob
        synthetic_blob = qim3d.generate.blob(noise_scale = 0.015)

        # Compute the volume of the blob
        volume = qim3d.processing.volume(synthetic_blob, level=0.5)
        print('Volume:', volume)
        ```

    """
    if isinstance(obj, np.ndarray):
        log.info("Converting volume to mesh.")
        obj = qim3d.processing.create_mesh(obj, **mesh_kwargs)

    return obj.volume


def area(obj, **mesh_kwargs) -> float:
    """
    Compute the surface area of a 3D volume or mesh.

    Args:
        obj: Either a np.ndarray volume or a mesh object of type trimesh.Trimesh.
        **mesh_kwargs: Additional arguments for mesh creation if the input is a volume.

    Returns:
        area: The surface area of the object.

    Example:
        Compute area from a mesh:
        ```python
        import qim3d

        # Load a mesh from a file
        mesh = qim3d.io.load_mesh('path/to/mesh.obj')

        # Compute the surface area of the mesh
        area = qim3d.processing.area(mesh)
        print(f"Area: {area}")
        ```

        Compute area from a np.ndarray:
        ```python
        import qim3d

        # Generate a 3D blob
        synthetic_blob = qim3d.generate.blob(noise_scale = 0.015)

        # Compute the surface area of the blob
        volume = qim3d.processing.area(synthetic_blob, level=0.5)
        print('Area:', volume)
        ```
    """
    if isinstance(obj, np.ndarray):
        log.info("Converting volume to mesh.")
        obj = qim3d.processing.create_mesh(obj, **mesh_kwargs)

    return obj.area


def sphericity(obj, **mesh_kwargs) -> float:
    """
    Compute the sphericity of a 3D volume or mesh.

    Sphericity is a measure of how spherical an object is. It is defined as the ratio
    of the surface area of a sphere with the same volume as the object to the object's
    actual surface area.

    Args:
        obj: Either a np.ndarray volume or a mesh object of type trimesh.Trimesh.
        **mesh_kwargs: Additional arguments for mesh creation if the input is a volume.

    Returns:
        sphericity: A float value representing the sphericity of the object.

    Example:
        Compute sphericity from a mesh:
        ```python
        import qim3d

        # Load a mesh from a file
        mesh = qim3d.io.load_mesh('path/to/mesh.obj')

        # Compute the sphericity of the mesh
        sphericity = qim3d.processing.sphericity(mesh)
        ```

        Compute sphericity from a np.ndarray:
        ```python
        import qim3d

        # Generate a 3D blob
        synthetic_blob = qim3d.generate.blob(noise_scale = 0.015)

        # Compute the sphericity of the blob
        sphericity = qim3d.processing.sphericity(synthetic_blob, level=0.5)
        ```

    !!! info "Limitations due to pixelation"
        Sphericity is particularly sensitive to the resolution of the mesh, as it directly impacts the accuracy of surface area and volume calculations.
        Since the mesh is generated from voxel-based 3D volume data, the discrete nature of the voxels leads to pixelation effects that reduce the precision of sphericity measurements.
        Higher resolution meshes may mitigate these errors but often at the cost of increased computational demands.
    """
    if isinstance(obj, np.ndarray):
        log.info("Converting volume to mesh.")
        obj = qim3d.processing.create_mesh(obj, **mesh_kwargs)

    volume = qim3d.processing.volume(obj)
    area = qim3d.processing.area(obj)

    if area == 0:
        log.warning("Surface area is zero, sphericity is undefined.")
        return np.nan

    sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / area
    log.info(f"Sphericity: {sphericity}")
    return sphericity
