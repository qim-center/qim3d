import numpy as np
import qim3d.processing
from qim3d.utils._logger import log
import trimesh
import qim3d


def volume(obj: np.ndarray|trimesh.Trimesh, 
           **mesh_kwargs
           ) -> float:
    """
    Compute the volume of a 3D volume or mesh.

    Args:
        obj (np.ndarray or trimesh.Trimesh): Either a np.ndarray volume or a mesh object of type trimesh.Trimesh.
        **mesh_kwargs (Any): Additional arguments for mesh creation if the input is a volume.

    Returns:
        volume (float): The volume of the object.

    Example:
        Compute volume from a mesh:
        ```python
        import qim3d

        # Load a mesh from a file
        mesh = qim3d.io.load_mesh('path/to/mesh.obj')

        # Compute the volume of the mesh
        vol = qim3d.features.volume(mesh)
        print('Volume:', vol)
        ```

        Compute volume from a np.ndarray:
        ```python
        import qim3d

        # Generate a 3D blob
        synthetic_blob = qim3d.generate.noise_object(noise_scale = 0.015)
        synthetic_blob = qim3d.generate.noise_object(noise_scale = 0.015)

        # Compute the volume of the blob
        volume = qim3d.features.volume(synthetic_blob, level=0.5)
        volume = qim3d.features.volume(synthetic_blob, level=0.5)
        print('Volume:', volume)
        ```

    """
    if isinstance(obj, np.ndarray):
        log.info("Converting volume to mesh.")
        obj = qim3d.mesh.from_volume(obj, **mesh_kwargs)

    return obj.volume


def area(obj: np.ndarray|trimesh.Trimesh, 
         **mesh_kwargs
         ) -> float:
    """
    Compute the surface area of a 3D volume or mesh.

    Args:
        obj (np.ndarray or trimesh.Trimesh): Either a np.ndarray volume or a mesh object of type trimesh.Trimesh.
        **mesh_kwargs (Any): Additional arguments for mesh creation if the input is a volume.

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
        area = qim3d.features.area(mesh)
        print(f"Area: {area}")
        ```

        Compute area from a np.ndarray:
        ```python
        import qim3d

        # Generate a 3D blob
        synthetic_blob = qim3d.generate.noise_object(noise_scale = 0.015)
        synthetic_blob = qim3d.generate.noise_object(noise_scale = 0.015)

        # Compute the surface area of the blob
        volume = qim3d.features.area(synthetic_blob, level=0.5)
        volume = qim3d.features.area(synthetic_blob, level=0.5)
        print('Area:', volume)
        ```
    """
    if isinstance(obj, np.ndarray):
        log.info("Converting volume to mesh.")
        obj = qim3d.mesh.from_volume(obj, **mesh_kwargs)
        obj = qim3d.mesh.from_volume(obj, **mesh_kwargs)

    return obj.area


def sphericity(obj: np.ndarray|trimesh.Trimesh, 
               **mesh_kwargs
               ) -> float:
    """
    Compute the sphericity of a 3D volume or mesh.

    Sphericity is a measure of how spherical an object is. It is defined as the ratio
    of the surface area of a sphere with the same volume as the object to the object's
    actual surface area.

    Args:
        obj (np.ndarray or trimesh.Trimesh): Either a np.ndarray volume or a mesh object of type trimesh.Trimesh.
        **mesh_kwargs (Any): Additional arguments for mesh creation if the input is a volume.

    Returns:
        sphericity (float): A float value representing the sphericity of the object.

    Example:
        Compute sphericity from a mesh:
        ```python
        import qim3d

        # Load a mesh from a file
        mesh = qim3d.io.load_mesh('path/to/mesh.obj')

        # Compute the sphericity of the mesh
        sphericity = qim3d.features.sphericity(mesh)
        sphericity = qim3d.features.sphericity(mesh)
        ```

        Compute sphericity from a np.ndarray:
        ```python
        import qim3d

        # Generate a 3D blob
        synthetic_blob = qim3d.generate.noise_object(noise_scale = 0.015)
        synthetic_blob = qim3d.generate.noise_object(noise_scale = 0.015)

        # Compute the sphericity of the blob
        sphericity = qim3d.features.sphericity(synthetic_blob, level=0.5)
        sphericity = qim3d.features.sphericity(synthetic_blob, level=0.5)
        ```

    !!! info "Limitations due to pixelation"
        Sphericity is particularly sensitive to the resolution of the mesh, as it directly impacts the accuracy of surface area and volume calculations.
        Since the mesh is generated from voxel-based 3D volume data, the discrete nature of the voxels leads to pixelation effects that reduce the precision of sphericity measurements.
        Higher resolution meshes may mitigate these errors but often at the cost of increased computational demands.
    """
    if isinstance(obj, np.ndarray):
        log.info("Converting volume to mesh.")
        obj = qim3d.mesh.from_volume(obj, **mesh_kwargs)
        obj = qim3d.mesh.from_volume(obj, **mesh_kwargs)

    volume = qim3d.features.volume(obj)
    area = qim3d.features.area(obj)
    volume = qim3d.features.volume(obj)
    area = qim3d.features.area(obj)

    if area == 0:
        log.warning("Surface area is zero, sphericity is undefined.")
        return np.nan

    sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / area
    log.info(f"Sphericity: {sphericity}")
    return sphericity
