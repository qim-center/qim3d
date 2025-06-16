import numpy as np
from pygel3d import hmesh
from skimage.filters import threshold_otsu

import qim3d
from qim3d.utils._logger import log

def prepare_obj(
    obj,
    threshold="otsu",
    mask=None,
    mesh_precision=1.0,
    return_mesh=True,
):
    """
    Prepares a 3D volume or mesh for further processing by applying thresholding and masking (if specified).
    Optionally returns a mesh or a binarized volume.

    Args:
        obj (np.ndarray or hmesh.Manifold): Input 3D volume or mesh.
        threshold (float, str, or None): Threshold value, ignored if input is a mesh or already binary. Defaults to 'otsu' for Otsu's method.
        mask (np.ndarray or None): Boolean mask to apply for a region of interest in the volume. Must match the shape of the input volume. Ignored if input is a mesh.
        mesh_precision (float): Precision parameter for mesh generation.
        return_mesh (bool): If True, returns a mesh. Otherwise, returns the binarized (and masked) volume.

    Returns:
        hmesh.Manifold or np.ndarray: Mesh or binarized/masked volume, depending on `return_mesh`.
    """

    # If already a mesh, return as is
    if isinstance(obj, hmesh.Manifold):
        if threshold is not None or mask is not None:
            log.info('A mesh is provided, threshold and mask will be ignored.')
        return obj

    volume = np.asarray(obj)
    processed_volume = volume.copy()

    # Determine if volume is already binary
    is_binary = np.array_equal(np.unique(volume), [0, 1]) or np.array_equal(np.unique(volume), [False, True])

    # Apply threshold if needed 
    if not is_binary and threshold is not None:
        if threshold == 'otsu':
            threshold = threshold_otsu(volume)
            
        processed_volume = (volume > threshold).astype(np.uint8)

    else:
        if threshold is not None:
            log.info('The volume is already binarized, threshold will be ignored.')

    # Apply mask if provided (set outside mask to 0)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != processed_volume.shape:
            raise ValueError(f'Mask shape {mask.shape} must match volume shape {processed_volume.shape}.')
        
        processed_volume = np.where(mask, processed_volume, 0)

    # Return mesh or binarized volume
    if return_mesh:
        mesh = qim3d.mesh.from_volume(processed_volume, mesh_precision=mesh_precision)
        return mesh
    
    return processed_volume

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

        # Generate a synthetic 3D object
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

def mean_std_intensity(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Compute the mean and standard deviation of intensities in a 3D volume. The background is ignored, and a mask can be applied to focus on a specific region of interest.

    Args:
        volume (numpy.ndarray): Input 3D volume.
        mask (numpy.ndarray or None): Boolean mask to apply for a region of interest in the volume. Must match the shape of the input volume. Defaults to None. 

    Returns:
        tuple: Mean and standard deviation of intensities.

    Example:
        ```python
        import qim3d

        # Generate a synthetic 3D object
        synthetic_object = qim3d.generate.volume()

        # Compute mean and standard deviation of intensities
        mean_intensity, std_intensity = qim3d.features.mean_std_intensity(synthetic_object)
        ```
    """

    # Mask the volume (if provided)
    volume = prepare_obj(volume, threshold=None, mask=mask, return_mesh=False)

    # Get only the non-zero intensities (i.e., ignoring the background)
    intensities = volume[volume > 0]
 
    # Compute mean and standard deviation
    mean_intensity = np.mean(intensities)
    std_intensity = np.std(intensities)

    return mean_intensity, std_intensity