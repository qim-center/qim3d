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
    Prepares a volume or mesh for feature extraction by applying thresholding and masking (if specified).
    Optionally returns a mesh or a binarized volume.

    Args:
        obj (np.ndarray or hmesh.Manifold): Input np.ndarray volume or a mesh object of type pygel3d.hmesh.Manifold.
        threshold (float, str, or None): Threshold value, ignored if input is a mesh or volume is already binary. Defaults to 'otsu' for Otsu's method.
        mask (np.ndarray or None): Boolean mask to apply for a region of interest in the volume. Must match the shape of the input volume. Ignored if input is a mesh.
        mesh_precision (float): Precision parameter for mesh generation.
        return_mesh (bool): If True, returns a mesh. Otherwise, returns the binarized and/or masked volume.

    Returns:
        hmesh.Manifold or np.ndarray: Mesh or binarized/masked volume, depending on `return_mesh`.
    """

    # If already a mesh, return as is
    if isinstance(obj, hmesh.Manifold):
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

    # Apply mask if provided (set voxels outside of mask to 0)
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

def volume(
    obj: np.ndarray | hmesh.Manifold,
    mask: np.ndarray | None = None,
    threshold: float | str = "otsu",
    ) -> float:
    """
    Compute the volume of an object from a volume or mesh using the Pygel3D library.

    Args:
        obj (np.ndarray or hmesh.Manifold): Input np.ndarray volume or a mesh object of type pygel3d.hmesh.Manifold.
        mask (numpy.ndarray or None): Boolean mask to apply for a region of interest in the volume. Must match the shape of the input volume. Defaults to None.
        threshold (float, str): Threshold value for binarization of the input volume. If 'otsu', Otsu's method is used. Defaults to 'otsu'.

    Returns:
        volume (float): The volume of the object.

    Example:
        Compute volume from a mesh:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume()

        # Convert into a mesh
        mesh = qim3d.mesh.from_volume(synthetic_object, mesh_precision=0.5)
        
        # Compute the volume of the mesh
        volume = qim3d.features.volume(mesh)
        ```

        Compute volume from a `np.ndarray`:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume()

        # Compute the volume of the object
        volume = qim3d.features.volume(synthetic_object)
        ```

    """
    # Prepare object
    mesh = prepare_obj(obj, threshold=threshold, mask=mask, return_mesh=True)

    # Compute volume
    volume = hmesh.volume(mesh)

    return volume


def area(
    obj: np.ndarray | hmesh.Manifold,
    mask: np.ndarray | None = None,
    threshold: float | str = "otsu",
    ) -> float:
    """
    Compute the surface area of an object from a volume or mesh using the Pygel3D library.

    Args:
        obj (np.ndarray or hmesh.Manifold): Input np.ndarray volume or a mesh object of type pygel3d.hmesh.Manifold.
        mask (numpy.ndarray or None): Boolean mask to apply for a region of interest in the volume. Must match the shape of the input volume. Defaults to None.
        threshold (float, str): Threshold value for binarization of the input volume. If 'otsu', Otsu's method is used. Defaults to 'otsu'.

    Returns:
        area (float): The surface area of the object.

    Example:
        Compute area from a mesh:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume()

        # Convert into a mesh
        mesh = qim3d.mesh.from_volume(synthetic_object, mesh_precision=0.5)

        # Compute the surface area of the mesh
        area = qim3d.features.area(mesh)
        ```

        Compute area from a `np.ndarray`:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume(noise_scale = 0.015)

        # Compute the surface area of the object
        area = qim3d.features.area(synthetic_object)
        ```

    """
    # Prepare object
    mesh = prepare_obj(obj, threshold=threshold, mask=mask, return_mesh=True)

    # Compute area
    area = hmesh.area(mesh)

    return area


def sphericity(
    obj: np.ndarray | hmesh.Manifold,
    mask: np.ndarray | None = None,
    threshold: float | str = "otsu",
    ) -> float:
    """
    Compute the sphericity of an object from a volume or mesh.

    Args:
        obj (np.ndarray or hmesh.Manifold): Input np.ndarray volume or a mesh object of type pygel3d.hmesh.Manifold.
        mask (numpy.ndarray or None): Boolean mask to apply for a region of interest in the volume. Must match the shape of the input volume. Defaults to None.
        threshold (float, str): Threshold value for binarization of the input volume. If 'otsu', Otsu's method is used. Defaults to 'otsu'.

    Returns:
        sphericity (float): The sphericity of the object.

    Example:
        Compute sphericity from a `np.ndarray` volume:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume(noise_scale = 0.015)

        # Compute the sphericity of the object
        sphericity = qim3d.features.sphericity(synthetic_object)
        ```

        Compute sphericity from a `pygel3d.hmesh.Manifold` mesh:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume()

        # Convert into a mesh
        mesh = qim3d.mesh.from_volume(synthetic_object, mesh_precision=0.5)

        # Compute the sphericity of the mesh
        sphericity = qim3d.features.sphericity(mesh)
        ```
    """
    # Prepare object
    mesh = prepare_obj(obj, threshold=threshold, mask=mask, return_mesh=True)

    # Compute surface area and volume
    area = qim3d.features.area(mesh)
    volume = qim3d.features.volume(mesh)

    if area == 0 or volume == 0:
        log.warning('Surface area or volume is zero, sphericity is undefined.')
        return np.nan

    # Compute sphericity
    sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / area

    return sphericity

def mean_std_intensity(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Compute the mean and standard deviation of intensities in a volume.

    Args:
        volume (numpy.ndarray): Input np.ndarray volume.
        mask (numpy.ndarray or None): Boolean mask to apply for a region of interest in the volume. Must match the shape of the input volume. Defaults to None. 

    Returns:
        tuple: Mean and standard deviation of intensities.

    Note: 
        - The background (intensities of 0) is excluded from the computation.
        - If a mask is provided, it will only compute the mean and standard deviation for that region of interest.

    Example:
        ```python
        import qim3d

        # Generate a synthetic object
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

def size(
    obj: np.ndarray | hmesh.Manifold,
    mask: np.ndarray | None = None,
    threshold: float | str = "otsu",
) -> float:
    """
    Compute the size (maximum side length of the bounding box enclosing the object) of an object from a volume or mesh.

    Args:
        obj (np.ndarray or hmesh.Manifold): Input np.ndarray volume or a mesh object of type pygel3d.hmesh.Manifold.
        mask (numpy.ndarray or None): Boolean mask to apply for a region of interest in the volume. Must match the shape of the input volume. Defaults to None. 
        threshold (float, str): Threshold value for binarization of the input volume. If 'otsu', Otsu's method is used. Defaults to 'otsu'.
        
    Returns:
        size: The size of the object, defined as the maximum side length of the bounding box enclosing the object.

    Note: 
        - There should only be one object in the input volume or mesh.
        - If the input is a mesh, the threshold and mask are ignored, and the size is computed directly from the mesh.
        - If the input is a volume, it is binarized first using the specified threshold (or Otsu's threshold otherwise), and a mask can be applied to focus on a specific region of interest.
    
    Example:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume()

        # Compute the size of the object
        size = qim3d.features.size(synthetic_object)
        ```
    """
    # Prepare object
    mesh = prepare_obj(obj, threshold=threshold, mask=mask, return_mesh=True)

    # Min and max corners of the bounding box
    bbox = hmesh.bbox(mesh)
    mins, maxs = bbox

    # Maximum side length of the bounding box
    side_lengths = maxs - mins
    size = np.max(side_lengths)

    return size

def roughness(
    obj: np.ndarray | hmesh.Manifold,
    mask: np.ndarray | None = None,
    threshold: float | str = "otsu",
) -> float:
    """ 
    Compute the roughness (ratio between surface area and volume) of an object from a volume or mesh.

    Args:
        obj (np.ndarray or hmesh.Manifold): Input np.ndarray volume or a mesh object of type pygel3d.hmesh.Manifold.
        mask (numpy.ndarray or None): Boolean mask to apply for a region of interest in the volume. Must match the shape of the input volume. Defaults to None.
        threshold (float, str): Threshold value for binarization of the input volume. If 'otsu', Otsu's method is used. Defaults to 'otsu'.
    
    Returns:
        roughness (float): The roughness of the object, defined as the ratio between surface area and volume.

    Note: 
        - There should only be one object in the input volume or mesh.
        - If the input is a mesh, the threshold and mask are ignored, and the size is computed directly from the mesh.
        - If the input is a volume, it is binarized first using the specified threshold (or Otsu's threshold otherwise), and a mask can be applied to focus on a specific region of interest.
    
    Example:
        ```python
        import qim3d

        # Compute roughness of a synthetic object with smooth surface
        synthetic_object_1 = qim3d.generate.volume(noise_scale=0.01)
        roughness_1 = qim3d.features.roughness(synthetic_object_1)

        # Compute roughness of a synthetic object with rough surface
        synthetic_object_2 = qim3d.generate.volume(noise_scale=0.05)
        roughness_2 = qim3d.features.roughness(synthetic_object_2)
        ```
    """
    # Prepare object
    mesh = prepare_obj(obj, threshold=threshold, mask=mask, return_mesh=True)

    # Compute surface area and volume
    area = qim3d.features.area(mesh)
    volume = qim3d.features.volume(mesh)

    if area == 0 or volume == 0:
        log.warning('Surface area or volume is zero, roughness is undefined.')
        return np.nan
    
    # Compute roughness
    roughness = area / volume

    return roughness