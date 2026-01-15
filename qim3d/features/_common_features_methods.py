import numpy as np
from pygel3d import hmesh
from skimage.filters import threshold_otsu

import qim3d
from qim3d.utils._logger import log


def prepare_obj(
    obj: np.ndarray | hmesh.Manifold,
    mask: np.ndarray | None = None,
    threshold: float | str = 'otsu',
    mesh_precision: float = 1.0,
    return_mesh: bool = True,
) -> np.ndarray | hmesh.Manifold:
    """
    Prepares a volume or mesh for feature extraction by applying thresholding and masking (if specified).
    Optionally returns a mesh or a binarized volume.

    Args:
        obj (np.ndarray or hmesh.Manifold): Input `np.ndarray` volume or a mesh object of type `pygel3d.hmesh.Manifold`.
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
    is_binary = np.array_equal(np.unique(volume), [0, 1]) or np.array_equal(
        np.unique(volume), [False, True]
    )

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
            msg = f'Mask shape {mask.shape} must match volume shape {processed_volume.shape}.'
            raise ValueError(msg)

        processed_volume = np.where(mask, processed_volume, 0)

    # Return mesh or binarized volume
    if return_mesh:
        mesh = qim3d.mesh.from_volume(processed_volume, mesh_precision=mesh_precision)
        return mesh

    return processed_volume


def volume(
    object: np.ndarray | hmesh.Manifold,
    mask: np.ndarray | None = None,
    threshold: float | str = 'otsu',
) -> float:
    """
    Computes the enclosed physical volume of a 3D object.

    This function quantifies the amount of space occupied by a structure. It is robust and versatile, handling both raw 3D image arrays (voxels) and pre-generated meshes. If a voxel array is provided, it is automatically converted into a mesh using the specified threshold before the volume is calculated, ensuring sub-voxel accuracy compared to simple voxel counting.

    Args:
        object (np.ndarray or hmesh.Manifold): The input data. Can be a 3D NumPy array (volume) or a `pygel3d.hmesh.Manifold` object (mesh).
        mask (np.ndarray, optional): A boolean mask defining a Region of Interest (ROI). Only the volume within this mask is included. Must have the same shape as `object`. Defaults to `None`.
        threshold (float or str, optional): The intensity value used to binarize the volume if a NumPy array is provided.

            * **float**: A specific intensity value.
            * **'otsu'**: Automatically calculates the threshold using Otsu's method. Defaults to 'otsu'.

    Returns:
        volume (float):
            The calculated volume in cubic units (determined by the input resolution).

    Raises:
        ValueError: If `mask` is provided but its shape does not match the input `object`.

    Example:
        ```python
        import qim3d
        import numpy as np

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume(noise_scale=0.01, final_shape=(100, 100, 100))

        # Create a mask for the bottom right corner
        mask = np.zeros_like(synthetic_object, dtype=bool)
        mask[50:100, 50:100, 50:100] = True

        # Compute the volume of the object within the region of interest defined by the mask
        volume = qim3d.features.volume(synthetic_object, threshold=50, mask=mask)
        print(volume)
        ```
        48774.99
    """
    # Prepare object
    mesh = prepare_obj(object, threshold=threshold, mask=mask, return_mesh=True)

    # Compute volume
    volume = hmesh.volume(mesh)

    return volume


def area(
    object: np.ndarray | hmesh.Manifold,
    mask: np.ndarray | None = None,
    threshold: float | str = 'otsu',
) -> float:
    """
    Calculates the total surface area of a 3D object.

    This function quantifies the surface size of a structure, which is a fundamental metric in morphometry and material science. It accepts either a raw 3D volume (which is automatically meshed using Marching Cubes) or a pre-computed `PyGEL3D` mesh.

    Args:
        object (np.ndarray or hmesh.Manifold): The input data. Can be a 3D NumPy array (volume) or a `pygel3d.hmesh.Manifold` object (mesh).
        mask (np.ndarray, optional): A boolean mask defining a Region of Interest (ROI) within the volume. Must have the same shape as `object`. Defaults to `None`.
        threshold (float or str, optional): The intensity value used to binarize the volume if a NumPy array is provided.
            
            * **float**: A specific intensity value.
            * **'otsu'**: Automatically calculates the threshold using Otsu's method. Defaults to 'otsu'.

    Returns:
        area (float):
            The computed surface area in square units (determined by the input resolution).

    Raises:
        ValueError: If `mask` is provided but its shape does not match the input `object`.

    Example:
        Compute area from a `np.ndarray` volume:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume()

        # Compute the surface area of the object
        area = qim3d.features.area(synthetic_object)
        print(area)
        ```
        58535.06

    Example:
        Compute area from a `pygel3d.hmesh.Manifold` mesh:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume()

        # Convert into a mesh
        mesh = qim3d.mesh.from_volume(synthetic_object)

        # Compute the surface area of the object
        area = qim3d.features.area(mesh)
        print(area)
        ```
        58535.06
    """
    # Prepare object
    mesh = prepare_obj(object, threshold=threshold, mask=mask, return_mesh=True)

    # Compute area
    area = hmesh.area(mesh)

    return area


def sphericity(
    object: np.ndarray | hmesh.Manifold,
    mask: np.ndarray | None = None,
    threshold: float | str = 'otsu',
) -> float:
    """
    Computes the sphericity (compactness) of a 3D object.

    Sphericity is a measure of how closely the shape of an object resembles a perfect sphere. It is defined as the ratio of the surface area of a sphere (with the same volume as the given object) to the actual surface area of the object. This morphological descriptor is widely used in particle analysis, geology, and biology to quantify roundness and regularity.

    * **1.0**: Indicates a perfect sphere.
    * **< 1.0**: Indicates irregular, elongated, or rough shapes (e.g., a cube has a sphericity of approx. 0.806).

    Args:
        object (np.ndarray or hmesh.Manifold): The input data. Can be a 3D NumPy array (volume) or a `pygel3d.hmesh.Manifold` object (mesh).
        mask (np.ndarray, optional): A boolean mask defining a Region of Interest (ROI). Only the data within this mask is considered. Must have the same shape as `object`. Defaults to `None`.
        threshold (float or str, optional): The intensity value used to binarize the volume if a NumPy array is provided.

            * **float**: A specific intensity value.
            * **'otsu'**: Automatically calculates the threshold using Otsu's method. Defaults to 'otsu'.

    Returns:
        sphericity (float):
            The calculated sphericity value (unitless, max 1.0). Returns `nan` if the object has zero volume or area.

    Raises:
        ValueError: If `mask` is provided but its shape does not match the input `object`.

    Example:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume(noise_scale=0.005)

        # Compute the sphericity of the object
        sphericity = qim3d.features.sphericity(synthetic_object)
        print(f"Sphericity: {sphericity:.4f}")

        # Visualize the synthetic object
        qim3d.viz.volumetric(synthetic_object)
        ```
        Sphericity: 0.9058
        <iframe src="https://platform.qim.dk/k3d/sphericity_feature_example_2.html" width="100%" height="500" frameborder="0"></iframe>

    Example:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume(noise_scale=0.008)

        # Manipulate the object
        synthetic_object = qim3d.operations.stretch(synthetic_object, z_stretch=50)
        synthetic_object = qim3d.operations.curve_warp(synthetic_object, x_amp=10, x_periods=4)

        # Compute the sphericity of the object
        sphericity = qim3d.features.sphericity(synthetic_object)
        print(f"Sphericity: {sphericity:.4f}")

        # Visualize the synthetic object
        qim3d.viz.volumetric(synthetic_object)
        ```
        Sphericity: 0.6876
        <iframe src="https://platform.qim.dk/k3d/sphericity_feature_example_1.html" width="100%" height="500" frameborder="0"></iframe>
    """
    # Prepare object
    mesh = prepare_obj(object, threshold=threshold, mask=mask, return_mesh=True)

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
    Calculates the mean (average) and standard deviation of voxel intensities within a volume.

    This function computes global image statistics, which are fundamental for tasks like data normalization (z-score), contrast analysis, and quality control. By default, it ignores zero-valued voxels (background) to ensure the statistics reflect the actual object or signal rather than the empty space around it. You can further restrict the calculation to a specific Region of Interest (ROI) using a binary mask.

    Args:
        volume (np.ndarray): The input 3D image stack (voxel data).
        mask (np.ndarray, optional): A boolean mask defining a specific region to analyze. Only voxels where `mask=True` are included. Must match the shape of `volume`. Defaults to `None`.

    Returns:
        stats (tuple[float, float]):
            A tuple containing `(mean, std_dev)`.

            * **mean**: The average intensity value of the non-zero voxels.
            * **std_dev**: The standard deviation of the intensity values.

    Raises:
        ValueError: If `mask` is provided but does not match the shape of `volume`.

    Note:
        This function automatically filters out background values (intensities equal to 0) before calculation.

    Example:
        ```python
        import qim3d

        # Load a sample object
        shell_object = qim3d.examples.shell_225x128x128

        # Compute mean and standard deviation of intensities in the object
        mean_intensity, std_intensity = qim3d.features.mean_std_intensity(shell_object)
        print(f"Mean intensity: {mean_intensity:.4f}")
        print(f"Standard deviation of intensity: {std_intensity:.4f}")

        # Visualize slices of the object
        qim3d.viz.slices_grid(shell_object, color_bar=True, color_bar_style="large")
        ```
        Mean intensity: 114.6734  
        Standard deviation of intensity: 45.8481
        ![mean_std_intensity_feature](../../assets/screenshots/mean_std_intensity_feature_example.png)
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
    object: np.ndarray | hmesh.Manifold,
    mask: np.ndarray | None = None,
    threshold: float | str = 'otsu',
) -> float:
    """
    Calculates the maximum dimension (size) of the object's bounding box.

    This function determines the largest spatial extent of the object along the Cartesian axes (X, Y, Z). It computes the Axis-Aligned Bounding Box (AABB) enclosing the structure and returns the length of its longest side. This metric is useful for characterizing the overall scale, fitting constraints, or the maximum caliper diameter along orthogonal axes.

    Args:
        object (np.ndarray or hmesh.Manifold): The input data. Can be a 3D NumPy array (volume) or a `pygel3d.hmesh.Manifold` object (mesh).
        mask (np.ndarray, optional): A boolean mask defining a Region of Interest (ROI). Only the data within this mask is considered. Must have the same shape as `object`. Defaults to `None`.
        threshold (float or str, optional): The intensity value used to binarize the volume if a NumPy array is provided.

            * **float**: A specific intensity value.
            * **'otsu'**: Automatically calculates the threshold using Otsu's method. Defaults to 'otsu'.

    Returns:
        size (float):
            The length of the largest side of the enclosing bounding box.

    Raises:
        ValueError: If `mask` is provided but its shape does not match the input `object`.

    Example:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume(
            final_shape=(100,30,30),
            noise_scale=0.008,
            shape="cylinder"
            )

        # Compute size of the object
        size = qim3d.features.size(synthetic_object)
        print(f"Size: {size}")

        # Visualize the synthetic object
        qim3d.viz.volumetric(synthetic_object)
        ```
        Size: 100.0
        <iframe src="https://platform.qim.dk/k3d/size_feature_example.html" width="100%" height="500" frameborder="0"></iframe>
    """
    # Prepare object
    mesh = prepare_obj(object, threshold=threshold, mask=mask, return_mesh=True)

    # Min and max corners of the bounding box
    bbox = hmesh.bbox(mesh)
    mins, maxs = bbox

    # Maximum side length of the bounding box
    side_lengths = maxs - mins
    size = np.max(side_lengths)

    return size


def roughness(
    object: np.ndarray | hmesh.Manifold,
    mask: np.ndarray | None = None,
    threshold: float | str = 'otsu',
) -> float:
    """
    Computes the roughness (Surface-Area-to-Volume ratio) of a 3D object.

    This feature quantifies the morphological complexity of an object. Unlike a simple measure of size, this ratio indicates how "folded", "spiky", or "porous" a structure is relative to its mass. A perfect sphere has the lowest possible surface-to-volume ratio (smoothest), while highly complex shapes like fractals, trabecular bone, or dendrites will have a much higher value. This metric is often used to analyze texture and surface irregularity.

    Args:
        object (np.ndarray or hmesh.Manifold): The input data. Can be a 3D NumPy array (volume) or a `pygel3d.hmesh.Manifold` object (mesh).
        mask (np.ndarray, optional): A boolean mask defining a Region of Interest (ROI). Only the data within this mask is considered. Must have the same shape as `object`. Defaults to `None`.
        threshold (float or str, optional): The intensity value used to binarize the volume if a NumPy array is provided.

            * **float**: A specific intensity value.
            * **'otsu'**: Automatically calculates the threshold using Otsu's method. Defaults to 'otsu'.

    Returns:
        roughness (float):
            The calculated ratio of Surface Area / Volume. Returns `nan` if the object has zero volume or area.

    Raises:
        ValueError: If `mask` is provided but its shape does not match the input `object`.

    Example:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume(
            base_shape=(128,128,128),
            noise_scale=0.019,
            )

        # Compute the roughness of the object
        roughness = qim3d.features.roughness(synthetic_object)
        print(f"Roughness: {roughness:.4f}")

        # Visualize the synthetic object
        qim3d.viz.volumetric(synthetic_object)
        ```
        Roughness: 0.1005
        <iframe src="https://platform.qim.dk/k3d/roughness_feature_example_v1.html" width="100%" height="500" frameborder="0"></iframe>

    Example:
        ```python
        import qim3d

        # Generate a synthetic object
        synthetic_object = qim3d.generate.volume(
            base_shape=(128,128,128),
            noise_scale=0.08,
            decay_rate=18,
            gamma=0.9,
            )

        # Compute the roughness of the object
        roughness = qim3d.features.roughness(synthetic_object)
        print(f"Roughness: {roughness:.4f}")

        # Visualize the synthetic object
        qim3d.viz.volumetric(synthetic_object)
        ```
        Roughness: 0.2534
        <iframe src="https://platform.qim.dk/k3d/roughness_feature_example_v2.html" width="100%" height="500" frameborder="0"></iframe>
    """
    # Prepare object
    mesh = prepare_obj(object, threshold=threshold, mask=mask, return_mesh=True)

    # Compute surface area and volume
    area = qim3d.features.area(mesh)
    volume = qim3d.features.volume(mesh)

    if area == 0 or volume == 0:
        log.warning('Surface area or volume is zero, roughness is undefined.')
        return np.nan

    # Compute roughness
    roughness = area / volume

    return roughness
