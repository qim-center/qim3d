import numpy as np
import scipy.ndimage
from noise import pnoise3

import qim3d.processing

__all__ = ['volume']


def volume(
    base_shape: tuple = (128, 128, 128),
    final_shape: tuple = (128, 128, 128),
    noise_scale: float = 0.05,
    order: int = 1,
    gamma: int = 1.0,
    max_value: int = 255,
    threshold: float = 0.5,
    smooth_borders: bool = False,
    volume_shape: str = None,
    dtype: str = 'uint8',
) -> np.ndarray:
    """
    Generate a 3D volume with Perlin noise, spherical gradient, and optional scaling and gamma correction.

    Args:
        base_shape (tuple of ints, optional): Shape of the initial volume to generate. Defaults to (128, 128, 128).
        final_shape (tuple of ints, optional): Desired shape of the final volume. Defaults to (128, 128, 128).
        noise_scale (float, optional): Scale factor for Perlin noise. Defaults to 0.05.
        order (int, optional): Order of the spline interpolation used in resizing. Defaults to 1.
        gamma (float, optional): Gamma correction factor. Defaults to 1.0.
        max_value (int, optional): Maximum value for the volume intensity. Defaults to 255.
        threshold (float, optional): Threshold value for clipping low intensity values. Defaults to 0.5.
        smooth_borders (bool, optional): Flag for automatic computation of the threshold value to ensure a blob with no straight edges. If True, the `threshold` parameter is ignored. Defaults to False.
        volume_shape (str, optional): Shape of the volume to generate, either "cylinder", or "tube". Defaults to None.
        dtype (data-type, optional): Desired data type of the output volume. Defaults to "uint8".

    Returns:
        volume (numpy.ndarray): Generated 3D volume with specified parameters.

    Raises:
        TypeError: If `final_shape` is not a tuple or does not have three elements.
        ValueError: If `dtype` is not a valid numpy number type.

    Example:
        ```python
        import qim3d

        # Generate synthetic blob
        vol = qim3d.generate.volume(noise_scale = 0.015)

        # Visualize 3D volume
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_blob.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize slices
        qim3d.viz.slices_grid(vol, value_min = 0, value_max = 255, num_slices = 15)
        ```
        ![synthetic_blob](../../assets/screenshots/synthetic_blob_slices.png)

    Example:
        ```python
        import qim3d

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(base_shape = (10, 300, 300),
                                final_shape = (100, 100, 100),
                                noise_scale = 0.3,
                                gamma = 2,
                                threshold = 0.0,
                                volume_shape = "cylinder"
                                )

        # Visualize synthetic volume
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_blob_cylinder.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize slices
        qim3d.viz.slices_grid(vol, num_slices=15, slice_axis=1)
        ```
        ![synthetic_blob_cylinder_slice](../../assets/screenshots/synthetic_blob_cylinder_slice.png)

    Example:
        ```python
        import qim3d

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(base_shape = (200, 100, 100),
                                final_shape = (400, 100, 100),
                                noise_scale = 0.03,
                                gamma = 0.12,
                                threshold = 0.85,
                                volume_shape = "tube"
                                )

        # Visualize synthetic blob
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_blob_tube.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize
        qim3d.viz.slices_grid(vol, num_slices=15)
        ```
        ![synthetic_blob_tube_slice](../../assets/screenshots/synthetic_blob_tube_slice.png)

    """

    if not isinstance(final_shape, tuple) or len(final_shape) != 3:
        message = 'Size must be a tuple of 3 dimensions'
        raise TypeError(message)
    if not np.issubdtype(dtype, np.number):
        message = 'Invalid data type'
        raise ValueError(message)

    # Initialize the 3D array for the shape
    volume = np.empty((base_shape[0], base_shape[1], base_shape[2]), dtype=np.float32)

    # Generate grid of coordinates
    z, y, x = np.indices(base_shape)

    # Calculate the distance from the center of the shape
    center = np.array(base_shape) / 2

    dist = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)

    dist /= np.sqrt(3 * (center[0] ** 2))

    # Generate Perlin noise and adjust the values based on the distance from the center
    vectorized_pnoise3 = np.vectorize(
        pnoise3
    )  # Vectorize pnoise3, since it only takes scalar input

    noise = vectorized_pnoise3(
        z.flatten() * noise_scale, y.flatten() * noise_scale, x.flatten() * noise_scale
    ).reshape(base_shape)

    volume = (1 + noise) * (1 - dist)

    # Normalize
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    # Gamma correction
    volume = np.power(volume, gamma)

    # Scale the volume to the maximum value
    volume = volume * max_value

    # If volume shape is specified, smooth borders are disabled
    if volume_shape:
        smooth_borders = False

    if smooth_borders:
        # Maximum value among the six sides of the 3D volume
        max_border_value = np.max(
            [
                np.max(volume[0, :, :]),
                np.max(volume[-1, :, :]),
                np.max(volume[:, 0, :]),
                np.max(volume[:, -1, :]),
                np.max(volume[:, :, 0]),
                np.max(volume[:, :, -1]),
            ]
        )

        # Compute threshold such that there will be no straight cuts in the blob
        threshold = max_border_value / max_value

    # Clip the low values of the volume to create a coherent volume
    volume[volume < threshold * max_value] = 0

    # Clip high values
    volume[volume > max_value] = max_value

    # Scale up the volume of volume to size
    volume = scipy.ndimage.zoom(
        volume, np.array(final_shape) / np.array(base_shape), order=order
    )

    # Fade into a shape if specified
    if volume_shape == 'cylinder':
        # Arguments for the fade_mask function
        geometry = 'cylindrical'  # Fade in cylindrical geometry
        axis = np.argmax(
            volume.shape
        )  # Fade along the dimension where the volume is the largest
        target_max_normalized_distance = (
            1.4  # This value ensures that the volume will become cylindrical
        )

        volume = qim3d.operations.fade_mask(
            volume,
            geometry=geometry,
            axis=axis,
            target_max_normalized_distance=target_max_normalized_distance,
        )

    elif volume_shape == 'tube':
        # Arguments for the fade_mask function
        geometry = 'cylindrical'  # Fade in cylindrical geometry
        axis = np.argmax(
            volume.shape
        )  # Fade along the dimension where the volume is the largest
        decay_rate = 5  # Decay rate for the fade operation
        target_max_normalized_distance = (
            1.4  # This value ensures that the volume will become cylindrical
        )

        # Fade once for making the volume cylindrical
        volume = qim3d.operations.fade_mask(
            volume,
            geometry=geometry,
            axis=axis,
            decay_rate=decay_rate,
            target_max_normalized_distance=target_max_normalized_distance,
            invert=False,
        )

        # Fade again with invert = True for making the volume a tube (i.e. with a hole in the middle)
        volume = qim3d.operations.fade_mask(
            volume, geometry=geometry, axis=axis, decay_rate=decay_rate, invert=True
        )

    # Convert to desired data type
    volume = volume.astype(dtype)

    return volume


def noise_volume(
    noise_volume_shape: tuple,
    baseline_value: float = 0,
    min_noise_value: float = 0,
    max_noise_value: float = 20,
    seed: int = 0,
    dtype: str = 'uint8',
    apply_to: np.ndarray = None,
) -> np.ndarray:
    """
    Generate a noise volume with random intensity values from a uniform distribution.

    Args:
        noise_volume_shape (tuple): The shape of the noise volume to generate.
        baseline_value (float, optional): The baseline intensity of the noise volume. Default is 0.
        min_noise_value (float, optional): The minimum intensity of the noise. Default is 0.
        max_noise_value (float, optional): The maximum intensity of the noise. Default is 100.
        seed (int, optional): The seed for the random number generator. Default is 0.
        dtype (data-type, optional): Desired data type of the output volume. Defaults to 'uint8'.
        apply_to (np.ndarray, optional): The input volume to add noise to. If None, the noise volume is returned. Otherwise, the input volume with the added noise is returned. Default is None.

    Returns:
        noise_volume (np.ndarray): The generated noise volume (if `apply_to` is None) or the input volume with added noise (if `apply_to` is not None).

    Raises:
        ValueError: If the shape of `apply_to` input volume does not match `noise_volume_shape`.

    Example:
        ```python
        import qim3d

        # Generate noise volume
        noise_volume = qim3d.generate.noise_volume(
            noise_volume_shape = (128, 128, 128),
            baseline_value = 20,
            min_noise_value = 100,
            max_noise_value = 200,
        )

        qim3d.viz.volumetric(noise_volume)
        ```
        <iframe src="https://platform.qim.dk/k3d/noise_volume.html" width="100%" height="500" frameborder="0"></iframe>

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of volumes
        volume_collection, labels = qim3d.generate.volume_collection(num_volumes = 15)

        # Apply noise to the synthetic collection
        noisy_collection = qim3d.generate.noise_volume(
            noise_volume_shape = volume_collection.shape,
            min_noise_value = 0,
            max_noise_value = 20,
            apply_to = volume_collection
        )

        qim3d.viz.volumetric(noisy_collection)
        ```
        <iframe src="https://platform.qim.dk/k3d/noisy_collection.html" width="100%" height="500" frameborder="0"></iframe>

    """
    # Check for shape mismatch
    if (apply_to is not None) and (apply_to.shape != noise_volume_shape):
        msg = f'Shape of input volume {apply_to.shape} does not match noise_volume_shape {noise_volume_shape}.'
        raise ValueError(msg)

    # Generate the noise volume
    baseline = np.full(shape=noise_volume_shape, fill_value=baseline_value)

    noise = np.random.default_rng(seed=seed).uniform(
        low=min_noise_value, high=max_noise_value, size=noise_volume_shape
    )

    noise_volume = baseline + noise

    # Convert to desired data type
    noise_volume = noise_volume.astype(dtype)

    # If specified, add the noise volume to the input volume
    if apply_to is not None:
        noise_volume = apply_to + noise_volume
        return noise_volume

    # Otherwise, return the noise volume
    else:
        return noise_volume
