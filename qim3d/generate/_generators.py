import numpy as np
import scipy.ndimage
from noise import pnoise3

import qim3d.processing
from qim3d.utils import log

__all__ = ['volume', 'background']


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


def background(
    background_shape: tuple,
    baseline_value: float = 0,
    min_noise_value: float = 0,
    max_noise_value: float = 20,
    generate_method: str = 'divide',
    apply_method: str = None,
    seed: int = 0,
    dtype: str = 'uint8',
    apply_to: np.ndarray = None,
) -> np.ndarray:
    """
    Generate a noise volume with random intensity values from a uniform distribution.

    Args:
        background_shape (tuple): The shape of the noise volume to generate.
        baseline_value (float, optional): The baseline intensity of the noise volume. Default is 0.
        min_noise_value (float, optional): The minimum intensity of the noise. Default is 0.
        max_noise_value (float, optional): The maximum intensity of the noise. Default is 20.
        generate_method (str, optional): The method used to combine `baseline_value` and noise. Choose from 'add' (`baseline + noise`), 'subtract' (`baseline - noise`), 'multiply' (`baseline * noise`), or 'divide' (`baseline / (1 + noise)`). Default is 'divide'.
        apply_method (str, optional): The method to apply the generated noise to `apply_to`, if provided. Choose from 'add' (`apply_to + background`), 'subtract' (`apply_to - background`), 'multiply' (`apply_to * background`), or 'divide' (`apply_to / (1 + background)`). Default is None.
        seed (int, optional): The seed for the random number generator. Default is 0.
        dtype (data-type, optional): Desired data type of the output volume. Default is 'uint8'.
        apply_to (np.ndarray, optional): An input volume to which noise will be applied. If None, the generated noise volume is returned.

    Returns:
        background (np.ndarray): The generated noise volume (if `apply_to` is None) or the input volume with added noise (if `apply_to` is not None).

    Raises:
        ValueError: If `apply_method` is not one of 'add', 'subtract', 'multiply', or 'divide'.
        ValueError: If `apply_method` is provided without `apply_to` input volume provided.
        ValueError: If the shape of `apply_to` input volume does not match `background_shape`.

    Example:
        ```python
        import qim3d

        # Generate noise volume
        background = qim3d.generate.background(
            background_shape = (128, 128, 128),
            baseline_value = 20,
            min_noise_value = 100,
            max_noise_value = 200,
        )

        qim3d.viz.volumetric(background)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_noise_background.html" width="100%" height="500" frameborder="0"></iframe>

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of volumes
        volume_collection, labels = qim3d.generate.volume_collection(num_volumes = 15)

        # Apply noise to the synthetic collection
        noisy_collection = qim3d.generate.background(
            background_shape = volume_collection.shape,
            min_noise_value = 0,
            max_noise_value = 20,
            apply_to = volume_collection
        )

        qim3d.viz.volumetric(noisy_collection)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_noisy_collection_1.html" width="100%" height="500" frameborder="0"></iframe>

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of volumes
        volume_collection, labels = qim3d.generate.volume_collection(num_volumes = 15)

        # Apply noise to the synthetic collection
        noisy_collection = qim3d.generate.background(
            background_shape = volume_collection.shape,
            baseline_value = 0,
            min_noise_value = 0,
            max_noise_value = 30,
            generate_method = 'add',
            apply_method = 'divide',
            apply_to = volume_collection
        )

        qim3d.viz.volumetric(noisy_collection)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_noisy_collection_2.html" width="100%" height="500" frameborder="0"></iframe>
        ```python
        qim3d.viz.slices_grid(noisy_collection, num_slices=10, color_bar=True, color_bar_style="large")
        ```
        ![synthetic_noisy_collection_slices](../../assets/screenshots/synthetic_noisy_collection_slices_2.png)

    Example:
        ```python
        import qim3d

        # Generate synthetic collection of volumes
        volume_collection, labels = qim3d.generate.volume_collection(num_volumes = 15)

        # Apply noise to the synthetic collection
        noisy_collection = qim3d.generate.background(
            background_shape = (200, 200, 200),
            baseline_value = 100,
            min_noise_value = 0.8,
            max_noise_value = 1.2,
            generate_method = "multiply",
            apply_method = "add",
            apply_to = volume_collection
        )

        qim3d.viz.slices_grid(noisy_collection, num_slices=10, color_bar=True, color_bar_style="large")
        ```
        ![synthetic_noisy_collection_slices](../../assets/screenshots/synthetic_noisy_collection_slices_3.png)

    """
    # Ensure dtype is a valid NumPy type
    dtype = np.dtype(dtype)

    # Define supported apply methods
    apply_operations = {
        'add': lambda a, b: a + b,
        'subtract': lambda a, b: a - b,
        'multiply': lambda a, b: a * b,
        'divide': lambda a, b: a / (b + 1e-8),  # Avoid division by zero
    }

    # Check if apply_method is provided without apply_to volume
    if (apply_to is None) and (apply_method is not None):
        msg = f"apply_method '{apply_method}' is only supported when apply_to input volume is provided."
        # Validate apply_method
        if apply_method not in apply_operations:
            msg = f"Invalid apply_method '{apply_method}'. Choose from {list(apply_operations.keys())}."
            raise ValueError(msg)

        raise ValueError(msg)

    # Check for shape mismatch
    if (apply_to is not None) and (apply_to.shape != background_shape):
        msg = f'Shape of input volume {apply_to.shape} does not match requested background_shape {background_shape}. Using input shape instead.'
        background_shape = apply_to.shape
        log.info(msg)

    # Generate the noise volume
    baseline = np.full(shape=background_shape, fill_value=baseline_value)

    # Start seeded generator
    rng = np.random.default_rng(seed=seed)
    noise = rng.uniform(
        low=float(min_noise_value), high=float(max_noise_value), size=background_shape
    )

    # Apply method to initial background computation
    background_volume = apply_operations[generate_method](baseline, noise)

    # Apply method to the target volume if specified
    if apply_to is not None:
        background_volume = apply_operations[apply_method](apply_to, background_volume)

    # Clip value before dtype convertion
    clip_value = (
        np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else np.finfo(dtype).max
    )
    background_volume = np.clip(background_volume, 0, clip_value).astype(dtype)

    return background_volume
