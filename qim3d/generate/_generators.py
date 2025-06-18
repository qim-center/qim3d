import ipywidgets as widgets
import k3d
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from IPython.display import display
from noise import pnoise3, snoise3

import qim3d
from qim3d.utils import log
from qim3d.utils._misc import scale_to_float16

__all__ = ['volume', 'background']


def _volume(
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
    generate_method: str = 'add',
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
        generate_method (str, optional): The method used to combine `baseline_value` and noise. Choose from 'add' (`baseline + noise`), 'subtract' (`baseline - noise`), 'multiply' (`baseline * noise`), or 'divide' (`baseline / (noise+ε)`). Default is 'add'.
        apply_method (str, optional): The method to apply the generated noise to `apply_to`, if provided. Choose from 'add' (`apply_to + background`), 'subtract' (`apply_to - background`), 'multiply' (`apply_to * background`), or 'divide' (`apply_to / (background+ε)`). Only applicable if apply_to is defined. Default is None.
        seed (int, optional): The seed for the random number generator. Default is 0.
        dtype (data-type, optional): Desired data type of the output volume. Default is 'uint8'.
        apply_to (np.ndarray, optional): An input volume to which noise will be applied. Only applicable if apply_method is defined. Defaults to None.

    Returns:
        background (np.ndarray): The generated noise volume (if `apply_to` is None) or the input volume with added noise (if `apply_to` is not None).

    Raises:
        ValueError: If `apply_method` is not one of 'add', 'subtract', 'multiply', or 'divide'.
        ValueError: If `apply_method` is provided without `apply_to` input volume provided, or vice versa.

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
            generate_method = 'add',
            apply_method = 'add',
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

    # Check if apply_method is provided without apply_to volume, or vice versa
    if (apply_to is None and apply_method is not None) or (
        apply_to is not None and apply_method is None
    ):
        msg = 'Supply both apply_method and apply_to when applying background to a volume.'
        # Validate apply_method
        raise ValueError(msg)

    # Validate apply_method
    if apply_method is not None and apply_method not in apply_operations:
        msg = f"Invalid apply_method '{apply_method}'. Choose from {list(apply_operations.keys())}."
        raise ValueError(msg)

    # Validate generate_method
    if generate_method not in apply_operations:
        msg = f"Invalid generate_method '{generate_method}'. Choose from {list(apply_operations.keys())}."
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

    # Return error if multiplying or dividing with 0
    if baseline_value == 0.0 and (
        generate_method == 'multiply' or generate_method == 'divide'
    ):
        msg = f'Selection of baseline_value=0 and generate_method="{generate_method}" will not generate background noise. Either add baseline_value>0 or change generate_method.'
        raise ValueError(msg)

    # Apply method to initial background computation
    background_volume = apply_operations[generate_method](baseline, noise)

    # Warn user if the background noise is constant or none
    if np.min(background_volume) == np.max(background_volume):
        msg = 'Warning: The used settings have generated a background with a uniform value.'
        log.info(msg)

    # Apply method to the target volume if specified
    if apply_to is not None:
        background_volume = apply_operations[apply_method](apply_to, background_volume)

    # Clip value before dtype convertion
    clip_value = (
        np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else np.finfo(dtype).max
    )
    background_volume = np.clip(background_volume, 0, clip_value).astype(dtype)

    return background_volume


def volume(
    base_shape: tuple = (128, 128, 128),
    final_shape: tuple = None,
    noise_scale: float = 0.02,
    noise_type: str = 'perlin',
    decay_rate: float = 10,
    gamma: float = 1,
    threshold: float = 0.5,
    max_value: float = 255,
    shape: str = None,
    tube_hole_ratio: float = 0.5,
    axis: int = 0,
    order: int = 1,
    dtype: str = 'uint8',
    hollow: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a 3D volume with Perlin noise, spherical gradient, and optional scaling and gamma correction.

    Args:
        base_shape (tuple of ints, optional): Shape of the initial volume to generate. Defaults to (128, 128, 128).
        final_shape (tuple of ints, optional): Desired shape of the final volume. If unspecified, will assume same shape as base_shape. Defaults to None.
        noise_scale (float, optional): Scale factor for Perlin noise. Defaults to 0.05.
        noise_type (str, optional): Type of noise to be used for volume generation. Should be `simplex` or `perlin`. Defaults to perlin.
        decay_rate (float, optional): The decay rate of the fading of the noise. Can also be interpreted as the sharpness of the edge of the volume. Defaults to 5.0.
        gamma (float, optional): Applies gamma correction, adjusting contrast in the volume. If gamma<0, the volume intensity is increased and if gamma>0 it's decreased. Defaults to 0.
        threshold (float, optional): Threshold value for clipping low intensity values. Defaults to 0.5.
        max_value (int, optional): Maximum value for the volume intensity. Defaults to 255.
        shape (str, optional): Shape of the volume to generate, either `cylinder`, or `tube`. Defaults to None.
        tube_hole_ratio (float, optional): Ratio for the inverted fade mask used to generate tubes. Will only have an effect if shape=`tube`. Defaults to 0.5.
        axis (int, optional): Axis of the given shape. Will only be active if shape is defined. Defaults to 0.
        order (int, optional): Order of the spline interpolation used in resizing. Defaults to 1.
        dtype (data-type, optional): Desired data type of the output volume. Defaults to `uint8`.
        hollow (bool, optional): Determines thickness of the hollowing operation. Volume is only hollowed if hollow>0. Defaults to 0.
        seed (int, optional): Specifies a fixed offset for the generated noise. Only works for perlin noise. Defaults to 0.

    Returns:
        volume (numpy.ndarray): Generated 3D volume with specified parameters.

    Raises:
        ValueError: If `shape` is invalid.
        ValueError: If `noise_type` is invalid.
        TypeError: If `base_shape` is not a tuple or does not have three elements.
        TypeError: If `final_shape` is not a tuple or does not have three elements.
        TypeError: If `dtype` is not a valid numpy number type.
        ValueError: If `hollow` is not 0 or a positive integer.

    Example:
        Example:
        ```python
        import qim3d

        # Generate synthetic blob
        vol = qim3d.generate.volume(noise_scale = 0.02)

        # Visualize 3D volume
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_blob_1.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize slices
        qim3d.viz.slices_grid(vol, value_min = 0, value_max = 255, num_slices = 15)
        ```
        ![synthetic_blob](../../assets/screenshots/synthetic_blob_slices.png)

    Example:
        ```python
        import qim3d

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(base_shape = (200, 100, 100),
                                    final_shape = (400,100,100),
                                    noise_scale = 0.03,
                                    threshold = 0.85,
                                    decay_rate=20,
                                    gamma=0.15,
                                    shape = "tube",
                                    tube_hole_ratio = 0.4,
                                    )

        # Visualize synthetic volume
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_blob_cylinder_1.html" width="100%" height="500" frameborder="0"></iframe>

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
                                shape = "tube",
                                )

        # Visualize synthetic blob
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_blob_tube_1.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize
        qim3d.viz.slices_grid(vol, num_slices=15)
        ```
        ![synthetic_blob_tube_slice](../../assets/screenshots/synthetic_blob_tube_slice.png)

    """
    # Control
    shape_types = ['cylinder', 'tube']
    if shape and shape not in shape_types:
        err = f'shape should be one of: {shape_types}'
        raise ValueError(err)
    noise_types = ['pnoise', 'perlin', 'p', 'snoise', 'simplex', 's']
    if noise_type not in noise_types:
        err = f'noise_type should be one of: {noise_types}'
        raise ValueError(err)

    if not isinstance(base_shape, tuple) or len(base_shape) != 3:
        message = 'base_shape must be a tuple with three dimensions (z, y, x)'
        raise TypeError(message)

    if final_shape and (not isinstance(final_shape, tuple) or len(final_shape) != 3):
        message = 'final_shape must be a tuple with three dimensions (z, y, x)'
        raise TypeError(message)

    try:
        d = np.dtype(dtype)
    except TypeError as e:
        err = f'Datatype {dtype} is not a valid dtype.'
        raise TypeError(err) from e

    if hollow < 0 or isinstance(hollow, float):
        err = 'Argument "hollow" should be 0 or a positive integer'
        raise ValueError(err)

    # Generate grid of coordinates
    z, y, x = np.indices(base_shape)

    # Generate noise
    if (
        np.round(noise_scale, 3) == 0
    ):  # Only detect three decimal position (0.001 is ok, but 0.0001 is 0)
        noise_scale = 0

    if noise_scale == 0:
        noise = np.ones(base_shape)
    else:
        if noise_type in noise_types[:3]:
            vectorized_noise = np.vectorize(pnoise3)
            noise = vectorized_noise(
                z.flatten() * noise_scale,
                y.flatten() * noise_scale,
                x.flatten() * noise_scale,
                base=seed,
            ).reshape(base_shape)
        elif noise_type in noise_types[3:]:
            vectorized_noise = np.vectorize(snoise3)
            noise = vectorized_noise(
                z.flatten() * noise_scale,
                y.flatten() * noise_scale,
                x.flatten() * noise_scale,
            ).reshape(base_shape)
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

    # Calculate the center of the array
    center = np.array([(s - 1) / 2 for s in base_shape])

    # Calculate the distance of each point from the center
    if not shape:
        distance = np.linalg.norm(
            [
                (z - center[0]) / center[0],
                (y - center[1]) / center[1],
                (x - center[2]) / center[2],
            ],
            axis=0,
        )
        max_distance = np.sqrt(3)
        # Set ratio
        miin = np.max(
            [
                distance[distance.shape[0] // 2, distance.shape[1] // 2, 0],
                distance[distance.shape[0] // 2, 0, distance.shape[2] // 2],
                distance[0, distance.shape[1] // 2, distance.shape[2] // 2],
            ]
        )
        ratio = miin / max_distance  # 0.577

    elif shape == 'cylinder' or shape == 'tube':
        distance_list = np.array(
            [
                (z - center[0]) / center[0],
                (y - center[1]) / center[1],
                (x - center[2]) / center[2],
            ]
        )
        # remove the axis along which the fading is not applied
        distance_list = np.delete(distance_list, axis, axis=0)
        distance = np.linalg.norm(distance_list, axis=0)
        max_distance = np.sqrt(2)
        # Set ratio
        miin = np.max(
            [
                distance[distance.shape[0] // 2, distance.shape[1] // 2, 0],
                distance[distance.shape[0] // 2, 0, distance.shape[2] // 2],
                distance[0, distance.shape[1] // 2, distance.shape[2] // 2],
            ]
        )
        ratio = miin / max_distance  # 0.707

    # Scale the distance such that the shortest distance (from center to any edge) is 1 (prevents clipping)
    scaled_distance = distance / (max_distance * ratio)

    # Apply decay rate
    faded_distance = np.power(scaled_distance, decay_rate)

    # Invert the distances to have 1 at the center and 0 at the edges
    fade_array = 1 - faded_distance
    fade_array[fade_array <= 0] = 0

    # Apply the fading to the volume
    vol_faded = noise * fade_array

    # Normalize the volume
    vol_normalized = vol_faded / np.max(vol_faded)

    # Apply gamma
    generated_vol = np.power(vol_normalized, gamma)

    # Scale to max_value
    generated_vol = generated_vol * max_value

    # Threshold
    generated_vol[generated_vol < threshold * max_value] = 0

    # Apply fade mask for creation of tube
    if shape == 'tube':
        generated_vol = qim3d.operations.fade_mask(
            generated_vol,
            geometry='cylindrical',
            axis=axis,
            ratio=tube_hole_ratio,
            decay_rate=5,
            invert=True,
        )

    # Scale up the volume of volume to size
    if final_shape:
        generated_vol = scipy.ndimage.zoom(
            generated_vol, np.array(final_shape) / np.array(base_shape), order=order
        )

    generated_vol = generated_vol.astype(dtype)

    if hollow > 0:
        generated_vol = qim3d.operations.make_hollow(generated_vol, hollow)

    return generated_vol


class ParameterVisualizer:
    """
    Class for visualizing and experimenting with parameter changes and combinations on synthetic data.

    Args:
        base_shape (tuple, optional): Determines the shape of the generate volume. This will not be update when exploring parameters and must be determined when generating the visualizer.
        seed (int, optional): Determines the seed for the volume generation. Enables the user to generate different volumes with the same parameters.
        initial_config (dict, optional): Dictionary that defines the starting parameters of the visualizer. Can be used if a specific setup is needed. The dictionary may contain the keywords: `noise_type`, `noise_scale`, `decay_rate`, `gamma`, `threshold`, `shape` and `tube_hole_ratio`.
        nsmin (float, optional): Determines minimum value for the noise scale slider. Defaults to 0.0.
        nsmax (float, optional): Determines maximum value for the noise scale slider. Defaults to 0.1.
        dsmin (float, optional): Determines minimum value for the decay rate slider. Defaults to 0.1.
        dsmax (float, optional): Determines maximum value for the decay rate slider. Defaults to 20.
        gsmin (float, optional): Determines minimum value for the gamma slider. Defaults to 0.1.
        gsmax (float, optional): Determines maximum value for the gamma slider. Defaults to 2.0.
        tsmin (float, optional): Determines minimum value for the threshold slider. Defaults to 0.0.
        tsmax (float, optional): Determines maximum value for the threshold slider. Defaults to 1.0.
        grid_visible (bool, optional): Determines if the grid should be visible upon plot generation. Defaults to False.

    Raises:
        ValueError: If base_shape is invalid.
        ValueError: If noise slider values are invalid.
        ValueError: If decay slider values are invalid.
        ValueError: If gamma slider values are invalid.
        ValueError: If threshold slider values are invalid.

    Example:
        ```python
        import qim3d

        viz = qim3d.generate.ParameterVisualizer(base_shape=(128,128,128), seed=0, grid_visible=True)
        ```
        ![paramter_visualizer](../../assets/screenshots/viz-synthetic_parameters.gif)

    """

    def __init__(
        self,
        base_shape: tuple = (128, 128, 128),
        seed: int = 0,
        initial_config: dict = None,
        nsmin: float = 0.0,
        nsmax: float = 0.1,
        dsmin: float = 0.1,
        dsmax: float = 20.0,
        gsmin: float = 0.1,
        gsmax: float = 2.0,
        tsmin: float = 0.0,
        tsmax: float = 1.0,
        grid_visible: bool = False,
    ):
        # Error checking:

        if not isinstance(base_shape, tuple) or len(base_shape) != 3:
            err = 'base_shape should be a tuple of three sizes.'
            raise ValueError(err)

        if nsmin > nsmax:
            err = f'Minimum slider value for noise must be less than or equal to the maximum. Given: min = {nsmin}, max = {nsmax}.'
            raise ValueError(err)

        if dsmin > dsmax:
            err = f'Minimum decay rate value must be less than or equal to the maximum. Given: min = {dsmin}, max = {dsmax}.'
            raise ValueError(err)

        if gsmin > gsmax:
            err = f'Minimum gamma value must be less than or equal to the maximum. Given: min = {gsmin}, max = {gsmax}.'
            raise ValueError(err)

        if tsmin > tsmax:
            err = f'Minimum threshold value must be less than or equal to the maximum. Given: min = {tsmin}, max = {tsmax}.'
            raise ValueError(err)

        self.base_shape = base_shape
        self.seed = int(seed)
        self.axis = 0  # Not customizable
        self.max_value = 255  # Not customizable

        # Min and max values for sliders
        self.nsmin = nsmin
        self.nsmax = nsmax
        self.dsmin = dsmin
        self.dsmax = dsmax
        self.gsmin = gsmin
        self.gsmax = gsmax
        self.tsmin = tsmin
        self.tsmax = tsmax

        self.grid_visible = grid_visible
        self.config = {
            'noise_scale': 0.02,
            'decay_rate': 10,
            'gamma': 1.0,
            'threshold': 0.5,
            'tube_hole_ratio': 0.5,
            'shape': None,
            'noise_type': 'perlin',
        }
        if initial_config:
            self.config.update(initial_config)

        self.state = {}
        self._build_widgets()
        self._setup_plot()
        self._display_ui()

    def _compute_volume(self) -> None:
        vol = volume(
            base_shape=self.base_shape,
            noise_type=self.config['noise_type'],
            noise_scale=self.config['noise_scale'],
            decay_rate=self.config['decay_rate'],
            gamma=self.config['gamma'],
            threshold=self.config['threshold'],
            shape=self.config['shape'],
            tube_hole_ratio=self.config['tube_hole_ratio'],
            seed=self.seed,
        )
        return scale_to_float16(vol)

    def _build_widgets(self) -> None:
        self.noise_slider = widgets.FloatSlider(
            value=self.config['noise_scale'],
            min=self.nsmin,
            max=self.nsmax,
            step=0.001,
            description='Noise',
            readout_format='.3f',
            continuous_update=False,
        )
        self.decay_slider = widgets.FloatSlider(
            value=self.config['decay_rate'],
            min=self.dsmin,
            max=self.dsmax,
            step=0.1,
            description='Decay',
            continuous_update=False,
        )
        self.gamma_slider = widgets.FloatSlider(
            value=self.config['gamma'],
            min=self.gsmin,
            max=self.gsmax,
            step=0.1,
            description='Gamma',
            continuous_update=False,
        )
        self.threshold_slider = widgets.FloatSlider(
            value=self.config['threshold'],
            min=self.tsmin,
            max=self.tsmax,
            step=0.05,
            description='Threshold',
            continuous_update=False,
        )
        self.noise_type_dropdown = widgets.Dropdown(
            options=['perlin', 'simplex'], value='perlin', description='Noise Type'
        )
        self.shape_dropdown = widgets.Dropdown(
            options=[None, 'cylinder', 'tube'], value=None, description='Shape'
        )
        self.tube_hole_ratio_slider = widgets.FloatSlider(
            value=self.config['tube_hole_ratio'],
            min=0.0,
            max=1.0,
            step=0.05,
            description='Tube hole ratio',
            style={'description_width': 'initial'},
            continuous_update=False,
        )

        # Observers
        self.noise_slider.observe(self._on_change, names='value')
        self.noise_type_dropdown.observe(self._on_change, names='value')
        self.decay_slider.observe(self._on_change, names='value')
        self.gamma_slider.observe(self._on_change, names='value')
        self.threshold_slider.observe(self._on_change, names='value')
        self.shape_dropdown.observe(self._on_change, names='value')
        self.tube_hole_ratio_slider.observe(self._on_change, names='value')

    def _setup_plot(self) -> None:
        vol = self._compute_volume()

        cmap = plt.get_cmap('magma')
        attr_vals = np.linspace(0.0, 1.0, num=cmap.N)
        rgb_vals = cmap(np.arange(0, cmap.N))[:, :3]
        color_map = np.column_stack((attr_vals, rgb_vals)).tolist()

        pixel_count = np.prod(vol.shape)
        y1, x1 = 256, 16777216  # 256 samples at res 256*256*256=16.777.216
        y2, x2 = 32, 134217728  # 32 samples at res 512*512*512=134.217.728
        a = (y1 - y2) / (x1 - x2)
        b = y1 - a * x1
        samples = int(min(max(a * pixel_count + b, 64), 512))

        self.plot = k3d.plot(grid_visible=self.grid_visible)
        self.plt_volume = k3d.volume(
            vol,
            bounds=[0, vol.shape[2], 0, vol.shape[1], 0, vol.shape[0]],
            color_map=color_map,
            samples=samples,
            color_range=[np.min(vol), np.max(vol)],
            opacity_function=[],
            interpolation=True,
        )
        self.plot += self.plt_volume

    def _on_change(self, change: None = None) -> None:
        self.config['noise_type'] = self.noise_type_dropdown.value
        self.config['noise_scale'] = self.noise_slider.value
        self.config['decay_rate'] = self.decay_slider.value
        self.config['gamma'] = self.gamma_slider.value
        self.config['threshold'] = self.threshold_slider.value
        self.config['shape'] = self.shape_dropdown.value
        self.config['tube_hole_ratio'] = self.tube_hole_ratio_slider.value

        new_vol = self._compute_volume()
        self.plt_volume.volume = new_vol

    def _display_ui(self) -> None:
        controls = widgets.VBox(
            [
                self.noise_type_dropdown,
                self.noise_slider,
                self.decay_slider,
                self.gamma_slider,
                self.threshold_slider,
                self.shape_dropdown,
                self.tube_hole_ratio_slider,
            ]
        )

        # Controls styling

        controls.layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            flex='0 1',
            min_width='350px',  # Ensure it doesn't get too small
            height='auto',
            overflow_y='auto',
            border='1px solid lightgray',
            padding='10px',
            margin='0 1em 0 0',
        )

        plot_output = widgets.Output()

        plot_output.layout = widgets.Layout(
            flex='1 1 auto',  # Expand to fill remaining space
            height='auto',
            border='1px solid lightgray',
            overflow='auto',
            margin='0 1em 0 0',
        )

        ui = widgets.HBox(
            [controls, plot_output],
            layout=widgets.Layout(
                width='100%', display='flex', flex_flow='row', align_items='stretch'
            ),
        )

        with plot_output:
            display(self.plot)

        display(ui)
