import numpy as np
import scipy.ndimage
from noise import pnoise3

import qim3d.processing

def blob(
    base_shape: tuple = (128, 128, 128),
    final_shape: tuple = (128, 128, 128),
    noise_scale: float = 0.05,
    order: int = 1,
    gamma: int = 1.0,
    max_value: int = 255,
    threshold: float = 0.5,
    smooth_borders: bool = False,
    object_shape: str = None,
    dtype: str = "uint8",
    ) -> np.ndarray:
    """
    Generate a 3D volume with Perlin noise, spherical gradient, and optional scaling and gamma correction.

    Args:
        base_shape (tuple, optional): Shape of the initial volume to generate. Defaults to (128, 128, 128).
        final_shape (tuple, optional): Desired shape of the final volume. Defaults to (128, 128, 128).
        noise_scale (float, optional): Scale factor for Perlin noise. Defaults to 0.05.
        order (int, optional): Order of the spline interpolation used in resizing. Defaults to 1.
        gamma (float, optional): Gamma correction factor. Defaults to 1.0.
        max_value (int, optional): Maximum value for the volume intensity. Defaults to 255.
        threshold (float, optional): Threshold value for clipping low intensity values. Defaults to 0.5.
        smooth_borders (bool, optional): Flag for automatic computation of the threshold value to ensure a blob with no straight edges. If True, the `threshold` parameter is ignored. Defaults to False.
        object_shape (str, optional): Shape of the object to generate, either "cylinder", or "tube". Defaults to None.
        dtype (str, optional): Desired data type of the output volume. Defaults to "uint8".

    Returns:
        synthetic_blob (numpy.ndarray): Generated 3D volume with specified parameters.

    Raises:
        TypeError: If `final_shape` is not a tuple or does not have three elements.
        ValueError: If `dtype` is not a valid numpy number type.

    Example:
        ```python
        import qim3d

        # Generate synthetic blob
        synthetic_blob = qim3d.generate.blob(noise_scale = 0.015)

        # Visualize 3D volume
        qim3d.viz.vol(synthetic_blob)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_blob.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize slices
        qim3d.viz.slices(synthetic_blob, vmin = 0, vmax = 255, n_slices = 15)
        ```
        ![synthetic_blob](assets/screenshots/synthetic_blob_slices.png)

    Example:
        ```python
        import qim3d

        # Generate tubular synthetic blob
        vol = qim3d.generate.blob(base_shape = (10, 300, 300),
                                final_shape = (100, 100, 100),
                                noise_scale = 0.3,
                                gamma = 2,
                                threshold = 0.0,
                                object_shape = "cylinder"
                                )

        # Visualize synthetic blob
        qim3d.viz.vol(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_blob_cylinder.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Visualize slices
        qim3d.viz.slices(vol, n_slices=15, axis=1)
        ```
        ![synthetic_blob_cylinder_slice](assets/screenshots/synthetic_blob_cylinder_slice.png)

    Example:
        ```python
        import qim3d

        # Generate tubular synthetic blob
        vol = qim3d.generate.blob(base_shape = (200, 100, 100),
                                final_shape = (400, 100, 100),
                                noise_scale = 0.03,
                                gamma = 0.12,
                                threshold = 0.85,
                                object_shape = "tube"
                                )

        # Visualize synthetic blob
        qim3d.viz.vol(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_blob_tube.html" width="100%" height="500" frameborder="0"></iframe>
        
        ```python
        # Visualize
        qim3d.viz.slices(vol, n_slices=15)
        ```
        ![synthetic_blob_tube_slice](assets/screenshots/synthetic_blob_tube_slice.png)    
    """

    if not isinstance(final_shape, tuple) or len(final_shape) != 3:
        raise TypeError("Size must be a tuple of 3 dimensions")
    if not np.issubdtype(dtype, np.number):
        raise ValueError("Invalid data type")

    # Initialize the 3D array for the shape
    volume = np.empty((base_shape[0], base_shape[1], base_shape[2]), dtype=np.float32)

    # Generate grid of coordinates
    z, y, x = np.indices(base_shape)

    # Calculate the distance from the center of the shape
    center = np.array(base_shape) / 2

    dist = np.sqrt((z - center[0])**2 + 
                   (y - center[1])**2 + 
                   (x - center[2])**2)
    
    dist /= np.sqrt(3 * (center[0]**2))

    # Generate Perlin noise and adjust the values based on the distance from the center
    vectorized_pnoise3 = np.vectorize(pnoise3) # Vectorize pnoise3, since it only takes scalar input

    noise = vectorized_pnoise3(z.flatten() * noise_scale, 
                               y.flatten() * noise_scale, 
                               x.flatten() * noise_scale
                               ).reshape(base_shape)

    volume = (1 + noise) * (1 - dist)

    # Normalize
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    # Gamma correction
    volume = np.power(volume, gamma)

    # Scale the volume to the maximum value
    volume = volume * max_value

    # If object shape is specified, smooth borders are disabled
    if object_shape:
        smooth_borders = False

    if smooth_borders: 
        # Maximum value among the six sides of the 3D volume
        max_border_value = np.max([
            np.max(volume[0, :, :]), np.max(volume[-1, :, :]),
            np.max(volume[:, 0, :]), np.max(volume[:, -1, :]),
            np.max(volume[:, :, 0]), np.max(volume[:, :, -1])
        ])

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
    if object_shape == "cylinder":

        # Arguments for the fade_mask function
        geometry = "cylindrical"        # Fade in cylindrical geometry
        axis = np.argmax(volume.shape)  # Fade along the dimension where the object is the largest
        target_max_normalized_distance = 1.4   # This value ensures that the object will become cylindrical

        volume = qim3d.processing.operations.fade_mask(volume, 
                                                       geometry = geometry, 
                                                       axis = axis, 
                                                       target_max_normalized_distance = target_max_normalized_distance
                                                       )

    elif object_shape == "tube":

        # Arguments for the fade_mask function
        geometry = "cylindrical"        # Fade in cylindrical geometry
        axis = np.argmax(volume.shape)  # Fade along the dimension where the object is the largest
        decay_rate = 5                  # Decay rate for the fade operation
        target_max_normalized_distance = 1.4   # This value ensures that the object will become cylindrical

        # Fade once for making the object cylindrical
        volume = qim3d.processing.operations.fade_mask(volume, 
                                                       geometry = geometry, 
                                                       axis = axis,
                                                       decay_rate = decay_rate,
                                                       target_max_normalized_distance = target_max_normalized_distance,
                                                       invert = False
                                                       )

        # Fade again with invert = True for making the object a tube (i.e. with a hole in the middle)
        volume = qim3d.processing.operations.fade_mask(volume, 
                                                       geometry = geometry, 
                                                       axis = axis, 
                                                       decay_rate = decay_rate,
                                                       invert = True
                                                       )
        
    # Convert to desired data type
    volume = volume.astype(dtype)

    return volume