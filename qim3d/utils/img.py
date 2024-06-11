import numpy as np
import scipy.ndimage
from noise import pnoise3


def overlay_rgb_images(background, foreground, alpha=0.5):
    """Overlay a RGB foreground onto an RGB background using alpha blending.

    Args:
        background (numpy.ndarray): The background RGB image.
        foreground (numpy.ndarray): The foreground RGB image (usually masks).
        alpha (float, optional): The alpha value for blending. Defaults to 0.5.

    Returns:
        numpy.ndarray: The composite RGB image with overlaid foreground.

    Raises:
        ValueError: If input images have different shapes.

    Note:
        - The function performs alpha blending to overlay the foreground onto the background.
        - It ensures that the background and foreground have the same shape before blending.
        - It calculates the maximum projection of the foreground and blends them onto the background.
        - Brightness outside the foreground is adjusted to maintain consistency with the background.
    """

    # Igonore alpha in case its there
    background = background[..., :3]
    foreground = foreground[..., :3]

    # Ensure both images have the same shape
    if background.shape != foreground.shape:
        raise ValueError("Input images must have the same shape")

    # Perform alpha blending
    foreground_max_projection = np.amax(foreground, axis=2)
    foreground_max_projection = np.stack((foreground_max_projection,) * 3, axis=-1)

    # Normalize if we have something
    if np.max(foreground_max_projection) > 0:
        foreground_max_projection = foreground_max_projection / np.max(
            foreground_max_projection
        )

    composite = background * (1 - alpha) + foreground * alpha
    composite = np.clip(composite, 0, 255).astype("uint8")

    # Adjust brightness outside foreground
    composite = composite + (background * (1 - alpha)) * (1 - foreground_max_projection)

    return composite.astype("uint8")


def generate_volume(
    base_shape=(128, 128, 128),
    final_shape=(128, 128, 128),
    noise_scale=0.05,
    order=1,
    gamma=1.0,
    max_value=255,
    threshold=0.5,
    dtype="uint8",
):
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
        dtype (str, optional): Desired data type of the output volume. Defaults to "uint8".

    Returns:
        numpy.ndarray: Generated 3D volume with specified parameters.

    Raises:
        ValueError: If `final_shape` is not a tuple or does not have three elements.
        ValueError: If `dtype` is not a valid numpy number type.

    Example:
        ```python
        import qim3d
        vol = qim3d.utils.generate_volume(noise_scale=0.05, threshold=0.4)
        qim3d.viz.slices(vol, vmin=0, vmax=255, n_slices=15)
        ```
        ![generate_volume](assets/screenshots/generate_volume.png)

        ```python
        qim3d.viz.vol(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/synthetic_volume.html" width="100%" height="500" frameborder="0"></iframe>

    """

    if not isinstance(final_shape, tuple) or len(final_shape) != 3:
        raise ValueError("Size must be a tuple")
    if not np.issubdtype(dtype, np.number):
        raise ValueError("Invalid data type")

    # Define the dimensions of the shape for generating Perlin noise

    # Initialize the 3D array for the shape
    volume = np.empty((base_shape[0], base_shape[1], base_shape[2]), dtype=np.float32)

    # Fill the 3D array with values from the Perlin noise function
    for i in range(base_shape[0]):
        for j in range(base_shape[1]):
            for k in range(base_shape[2]):
                # Calculate the distance from the center of the shape
                dist = np.sqrt(
                    (i - base_shape[0] / 2) ** 2
                    + (j - base_shape[1] / 2) ** 2
                    + (k - base_shape[2] / 2) ** 2
                ) / np.sqrt(3 * ((base_shape[0] / 2) ** 2))
                # Generate Perlin noise and adjust the values based on the distance from the center
                # This creates a spherical shape with noise
                volume[i][j][k] = (
                    1 + pnoise3(i * noise_scale, j * noise_scale, k * noise_scale)
                ) * (1 - dist)

    # Normalize
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    # Gamma correction
    volume = np.power(volume, gamma)

    # Scale the volume to the maximum value
    volume = volume * max_value

    # clip the low values of the volume to create a coherent volume
    volume[volume < threshold * max_value] = 0

    # Clip high values
    volume[volume > max_value] = max_value

    # Scale up the volume of volume to size
    volume = scipy.ndimage.zoom(
        volume, np.array(final_shape) / np.array(base_shape), order=order
    )

    return volume.astype(dtype)
