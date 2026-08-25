import numpy as np

import qim3d.filters as filters


def remove_background(
    volume: np.ndarray,
    median_filter_size: int = 2,
    min_object_radius: int = 3,
    background: str = 'dark',
    **median_kwargs,
) -> np.ndarray:
    """
    Applies a background correction pipeline using median filtering and morphological operations.

    This function acts as a convenience wrapper for a sequential processing pipeline designed to smooth the image and suppress clutter. It performs two distinct operations:

    1.  **Median Filter:** Reduces high-frequency impulse noise (salt-and-pepper) using a kernel of size `median_filter_size`.
    2.  **Morphological Opening:** Removes bright features (or dark, if `background='bright'`) that are smaller than the `min_object_radius`. This effectively separates large structural components from small background artifacts or texture.

    Args:
        volume (np.ndarray): The input volume to process.
        median_filter_size (int, optional): The size of the kernel for the initial median denoising step. Defaults to 2.
        min_object_radius (int, optional): The radius of the structuring element (ball) used for the morphological operation. Details smaller than this size are removed. Defaults to 3.
        background (str, optional): The intensity of the background relative to the objects.

            * **'dark'**: Use for bright objects on a dark background.
            * **'bright'**: Use for dark objects on a bright background (the volume is inverted during processing).

        **median_kwargs (Any): Additional keyword arguments passed to the underlying `Median` filter.

    Returns:
        filtered_volume (np.ndarray):
            The processed volume with background clutter and noise suppressed.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.cement_128x128x128
        fig1 = qim3d.viz.slices_grid(vol, value_min=0, value_max=255, num_slices=5, display_figure=True)
        ```
        ![operations-remove_background_before](../../assets/screenshots/operations-remove_background_before.png)

        ```python
        vol_filtered  = qim3d.operations.remove_background(vol,
                                                              min_object_radius=3,
                                                              background="bright")
        fig2 = qim3d.viz.slices_grid(vol_filtered, value_min=0, value_max=255, num_slices=5, display_figure=True)
        ```
        ![operations-remove_background_after](../../assets/screenshots/operations-remove_background_after.png)
    """

    # Create a pipeline with a median filter and a tophat filter
    pipeline = filters.Pipeline(
        filters.Median(size=median_filter_size, **median_kwargs),
        filters.Tophat(radius=min_object_radius, background=background),
    )

    # Apply the pipeline to the volume
    return pipeline(volume)


def fade_mask(
    volume: np.ndarray,
    decay_rate: float = 10,
    ratio: float = 0.5,
    geometry: str = 'spherical',
    invert: bool = False,
    axis: int = 0,
    **kwargs,
) -> np.ndarray:
    """
    Applies a soft attenuation mask (vignetting) to the volume to suppress boundary artifacts.

    This function multiplies the input volume by a generated mask that decays from the center outwards based on a power-law profile. It is commonly used to remove reconstruction artifacts at the edges of a scan or to isolate a central Region of Interest (ROI) by suppressing peripheral data. The shape of the mask can be spherical or cylindrical.

    Args:
        volume (np.ndarray): The 3D input volume.
        decay_rate (float, optional): The exponent for the power-law decay. Higher values create a "flatter" central region with a sharper drop-off near the mask edge, while lower values cause a more gradual fade from the center. Defaults to 10.
        ratio (float, optional): The effective radius of the non-zero mask region relative to the volume size. Defaults to 0.5.
        geometry (str, optional): The geometric shape of the mask.

            * **'spherical'**: Fades in all directions from the volume center.
            * **'cylindrical'**: Fades radially from a central axis (defined by `axis`), maintaining constant intensity along that axis. Defaults to 'spherical'.

        invert (bool, optional): If `True`, inverts the mask (fades the center and keeps the edges). Defaults to `False`.
        axis (int, optional): The axis of alignment for the cylinder if `geometry='cylindrical'`. Defaults to 0.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        faded_vol (np.ndarray):
            The volume with the attenuation mask applied, renormalized to match the original maximum intensity.

    Raises:
        ValueError: If `axis` is invalid or `geometry` is not 'spherical' or 'cylindrical'.

    Example:
        ```python
        import qim3d
        vol = qim3d.examples.fly_150x256x256
        qim3d.viz.volumetric(vol)
        ```
        Image before edge fading has visible artifacts, which obscures the object of interest.
        <iframe src="https://platform.qim.dk/k3d/fly.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        vol_faded = qim3d.operations.fade_mask(vol, geometric='cylindrical', decay_rate=5, ratio=0.65, axis=1)
        qim3d.viz.volumetric(vol_faded)
        ```
        Afterwards the artifacts are faded out, making the object of interest more visible for visualization purposes.
        <iframe src="https://platform.qim.dk/k3d/fly_faded.html" width="100%" height="500" frameborder="0"></iframe>
    """
    if axis < 0 or axis >= volume.ndim:
        error = 'Axis must be between 0 and the number of dimensions of the volume'
        raise ValueError(error)

    # Generate the coordinates of each point in the array
    shape = volume.shape
    z, y, x = np.indices(shape)

    # Store the original maximum value of the volume
    original_max_value = np.max(volume)

    # Calculate the center of the array
    center = np.array([(s - 1) / 2 for s in shape])

    # Calculate the distance of each point from the center
    if geometry == 'spherical':
        distance = np.linalg.norm([z - center[0], y - center[1], x - center[2]], axis=0)
    elif geometry == 'cylindrical':
        distance_list = np.array([z - center[0], y - center[1], x - center[2]])
        # remove the axis along which the fading is not applied
        distance_list = np.delete(distance_list, axis, axis=0)
        distance = np.linalg.norm(distance_list, axis=0)
    else:
        error = "Geometry must be 'spherical' or 'cylindrical'"
        raise ValueError(error)

    # Compute the maximum distance from the center
    max_distance = np.linalg.norm(center)

    # Compute ratio to make synthetic blobs exactly cylindrical
    # target_max_normalized_distance = 1.4 works well to make the blobs cylindrical
    if 'target_max_normalized_distance' in kwargs:
        target_max_normalized_distance = kwargs['target_max_normalized_distance']
        ratio = np.max(distance) / (target_max_normalized_distance * max_distance)

    # Normalize the distances so that they go from 0 at the center to 1 at the farthest point
    normalized_distance = distance / (max_distance * ratio)

    # Apply the decay rate
    faded_distance = normalized_distance**decay_rate

    # Invert the distances to have 1 at the center and 0 at the edges
    fade_array = 1 - faded_distance
    fade_array[fade_array <= 0] = 0

    if invert:
        fade_array = -(fade_array - 1)

    # Apply the fading to the volume
    vol_faded = volume * fade_array

    # Normalize the volume to retain the original maximum value
    vol_normalized = vol_faded * (original_max_value / np.max(vol_faded))

    return vol_normalized


def overlay_rgb_images(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha: float = 0.5,
    hide_black: bool = True,
) -> np.ndarray:
    """
    Composites a foreground image onto a background using alpha blending.

    This function overlays a mask or secondary image (foreground) onto a base image (background). It automatically normalizes inputs (handling 2D/3D, float/integer, and range mismatches) to ensure compatible 8-bit RGB formats before blending. A key feature is the conditional transparency (`hide_black`), which treats black pixels in the foreground as fully transparent, making it ideal for overlaying sparse segmentation masks without obscuring the rest of the image.

    Args:
        background (np.ndarray): The base image.
        foreground (np.ndarray): The overlay image (e.g., a segmentation mask or heatmap). Must match the spatial dimensions of the background.
        alpha (float, optional): The global opacity of the foreground, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Defaults to 0.5.
        hide_black (bool, optional): If `True`, forces the alpha channel to 0 for all perfectly black pixels `[0, 0, 0]` in the foreground. This prevents the background of a sparse mask from darkening the base image. Defaults to `True`.

    Returns:
        composite (np.ndarray):
            The resulting 8-bit RGB image after blending.

    Raises:
        ValueError: If the spatial dimensions (height/width) of the input images do not match.
    """

    def to_uint8(image: np.ndarray) -> np.ndarray:
        if np.min(image) < 0:
            image = image - np.min(image)

        maxim = np.max(image)
        if maxim > 255:
            image = (image / maxim) * 255
        elif maxim <= 1:
            image = image * 255

        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, -1)
        elif image.ndim == 3:
            image = image[..., :3]  # Ignoring alpha channel
        else:
            error = f'Input image can not have higher dimension than 3. Yours have {image.ndim}'
            raise ValueError(error)

        return image.astype(np.uint8)

    background = to_uint8(background)
    foreground = to_uint8(foreground)

    # Ensure both images have the same shape
    if background.shape != foreground.shape:
        error = f'Input images must have the same first two dimensions. But background is of shape {background.shape} and foreground is of shape {foreground.shape}'
        raise ValueError(error)

    # Perform alpha blending
    foreground_max_projection = np.amax(foreground, axis=2)
    foreground_max_projection = np.stack((foreground_max_projection,) * 3, axis=-1)

    # Normalize if we have something
    if np.max(foreground_max_projection) > 0:
        foreground_max_projection = foreground_max_projection / np.max(
            foreground_max_projection
        )
    # Check alpha validity
    if alpha < 0:
        error = f'Alpha has to be positive number. You used {alpha}'
        raise ValueError(error)
    elif alpha > 1:
        alpha = 1

    # If the pixel is black, its alpha value is set to 0, so it has no effect on the image
    if hide_black:
        alpha = np.full((background.shape[0], background.shape[1], 1), alpha)
        alpha[
            np.apply_along_axis(
                lambda x: (x == [0, 0, 0]).all(), axis=2, arr=foreground
            )
        ] = 0

    composite = background * (1 - alpha) + foreground * alpha
    composite = np.clip(composite, 0, 255).astype('uint8')

    return composite.astype('uint8')


def make_hollow(
    volume: np.ndarray,
    thickness: int,
) -> np.ndarray:
    """
    Constructs a hollow shell from a solid 3D volume.

    This function isolates the outer boundary layer of an object. It achieves this by performing a morphological erosion (using a minimum filter) to identify the inner core, which is then subtracted from the original volume via a logical XOR operation. The result is a shell that retains the original intensity values, while the interior is set to zero.

    Args:
        volume (np.ndarray): The input 3D volume. Non-zero values are treated as the object.
        thickness (int): The width of the resulting shell in voxels. This value determines the size of the erosion kernel used to define the hollow core.

    Returns:
        vol_hollowed (np.ndarray):
            The processed volume containing only the outer shell of the object.

    Example:
        ```python
        import qim3d

        # Generate volume and visualize it
        vol = qim3d.generate.volume(noise_scale = 0.01)
        qim3d.viz.slicer(vol)
        ```
        ![synthetic_collection](../../assets/screenshots/hollow_slicer_1.gif)
        ```python
        # Hollow volume and visualize it
        vol_hollowed = qim3d.operations.make_hollow(vol, thickness=10)
        qim3d.viz.slicer(vol_hollowed)
        ```
        ![synthetic_collection](../../assets/screenshots/hollow_slicer_2.gif)
    """
    # Create base mask
    vol_mask_base = volume > 0

    # apply minimum filter to the mask
    vol_eroded = filters.minimum(vol_mask_base, size=thickness)
    # Apply xor to only keep the voxels eroded by the minimum filter
    vol_mask = np.logical_xor(vol_mask_base, vol_eroded)

    # Apply the mask to the original volume to remove 'inner' voxels
    vol_hollowed = volume * vol_mask

    return vol_hollowed
