import numpy as np

import qim3d.filters as filters


def remove_background(
    vol: np.ndarray,
    median_filter_size: int = 2,
    min_object_radius: int = 3,
    background: str = 'dark',
    **median_kwargs,
) -> np.ndarray:
    """
    Remove background from a volume using a qim3d filters.

    Args:
        vol (np.ndarray): The volume to remove background from.
        median_filter_size (int, optional): The size of the median filter. Defaults to 2.
        min_object_radius (int, optional): The radius of the structuring element for the tophat filter. Defaults to 3.
        background ('dark' or 'bright, optional): The background type. Can be 'dark' or 'bright'. Defaults to 'dark'.
        **median_kwargs (Any): Additional keyword arguments for the Median filter.

    Returns:
        filtered_vol (np.ndarray): The volume with background removed.


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
    return pipeline(vol)


def fade_mask(
    vol: np.ndarray,
    decay_rate: float = 10,
    ratio: float = 0.5,
    geometry: str = 'spherical',
    invert: bool = False,
    axis: int = 0,
    **kwargs,
) -> np.ndarray:
    """
    Apply edge fading to a volume.

    Args:
        vol (np.ndarray): The volume to apply edge fading to.
        decay_rate (float, optional): The decay rate of the fading. Defaults to 10.
        ratio (float, optional): The ratio of the volume to fade. Defaults to 0.5.
        geometry ('spherical' or 'cylindrical', optional): The geometric shape of the fading. Can be 'spherical' or 'cylindrical'. Defaults to 'spherical'.
        invert (bool, optional): Flag for inverting the fading. Defaults to False.
        axis (int, optional): The axis along which to apply the fading. Defaults to 0.
        **kwargs (Any): Additional keyword arguments for the edge fading.

    Returns:
        faded_vol (np.ndarray): The volume with edge fading applied.

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
    if axis < 0 or axis >= vol.ndim:
        error = 'Axis must be between 0 and the number of dimensions of the volume'
        raise ValueError(error)

    # Generate the coordinates of each point in the array
    shape = vol.shape
    z, y, x = np.indices(shape)

    # Store the original maximum value of the volume
    original_max_value = np.max(vol)

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
    vol_faded = vol * fade_array

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
    Overlay an RGB foreground onto an RGB background using alpha blending.

    Args:
        background (numpy.ndarray): The background RGB image.
        foreground (numpy.ndarray): The foreground RGB image (usually masks).
        alpha (float, optional): The alpha value for blending. Defaults to 0.5.
        hide_black (bool, optional): If True, black pixels will have alpha value 0, so the black won't be visible. Used for segmentation where we don't care about background. Defaults to True.

    Returns:
        composite (numpy.ndarray): The composite RGB image with overlaid foreground.

    Raises:
        ValueError: If input images have different shapes.

    Note:
        - The function performs alpha blending to overlay the foreground onto the background.
        - It ensures that the background and foreground have the same first two dimensions (image size matches).
        - It can handle greyscale images, values from 0 to 1, raw values which are negative or bigger than 255.
        - It calculates the maximum projection of the foreground and blends them onto the background.

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
    vol: np.ndarray,
    thickness: int,
) -> np.ndarray:
    """
    Make a volume hollow by applying a mask created by a minimum filter and an xor-gate.

    Args:
        vol (np.ndarray): The volume to hollow.
        thickness (int): The thickness of the shell after hollowing.

    Returns:
        vol_hollowed (np.ndarray): The hollowed volume.

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
    vol_mask_base = vol > 0

    # apply minimum filter to the mask
    vol_eroded = filters.minimum(vol_mask_base, size=thickness)
    # Apply xor to only keep the voxels eroded by the minimum filter
    vol_mask = np.logical_xor(vol_mask_base, vol_eroded)

    # Apply the mask to the original volume to remove 'inner' voxels
    vol_hollowed = vol * vol_mask

    return vol_hollowed
