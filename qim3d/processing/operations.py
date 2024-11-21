import numpy as np
import qim3d.processing.filters as filters
from qim3d.utils.logger import log


def remove_background(
    vol: np.ndarray,
    median_filter_size: int = 2,
    min_object_radius: int = 3,
    background: str = "dark",
    **median_kwargs,
) -> np.ndarray:
    """
    Remove background from a volume using a qim3d filters.

    Args:
        vol (np.ndarray): The volume to remove background from.
        median_filter_size (int, optional): The size of the median filter. Defaults to 2.
        min_object_radius (int, optional): The radius of the structuring element for the tophat filter. Defaults to 3.
        background (str, optional): The background type. Can be 'dark' or 'bright'. Defaults to 'dark'.
        **median_kwargs: Additional keyword arguments for the Median filter.

    Returns:
        np.ndarray: The volume with background removed.


    Example:
        ```python
        import qim3d

        vol = qim3d.examples.cement_128x128x128
        qim3d.viz.slices(vol, vmin=0, vmax=255)
        ```
        ![operations-remove_background_before](assets/screenshots/operations-remove_background_before.png)

        ```python
        vol_filtered  = qim3d.processing.operations.remove_background(vol,
                                                              min_object_radius=3,
                                                              background="bright")
        qim3d.viz.slices(vol_filtered, vmin=0, vmax=255)
        ```
        ![operations-remove_background_after](assets/screenshots/operations-remove_background_after.png)
    """

    # Create a pipeline with a median filter and a tophat filter
    pipeline = filters.Pipeline(
        filters.Median(size=median_filter_size, **median_kwargs),
        filters.Tophat(radius=min_object_radius, background=background),
    )

    # Apply the pipeline to the volume
    return pipeline(vol)


def watershed(bin_vol: np.ndarray, min_distance: int = 5) -> tuple[np.ndarray, int]:
    """
    Apply watershed segmentation to a binary volume.

    Args:
        bin_vol (np.ndarray): Binary volume to segment. The input should be a 3D binary image where non-zero elements 
                              represent the objects to be segmented.
        min_distance (int): Minimum number of pixels separating peaks in the distance transform. Peaks that are 
                            too close will be merged, affecting the number of segmented objects. Default is 5.

    Returns:
        tuple[np.ndarray, int]: 
            - Labeled volume (np.ndarray): A 3D array of the same shape as the input `bin_vol`, where each segmented object
              is assigned a unique integer label.
            - num_labels (int): The total number of unique objects found in the labeled volume.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.cement_128x128x128
        binary = qim3d.processing.filters.gaussian(vol, sigma = 2)<60

        qim3d.viz.slices(binary, axis=1)
        ```
        ![operations-watershed_before](assets/screenshots/operations-watershed_before.png)

        ```python
        labeled_volume, num_labels = qim3d.processing.operations.watershed(binary)

        cmap = qim3d.viz.colormaps.objects(num_labels)
        qim3d.viz.slices(labeled_volume, axis = 1, cmap = cmap)
        ```
        ![operations-watershed_after](assets/screenshots/operations-watershed_after.png)

    """
    import skimage
    import scipy

    # Compute distance transform of binary volume
    distance = scipy.ndimage.distance_transform_edt(bin_vol)

    # Find peak coordinates in distance transform
    coords = skimage.feature.peak_local_max(
        distance, min_distance=min_distance, labels=bin_vol
    )

    # Create a mask with peak coordinates
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # Label peaks
    markers, _ = scipy.ndimage.label(mask)

    # Apply watershed segmentation
    labeled_volume = skimage.segmentation.watershed(
        -distance, markers=markers, mask=bin_vol
    )

    # Extract number of objects found
    num_labels = len(np.unique(labeled_volume)) - 1
    log.info(f"Total number of objects found: {num_labels}")

    return labeled_volume, num_labels


def fade_mask(
    vol: np.ndarray,
    decay_rate: float = 10,
    ratio: float = 0.5,
    geometry: str = "spherical",
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
        geometry (str, optional): The geometric shape of the fading. Can be 'spherical' or 'cylindrical'. Defaults to 'spherical'.
        invert (bool, optional): Flag for inverting the fading. Defaults to False.
        axis (int, optional): The axis along which to apply the fading. Defaults to 0.
        **kwargs: Additional keyword arguments for the edge fading.

    Returns:
        vol_faded (np.ndarray): The volume with edge fading applied.

    Example:
        ```python
        import qim3d
        qim3d.viz.vol(vol)
        ```
        Image before edge fading has visible artifacts from the support. Which obscures the object of interest.
        ![operations-edge_fade_before](assets/screenshots/operations-edge_fade_before.png)

        ```python
        import qim3d
        vol_faded = qim3d.processing.operations.edge_fade(vol, decay_rate=4, ratio=0.45, geometric='cylindrical')
        qim3d.viz.vol(vol_faded)
        ```
        Afterwards the artifacts are faded out, making the object of interest more visible for visualization purposes.
        ![operations-edge_fade_after](assets/screenshots/operations-edge_fade_after.png)

    """
    if 0 > axis or axis >= vol.ndim:
        raise ValueError(
            "Axis must be between 0 and the number of dimensions of the volume"
        )

    # Generate the coordinates of each point in the array
    shape = vol.shape
    z, y, x = np.indices(shape)

    # Store the original maximum value of the volume
    original_max_value = np.max(vol)

    # Calculate the center of the array
    center = np.array([(s - 1) / 2 for s in shape])

    # Calculate the distance of each point from the center
    if geometry == "spherical":
        distance = np.linalg.norm([z - center[0], y - center[1], x - center[2]], axis=0)
    elif geometry == "cylindrical":
        distance_list = np.array([z - center[0], y - center[1], x - center[2]])
        # remove the axis along which the fading is not applied
        distance_list = np.delete(distance_list, axis, axis=0)
        distance = np.linalg.norm(distance_list, axis=0)
    else:
        raise ValueError("Geometry must be 'spherical' or 'cylindrical'")
    
    # Compute the maximum distance from the center
    max_distance = np.linalg.norm(center)
    
    # Compute ratio to make synthetic blobs exactly cylindrical
    # target_max_normalized_distance = 1.4 works well to make the blobs cylindrical
    if "target_max_normalized_distance" in kwargs:
        target_max_normalized_distance = kwargs["target_max_normalized_distance"]
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
    background: np.ndarray, foreground: np.ndarray, alpha: float = 0.5, hide_black:bool = True,
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

    def to_uint8(image:np.ndarray):
        if np.min(image) < 0:
            image = image - np.min(image)

        maxim = np.max(image)
        if maxim > 255:
            image = (image / maxim)*255
        elif maxim <= 1:
            image = image*255

        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, -1)
        elif image.ndim == 3:
            image = image[..., :3] # Ignoring alpha channel
        else:
            raise ValueError(F'Input image can not have higher dimension than 3. Yours have {image.ndim}')
        
        return image.astype(np.uint8)
        
    background = to_uint8(background)
    foreground = to_uint8(foreground)

    # Ensure both images have the same shape
    if background.shape != foreground.shape:
        raise ValueError(F"Input images must have the same first two dimensions. But background is of shape {background.shape} and foreground is of shape {foreground.shape}")

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
        raise ValueError(F'Alpha has to be positive number. You used {alpha}')
    elif alpha > 1:
        alpha = 1
    
    # If the pixel is black, its alpha value is set to 0, so it has no effect on the image
    if hide_black:
        alpha = np.full((background.shape[0], background.shape[1],1), alpha)
        alpha[np.apply_along_axis(lambda x: (x == [0,0,0]).all(), axis = 2, arr = foreground)] = 0

    composite = background * (1 - alpha) + foreground * alpha
    composite = np.clip(composite, 0, 255).astype("uint8")

    return composite.astype("uint8")