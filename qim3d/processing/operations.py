import numpy as np
import scipy
import skimage

import qim3d.processing.filters as filters
from qim3d.io.logger import log


def remove_background(
    vol: np.ndarray,
    median_filter_size: int = 2,
    min_object_radius: int = 3,
    background: str = "dark",
    **median_kwargs
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

def watershed(
        bin_vol: np.ndarray
) -> tuple[np.ndarray, int]:
    """
    Apply watershed segmentation to a binary volume.

    Args:
        bin_vol (np.ndarray): Binary volume to segment.
    
    Returns:
        tuple[np.ndarray, int]: Labeled volume after segmentation, number of objects found.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.cement_128x128x128
        binary = qim3d.processing.filters.gaussian(vol, 2)<60

        qim3d.viz.slices(binary, axis=1)
        ```
        ![operations-watershed_before](assets/screenshots/operations-watershed_before.png)  

        ```python
        labeled_volume, num_labels = qim3d.processing.operations.watershed(binary)

        cmap = qim3d.viz.colormaps.objects(num_labels)
        qim3d.viz.slices(labeled_volume, axis = 1, cmap=cmap)
        ```
        ![operations-watershed_after](assets/screenshots/operations-watershed_after.png)  

    """
    # Compute distance transform of binary volume
    distance= scipy.ndimage.distance_transform_edt(bin_vol)

    # Find peak coordinates in distance transform
    coords = skimage.feature.peak_local_max(distance, labels=bin_vol)

    # Create a mask with peak coordinates
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # Label peaks
    markers, _ = scipy.ndimage.label(mask)

    # Apply watershed segmentation
    labeled_volume = skimage.segmentation.watershed(-distance, markers=markers, mask=bin_vol)
    
    # Extract number of objects found
    num_labels = len(np.unique(labeled_volume))-1
    log.info(f"Total number of objects found: {num_labels}")

    return labeled_volume, num_labels

def fade_mask(
    vol: np.ndarray,
    decay_rate: float = 10,
    ratio: float = 0.5,
    geometry: str = "sphere",
    invert=False,
    axis: int = 0,
    ):
    """
    Apply edge fading to a volume.

    Args:
        vol (np.ndarray): The volume to apply edge fading to.
        decay_rate (float, optional): The decay rate of the fading. Defaults to 10.
        ratio (float, optional): The ratio of the volume to fade. Defaults to 0.
        geometric (str, optional): The geometric shape of the fading. Can be 'spherical' or 'cylindrical'. Defaults to 'spherical'.
        axis (int, optional): The axis along which to apply the fading. Defaults to 0.
    
    Returns:
        np.ndarray: The volume with edge fading applied.

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
        raise ValueError("Axis must be between 0 and the number of dimensions of the volume")

    # Generate the coordinates of each point in the array
    shape = vol.shape
    z, y, x = np.indices(shape)
    
    # Calculate the center of the array
    center = np.array([(s - 1) / 2 for s in shape])
    
    # Calculate the distance of each point from the center
    if geometry == "sphere":
        distance = np.linalg.norm([z - center[0], y - center[1], x - center[2]], axis=0)
    elif geometry == "cilinder":
        distance_list = np.array([z - center[0], y - center[1], x - center[2]])
        # remove the axis along which the fading is not applied
        distance_list = np.delete(distance_list, axis, axis=0)
        distance = np.linalg.norm(distance_list, axis=0)
    else:
        raise ValueError("geometric must be 'spherical' or 'cylindrical'")
    
    # Normalize the distances so that they go from 0 at the center to 1 at the farthest point
    max_distance = np.linalg.norm(center)
    normalized_distance = distance / (max_distance*ratio)
    
    # Apply the decay rate
    faded_distance = normalized_distance ** decay_rate
    
    # Invert the distances to have 1 at the center and 0 at the edges
    fade_array = 1 - faded_distance
    fade_array[fade_array<=0]=0
    
    if invert:
        fade_array = -(fade_array-1)

    # Apply the fading to the volume
    vol_faded = vol * fade_array

    return vol_faded
