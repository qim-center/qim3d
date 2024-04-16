import numpy as np
import qim3d.processing.filters as filters
import scipy
import skimage
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