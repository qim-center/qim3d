import numpy as np
import qim3d.processing.filters as filters


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
