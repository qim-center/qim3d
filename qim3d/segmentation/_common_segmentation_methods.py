import numpy as np

from qim3d.utils._logger import log

def watershed(binary_volume: np.ndarray, min_distance: int = 5) -> tuple[np.ndarray, int]:
    """
    Apply watershed segmentation to a binary volume.

    Args:
        binary_volume (np.ndarray): Binary volume to segment. The input should be a 3D binary image where non-zero elements
                              represent the objects to be segmented.
        min_distance (int): Minimum number of pixels separating peaks in the distance transform. Peaks that are
                            too close will be merged, affecting the number of segmented objects. Default is 5.

    Returns:
        labeled_vol (np.ndarray): A 3D array of the same shape as the input `binary_volume`, where each segmented object is assigned a unique integer label.
        num_labels (int): The total number of unique objects found in the labeled volume.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.cement_128x128x128
        binary_volume = qim3d.filters.gaussian(vol, sigma = 2)<60

        fig1 = qim3d.viz.slices_grid(binary_volume, slice_axis=1, display_figure=True)
        ```
        ![operations-watershed_before](../../assets/screenshots/operations-watershed_before.png)

        ```python
        labeled_volume, num_labels = qim3d.segmentation.watershed(binary_volume)

        cmap = qim3d.viz.colormaps.segmentation(num_labels)
        fig2 = qim3d.viz.slices_grid(labeled_volume, slice_axis=1, color_map=cmap, display_figure=True)
        ```
        ![operations-watershed_after](../../assets/screenshots/operations-watershed_after.png)

    """
    import scipy
    import skimage

    if len(np.unique(binary_volume)) > 2:
        raise ValueError(
            'binary_volume has to be binary volume - it must contain max 2 unique values.'
        )

    # Compute distance transform of binary volume
    distance = scipy.ndimage.distance_transform_edt(binary_volume)

    # Find peak coordinates in distance transform
    coords = skimage.feature.peak_local_max(
        distance, min_distance=min_distance, labels=binary_volume
    )

    # Create a mask with peak coordinates
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # Label peaks
    markers, _ = scipy.ndimage.label(mask)

    # Apply watershed segmentation
    labeled_volume = skimage.segmentation.watershed(
        -distance, markers=markers, mask=binary_volume
    )

    # Extract number of objects found
    num_labels = len(np.unique(labeled_volume)) - 1
    log.info(f'Total number of objects found: {num_labels}')

    return labeled_volume, num_labels
