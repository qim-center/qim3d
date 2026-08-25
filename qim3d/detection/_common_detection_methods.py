"""Blob detection using Difference of Gaussian (DoG) method"""

import numpy as np

from qim3d.utils._logger import log

def blobs(
    volume: np.ndarray,
    background: str = 'dark',
    min_sigma: float = 1,
    max_sigma: float = 50,
    sigma_ratio: float = 1.6,
    threshold: float = 0.5,
    overlap: float = 0.5,
    threshold_rel: float = None,
    exclude_border: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detects spherical objects (blobs) in a 3D volume using the Difference of Gaussian (DoG) method.

    This function is essential for particle analysis, cell counting, or void detection. It identifies local maxima in the scale-space of the image to locate blobs of various sizes. In addition to returning the precise coordinates and radii of the detected features, it generates a binary mask where the spherical regions are fully reconstructed.

    Args:
        volume (np.ndarray): The 3D input volume.
        background (str, optional): The intensity of the background relative to the objects.

            * **'dark'**: Use if the background is darker than the objects (detects bright spots).
            * **'bright'**: Use if the background is lighter than the objects (detects dark spots/pores).

        min_sigma (float, optional): The minimum standard deviation for the Gaussian kernel. Controls the smallest blob size to detect. Defaults to 1.
        max_sigma (float, optional): The maximum standard deviation for the Gaussian kernel. Controls the largest blob size to detect. Defaults to 50.
        sigma_ratio (float, optional): The ratio between the standard deviations of consecutive Gaussian kernels in the scale space. Defaults to 1.6.
        threshold (float, optional): The absolute lower bound for intensity. reduce this value to detect fainter blobs. Defaults to 0.5.
        overlap (float, optional): The maximum fraction of spatial overlap allowed between two blobs. If two blobs overlap by more than this fraction, the smaller one is eliminated. Defaults to 0.5.
        threshold_rel (float or None, optional): Minimum intensity of peaks, calculated as `max(image) * threshold_rel`. Defaults to `None`.
        exclude_border (bool, optional): If `True`, ignores blobs found near the border of the volume. Defaults to `False`.

    Returns:
        results (tuple[np.ndarray, np.ndarray]):
            A tuple containing the detection metrics and the resulting segmentation.

            * **blobs**: An array of shape `(N, 4)` containing the parameters `(z, y, x, radius)` for each detected blob.
            * **binary_volume**: A boolean mask of the same shape as `volume`, where the detected spherical regions are marked as `True`.

    Example:
            ```python
            import qim3d
            import qim3d.detection

            # Get data
            vol = qim3d.examples.cement_128x128x128
            vol_blurred = qim3d.filters.gaussian(vol, sigma=2)

            # Detect blobs, and get binary_volume
            blobs, binary_volume = qim3d.detection.blobs(
                vol_blurred,
                min_sigma=1,
                max_sigma=8,
                threshold=0.001,
                overlap=0.1,
                background="bright"
                )

            # Visualize detected blobs
            qim3d.viz.circles(blobs, vol, alpha=0.8, color='blue')
            ```
            ![blob detection](../../assets/screenshots/blob_detection.gif)

            ```python
            # Visualize binary binary_volume
            qim3d.viz.slicer(binary_volume)
            ```
            ![blob detection](../../assets/screenshots/blob_get_mask.gif)
    """
    from skimage.feature import blob_dog

    if background == 'bright':
        log.info('Bright background selected, volume will be inverted.')
        volume = np.invert(volume)

    blobs = blob_dog(
        volume,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_ratio=sigma_ratio,
        threshold=threshold,
        overlap=overlap,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
    )

    # Change sigma to radius
    blobs[:, 3] = blobs[:, 3] * np.sqrt(3)

    # Create binary mask of detected blobs
    volume_shape = volume.shape
    binary_volume = np.zeros(volume_shape, dtype=bool)

    for z, y, x, radius in blobs:
        # Calculate the bounding box around the blob
        z_start = max(0, int(z - radius))
        z_end = min(volume_shape[0], int(z + radius) + 1)
        y_start = max(0, int(y - radius))
        y_end = min(volume_shape[1], int(y + radius) + 1)
        x_start = max(0, int(x - radius))
        x_end = min(volume_shape[2], int(x + radius) + 1)

        z_indices, y_indices, x_indices = np.indices(
            (z_end - z_start, y_end - y_start, x_end - x_start)
        )
        z_indices += z_start
        y_indices += y_start
        x_indices += x_start

        # Calculate distances from the center of the blob to voxels within the bounding box
        dist = np.sqrt(
            (x_indices - x) ** 2 + (y_indices - y) ** 2 + (z_indices - z) ** 2
        )

        binary_volume[z_start:z_end, y_start:y_end, x_start:x_end][dist <= radius] = (
            True
        )

    return blobs, binary_volume
