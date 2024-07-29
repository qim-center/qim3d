""" Blob detection using Difference of Gaussian (DoG) method """

import numpy as np
from qim3d.utils.logger import log

def blob_detection(
    vol: np.ndarray,
    background: str = "dark",
    min_sigma: float = 1,
    max_sigma: float = 50,
    sigma_ratio: float = 1.6,
    threshold: float = 0.5,
    overlap: float = 0.5,
    threshold_rel: float = None,
    exclude_border: bool = False,
) -> np.ndarray:
    """
    Extract blobs from a volume using Difference of Gaussian (DoG) method, and retrieve a binary volume with the blobs marked as True

    Args:
        vol: The volume to detect blobs in
        background: 'dark' if background is darker than the blobs, 'bright' if background is lighter than the blobs
        min_sigma: The minimum standard deviation for Gaussian kernel
        max_sigma: The maximum standard deviation for Gaussian kernel
        sigma_ratio: The ratio between the standard deviation of Gaussian Kernels
        threshold: The absolute lower bound for scale space maxima. Reduce this to detect blobs with lower intensities
        overlap: The fraction of area of two blobs that overlap
        threshold_rel: The relative lower bound for scale space maxima
        exclude_border: If True, exclude blobs that are too close to the border of the image

    Returns:
        blobs: The blobs found in the volume as (p, r, c, radius)
        binary_volume: A binary volume with the blobs marked as True

    Example:
            ```python
            import qim3d

            # Get data
            vol = qim3d.examples.cement_128x128x128
            vol_blurred = qim3d.processing.gaussian(vol, sigma=2)

            # Detect blobs, and get binary mask
            blobs, mask = qim3d.processing.blob_detection(
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
            ![blob detection](assets/screenshots/blob_detection.gif)

            ```python
            # Visualize binary mask
            qim3d.viz.slicer(mask)
            ```
            ![blob detection](assets/screenshots/blob_get_mask.gif)
    """
    from skimage.feature import blob_dog

    if background == "bright":
        log.info("Bright background selected, volume will be inverted.")
        vol = np.invert(vol)

    blobs = blob_dog(
        vol,
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
    vol_shape = vol.shape
    binary_volume = np.zeros(vol_shape, dtype=bool)

    for z, y, x, radius in blobs:
        # Calculate the bounding box around the blob
        z_start = max(0, int(z - radius))
        z_end = min(vol_shape[0], int(z + radius) + 1)
        y_start = max(0, int(y - radius))
        y_end = min(vol_shape[1], int(y + radius) + 1)
        x_start = max(0, int(x - radius))
        x_end = min(vol_shape[2], int(x + radius) + 1)

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

        binary_volume[z_start:z_end, y_start:y_end, x_start:x_end][
            dist <= radius
        ] = True

    return blobs, binary_volume
