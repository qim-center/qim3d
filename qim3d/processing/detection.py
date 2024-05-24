import numpy as np
from qim3d.io.logger import log
from skimage.feature import blob_dog

__all__ = ["Blob"]


class Blob:
    """
    Extract blobs from a volume using Difference of Gaussian (DoG) method
    """
    def __init__(
        self,
        background="dark",
        min_sigma=1,
        max_sigma=50,
        sigma_ratio=1.6,
        threshold=0.5,
        overlap=0.5,
        threshold_rel=None,
        exclude_border=False,
    ):
        """
        Initialize the blob detection object

        Args:
            background: 'dark' if background is darker than the blobs, 'bright' if background is lighter than the blobs
            min_sigma: The minimum standard deviation for Gaussian kernel
            max_sigma: The maximum standard deviation for Gaussian kernel
            sigma_ratio: The ratio between the standard deviation of Gaussian Kernels
            threshold: The absolute lower bound for scale space maxima. Reduce this to detect blobs with lower intensities.
            overlap: The fraction of area of two blobs that overlap
            threshold_rel: The relative lower bound for scale space maxima
            exclude_border: If True, exclude blobs that are too close to the border of the image
        """
        self.background = background
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_ratio = sigma_ratio
        self.threshold = threshold
        self.overlap = overlap
        self.threshold_rel = threshold_rel
        self.exclude_border = exclude_border
        self.vol_shape = None
        self.blobs = None

    def detect(self, vol):
        """
        Detect blobs in the volume

        Args:
            vol: The volume to detect blobs in

        Returns:
            blobs: The blobs found in the volume as (p, r, c, radius)

        Example:
            ```python
            import qim3d

            # Get data
            vol = qim3d.examples.cement_128x128x128
            vol_blurred = qim3d.processing.gaussian(vol, sigma=2)

            # Initialize Blob detector
            blob_detector = qim3d.processing.Blob(
                min_sigma=1,
                max_sigma=8,
                threshold=0.001,
                overlap=0.1,
                background="bright"
                )

            # Detect blobs
            blobs = blob_detector.detect(vol_blurred)

            # Visualize results
            qim3d.viz.circles(blobs,vol,alpha=0.8,color='blue')
            ```
            ![blob detection](assets/screenshots/blob_detection.gif)
        """
        self.vol_shape = vol.shape
        if self.background == "bright":
            log.info("Bright background selected, volume will be inverted.")
            vol = np.invert(vol)

        blobs = blob_dog(
            vol,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            sigma_ratio=self.sigma_ratio,
            threshold=self.threshold,
            overlap=self.overlap,
            threshold_rel=self.threshold_rel,
            exclude_border=self.exclude_border,
        )
        blobs[:, 3] = blobs[:, 3] * np.sqrt(3)  # Change sigma to radius
        self.blobs = blobs
        return self.blobs
    
    def get_mask(self):
        '''
        Retrieve a binary volume with the blobs marked as True

        Returns:
            binary_volume: A binary volume with the blobs marked as True

        Example:
            ```python
            import qim3d

            # Get data
            vol = qim3d.examples.cement_128x128x128
            vol_blurred = qim3d.processing.gaussian(vol, sigma=2)

            # Initialize Blob detector
            blob_detector = qim3d.processing.Blob(
                min_sigma=1,
                max_sigma=8,
                threshold=0.001,
                overlap=0.1,
                background="bright"
                )

                
            # Detect blobs
            blobs = blob_detector.detect(vol_blurred)

            # Get mask and visualize
            mask = blob_detector.get_mask()
            qim3d.viz.slicer(mask)
            ```
            ![blob detection](assets/screenshots/blob_get_mask.gif)
        '''
        binary_volume = np.zeros(self.vol_shape, dtype=bool)

        for z, y, x, radius in self.blobs:
            # Calculate the bounding box around the blob
            z_start = max(0, int(z - radius))
            z_end = min(self.vol_shape[0], int(z + radius) + 1)
            y_start = max(0, int(y - radius))
            y_end = min(self.vol_shape[1], int(y + radius) + 1)
            x_start = max(0, int(x - radius))
            x_end = min(self.vol_shape[2], int(x + radius) + 1)

            z_indices, y_indices, x_indices = np.indices((z_end - z_start, y_end - y_start, x_end - x_start))
            z_indices += z_start
            y_indices += y_start
            x_indices += x_start

            # Calculate distances from the center of the blob to voxels within the bounding box
            dist = np.sqrt((x_indices - x)**2 + (y_indices - y)**2 + (z_indices - z)**2)

            binary_volume[z_start:z_end, y_start:y_end, x_start:x_end][dist <= radius] = True

        return binary_volume


