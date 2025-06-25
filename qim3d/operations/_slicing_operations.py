import numpy as np
from interactive_unet.slicer import Slicer
from scipy.ndimage import map_coordinates


class RectangularSlicer(Slicer):
    def get_slice(
        self, volume: np.ndarray, width: int, length: int, axis: int = 0, order: int = 1
    ) -> np.ndarray:
        """Override to accept separate width/length for the slice."""
        # compute the interpolation coords once, but allow distinct dims
        coords = self.get_interpolation_coords(slice_width=width)
        # coords has shape (3, width, width); we need (3, width, length)
        # so rebuild it replacing one axis:
        start_w = int(-np.floor(width / 2))
        start_l = int(-np.floor(length / 2))
        idx_w = np.linspace(start_w, start_w + width - 1, width)
        idx_l = np.linspace(start_l, start_l + length - 1, length)
        # basis vectors:
        v = self.v[:, None, None]
        w = self.w[:, None, None]
        origin = self.origin[:, None, None]
        # build new 3×width×length coords:
        x = v * idx_w[None, :, None] + w * idx_l[None, None, :] + origin
        y = (
            self.u[:, None, None] * idx_w[None, :, None]
            + w * idx_l[None, None, :]
            + origin
        )
        z = (
            self.u[:, None, None] * idx_w[None, :, None]
            + self.v[:, None, None] * idx_l[None, None, :]
            + origin
        )
        new_coords = np.stack([x, y, z], axis=0)
        # interpolate
        if volume.ndim > 3:
            # color channels last
            return np.moveaxis(
                np.array(
                    [
                        map_coordinates(volume[..., c], new_coords[axis], order=order)
                        for c in range(volume.shape[-1])
                    ]
                ),
                0,
                -1,
            )
        else:
            return map_coordinates(volume, new_coords[axis], order=order)


def get_random_slice(
    volume: np.ndarray, width: int, length: int, seed: int | None = None
) -> np.ndarray:
    """
    Get a random slice from a 3D volume.

    Args:
        volume (np.ndarray): The 3D volume from which to extract the slice.
        width (int): The width of the slice.
        length (int): The length of the slice.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        slice2d (np.ndarray): A 2D slice of the specified width and length.

    Example:
        ```python
        import qim3d
        downloader = qim3d.io.Downloader()
        data = downloader.Cowry_Shell.Cowry_DOWNSAMPLED(load_file=True)
        slice2d = qim3d.operations.get_random_slice(data, width=64, length=100)
        ```

    """

    if seed is not None:
        np.random.seed(seed)

    # Build the slicer for this volume
    slicer = RectangularSlicer(volume.shape)

    # Randomize orientation and origin
    slicer.randomize(sampling_mode='random')

    # Extract square slice
    slice2d = slicer.get_slice(volume, width=width, length=length)

    return slice2d
