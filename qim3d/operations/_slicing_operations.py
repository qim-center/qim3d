import numpy as np
from interactive_unet.slicer import Slicer


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
        np.ndarray: A 2D slice of the specified width and length.

    """

    if seed is not None:
        np.random.seed(seed)

    # Build the slicer for this volume
    slicer = Slicer(volume.shape)

    # Randomize orientation and origin
    slicer.randomize(sampling_mode='random')

    # Extract square slice
    slice2d = slicer.get_slice(volume, slice_width=width, order=1)

    return slice2d
