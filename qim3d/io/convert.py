from itertools import product

import numpy as np
import tifffile as tiff
import zarr
from tqdm import tqdm


def convert_tif_to_zarr(tif_path, zarr_path, chunks=(64, 64, 64)):
    """ Convert a tiff file to a zarr file

    Args:
        tif_path (str): path to the tiff file
        zarr_path (str): path to the zarr file
        chunks (tuple, optional): chunk size for the zarr file. Defaults to (64, 64, 64).

    Returns:
        zarr.core.Array: zarr array containing the data from the tiff file
    """
    vol = tiff.memmap(tif_path)
    z = zarr.open(zarr_path, mode='w', shape=vol.shape, chunks=chunks, dtype=vol.dtype)
    chunk_shape = tuple((s + c - 1) // c for s, c in zip(z.shape, z.chunks))
    for chunk_indices in tqdm(product(*[range(n) for n in chunk_shape]), total=np.prod(chunk_shape)):
        slices = tuple(slice(c * i, min(c * (i + 1), s))
                    for s, c, i in zip(z.shape, z.chunks, chunk_indices))
        temp_data = vol[slices]
        # The assignment takes 99% of the cpu-time
        z.blocks[chunk_indices] = temp_data

    return z

def convert_npy_to_zarr(npy_path, zarr_path, shape, dtype=np.float32, chunks=(64, 64, 64)):
    """ Convert a numpy file to a zarr file

    Args:
        npy_path (str): path to the numpy file
        zarr_path (str): path to the zarr file
        chunks (tuple, optional): chunk size for the zarr file. Defaults to (64, 64, 64).

    Returns:
        zarr.core.Array: zarr array containing the data from the numpy file
    """
    vol = np.memmap(npy_path, dtype=dtype, mode='r', shape=shape)
    z = zarr.open(zarr_path, mode='w', shape=vol.shape, chunks=chunks, dtype=vol.dtype)
    z[:] = vol[:]

    return z

def convert_zarr_to_tif(zarr_path, tif_path):
    """ Convert a zarr file to a tiff file

    Args:
        zarr_path (str): path to the zarr file
        tif_path (str): path to the tiff file

    returns:
        None
    """
    z = zarr.open(zarr_path)
    tiff.imwrite(tif_path, z)