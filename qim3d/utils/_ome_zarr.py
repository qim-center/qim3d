from zarr.util import normalize_chunks, normalize_dtype, normalize_shape
import numpy as np

def get_chunk_size(shape:tuple, dtype):
    """
    How the chunk size is computed in zarr.storage.init_array_metadata which is ran in the chain of functions we use 
    in qim3d.io.export_ome_zarr function

    Parameters
    ----------
    - shape: shape of the data
    - dtype: dtype of the data
    """
    object_codec = None
    dtype, object_codec = normalize_dtype(dtype, object_codec)
    shape = normalize_shape(shape) + dtype.shape
    dtype = dtype.base
    chunks = None
    chunks = normalize_chunks(chunks, shape, dtype.itemsize)
    return chunks


def get_n_chunks(shapes:tuple, dtypes:tuple):
    """
    Estimates how many chunks we will use in advence so we can pass this number to a progress bar and track how many
    have been already written to disk

    Parameters
    ----------
    - shapes: list of shapes of the data for each scale
    - dtype: dtype of the data
    """
    n_chunks = 0
    for shape, dtype in zip(shapes, dtypes):
        chunk_size = np.array(get_chunk_size(shape, dtype))
        shape = np.array(shape)
        ratio = shape/chunk_size
        n_chunks += np.prod(ratio)
    return int(n_chunks)

