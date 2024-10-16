"""
Exporting data to different formats.
"""

import os
import math
import shutil
import logging

import numpy as np
import zarr
import tqdm
from ome_zarr.io import parse_url
from ome_zarr.writer import (
    write_image,
    _create_mip,
    write_multiscale,
    CurrentFormat,
    Format,
)
from ome_zarr.scale import dask_resize
from ome_zarr.reader import Reader
from ome_zarr import scale
from scipy.ndimage import zoom
from typing import Any, Callable, Iterator, List, Tuple, Union
import dask.array as da
import dask
from dask.distributed import Client, LocalCluster

from skimage.transform import (
    resize,
)

from qim3d.utils.logger import log
from qim3d.utils.progress_bar import OmeZarrExportProgressBar
from qim3d.utils.ome_zarr import get_n_chunks


ListOfArrayLike = Union[List[da.Array], List[np.ndarray]]
ArrayLike = Union[da.Array, np.ndarray]


class OMEScaler(
    scale.Scaler,
):
    """Scaler in the style of OME-Zarr.
    This is needed because their current zoom implementation is broken."""

    def __init__(self, order=0, downscale=2, max_layer=5, method="scaleZYXdask"):
        self.order = order
        self.downscale = downscale
        self.max_layer = max_layer
        self.method = method

    def scaleZYX(self, base):
        """Downsample using :func:`scipy.ndimage.zoom`."""
        rv = [base]
        log.info(f"- Scale 0: {rv[-1].shape}")

        for i in range(self.max_layer):
            rv.append(zoom(rv[-1], zoom=1 / self.downscale, order=self.order))
            log.info(f"- Scale {i+1}: {rv[-1].shape}")

        return list(rv)

    def scaleZYXdask(self, base):
        """
        Downsample a 3D volume using Dask and scipy.ndimage.zoom.

        This method performs multi-scale downsampling on a 3D dataset, generating image pyramids. It processes the data in chunks using Dask.

        Args:
            base (dask.array): The 3D array (volume) to be downsampled. Must be a Dask array for chunked processing.

        Returns:
            list of dask.array: A list of downsampled volumes, where each element represents a different scale. The first element corresponds to the original resolution, and subsequent elements represent progressively downsampled versions.

        The downsampling process occurs scale by scale, using the following steps:
        - For each scale, the array is resized based on the downscale factor, computed as a function of the current scale level.
        - The `scipy.ndimage.zoom` function is used to perform interpolation, with chunk-wise processing handled by Dask's `map_blocks` function.
        - The output is rechunked to match the input volume's original chunk size.


        """
        def resize_zoom(vol, scale_factors, order, scaled_shape):

            # Get the chunksize needed so that all the blocks match the new shape
            # This snippet comes from the original OME-Zarr-python library
            better_chunksize = tuple(
                np.maximum(
                    1, np.round(np.array(vol.chunksize) * scale_factors) / scale_factors
                ).astype(int)
            )

            log.debug(f"better chunk size: {better_chunksize}")

            # Compute the chunk size after the downscaling
            new_chunk_size = tuple(
                np.ceil(np.multiply(better_chunksize, scale_factors)).astype(int)
            )

            log.debug(
                f"orginal chunk size: {vol.chunksize}, chunk size after downscale: {new_chunk_size}"
            )

            def resize_chunk(chunk, scale_factors, order):

                #print(f"zoom factors: {scale_factors}")
                resized_chunk = zoom(
                    chunk,
                    zoom=scale_factors,
                    order=order,
                    mode="grid-constant",
                    grid_mode=True,
                )
                #print(f"resized chunk shape: {resized_chunk.shape}")

                return resized_chunk

            output_slices = tuple(slice(0, d) for d in scaled_shape)

            # Testing new shape
            predicted_shape = np.multiply(vol.shape, scale_factors)
            log.debug(f"predicted shape: {predicted_shape}")
            scaled_vol = da.map_blocks(
                resize_chunk,
                vol,
                scale_factors,
                order,
                chunks=new_chunk_size,
            )[output_slices]

            # Rechunk the output to match the input
            # This is needed because chunks were scaled down
            scaled_vol = scaled_vol.rechunk(vol.chunksize)
            return scaled_vol

        rv = [base]
        log.info(f"- Scale 0: {rv[-1].shape}")

        for i in range(self.max_layer):
            log.debug(f"\nScale {i+1}\n{'-'*32}")
            # Calculate the downscale factor for this scale
            downscale_factor = 1 / (self.downscale ** (i + 1))

            scaled_shape = tuple(
                np.ceil(np.multiply(base.shape, downscale_factor)).astype(int)
            )

            log.debug(f"target shape: {scaled_shape}")
            downscale_rate = tuple(np.divide(rv[-1].shape, scaled_shape).astype(float))
            log.debug(f"downscale rate: {downscale_rate}")
            scale_factors = tuple(np.divide(1, downscale_rate))
            log.debug(f"scale factors: {scale_factors}")

            log.debug("\nResizing volume chunk-wise")
            scaled_vol = resize_zoom(rv[-1], scale_factors, self.order, scaled_shape)
            rv.append(scaled_vol)

            log.info(f"- Scale {i+1}: {rv[-1].shape}")

        return list(rv)

    def scaleZYXdask_legacy(self, base):
        """Downsample using the original OME-Zarr python library"""

        rv = [base]
        log.info(f"- Scale 0: {rv[-1].shape}")

        for i in range(self.max_layer):

            scaled_shape = tuple(
                base.shape[j] // (self.downscale ** (i + 1)) for j in range(3)
            )

            scaled = dask_resize(base, scaled_shape, order=self.order)
            rv.append(scaled)

            log.info(f"- Scale {i+1}: {rv[-1].shape}")
        return list(rv)


def export_ome_zarr(
    path,
    data,
    chunk_size=256,
    downsample_rate=2,
    order=1,
    replace=False,
    method="scaleZYX",
    progress_bar: bool = True,
    progress_bar_repeat_time="auto",
):
    """
    Export 3D image data to OME-Zarr format with pyramidal downsampling.

    This function generates a multi-scale OME-Zarr representation of the input data, which is commonly used for large imaging datasets. The downsampled scales are calculated such that the smallest scale fits within the specified `chunk_size`.

    Args:
        path (str): The directory where the OME-Zarr data will be stored.
        data (np.ndarray or dask.array): The 3D image data to be exported. Supports both NumPy and Dask arrays.
        chunk_size (int, optional): The size of the chunks for storing data. This affects both the original data and the downsampled scales. Defaults to 256.
        downsample_rate (int, optional): The factor by which to downsample the data for each scale. Must be greater than 1. Defaults to 2.
        order (int, optional): The interpolation order to use when downsampling. Defaults to 1 (linear). Use 0 for a faster nearest-neighbor interpolation.
        replace (bool, optional): Whether to replace the existing directory if it already exists. Defaults to False.
        method (str, optional): The method used for downsampling. If set to "dask", Dask arrays are used for chunking and downsampling. Defaults to "scaleZYX".
        progress_bar (bool, optional): Whether to display a progress bar during export. Defaults to True.
        progress_bar_repeat_time (str or int, optional): The repeat interval (in seconds) for updating the progress bar. Defaults to "auto".

    Raises:
        ValueError: If the directory already exists and `replace` is False.
        ValueError: If `downsample_rate` is less than or equal to 1.

    Example:
        ```python
        import qim3d

        downloader = qim3d.io.Downloader()
        data = downloader.Snail.Escargot(load_file=True)

        qim3d.io.export_ome_zarr("Escargot.zarr", data, chunk_size=100, downsample_rate=2)
        ```

    Returns:
        None: This function writes the OME-Zarr data to the specified directory and does not return any value.
    """

    # Check if directory exists
    if os.path.exists(path):
        if replace:
            shutil.rmtree(path)
        else:
            raise ValueError(
                f"Directory {path} already exists. Use replace=True to overwrite."
            )

    # Check if downsample_rate is valid
    if downsample_rate <= 1:
        raise ValueError("Downsample rate must be greater than 1.")

    log.info(f"Exporting data to OME-Zarr format at {path}")

    # Get the number of scales
    min_dim = np.max(np.shape(data))
    nscales = math.ceil(math.log(min_dim / chunk_size) / math.log(downsample_rate))
    log.info(f"Number of scales: {nscales + 1}")

    # Create scaler
    scaler = OMEScaler(
        downscale=downsample_rate, max_layer=nscales, method=method, order=order
    )

    # write the image data
    os.mkdir(path)
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)

    # Check if we want to process using Dask
    if "dask" in method and not isinstance(data, da.Array):
        log.info("\nConverting input data to Dask array")
        data = da.from_array(data, chunks=(chunk_size, chunk_size, chunk_size))
        log.info(f" - shape...: {data.shape}\n - chunks..: {data.chunksize}\n")

    elif "dask" in method and isinstance(data, da.Array):
        log.info("\nInput data will be rechunked")
        data = data.rechunk((chunk_size, chunk_size, chunk_size))
        log.info(f" - shape...: {data.shape}\n - chunks..: {data.chunksize}\n")


    log.info("Calculating the multi-scale pyramid")

    # Generate multi-scale pyramid
    mip = scaler.func(data)

    log.info("Writing data to disk")
    kwargs = dict(
        pyramid=mip,
        group=root,
        fmt=CurrentFormat(),
        axes="zyx",
        name=None,
        compute=True,
        storage_options=dict(chunks=(chunk_size, chunk_size, chunk_size)),
    )
    if progress_bar:
        n_chunks = get_n_chunks(
            shapes=(scaled_data.shape for scaled_data in mip),
            dtypes=(scaled_data.dtype for scaled_data in mip),
        )
        with OmeZarrExportProgressBar(
            path=path, n_chunks=n_chunks, reapeat_time=progress_bar_repeat_time
        ):
            write_multiscale(**kwargs)
    else:
        write_multiscale(**kwargs)

    log.info("\nAll done!")

    return


def import_ome_zarr(path, scale=0, load=True):
    """
    Import image data from an OME-Zarr file.

    This function reads OME-Zarr formatted volumetric image data and returns the specified scale.
    The image data can be lazily loaded (as Dask arrays) or fully computed into memory.

    Args:
        path (str): The file path to the OME-Zarr data.
        scale (int or str, optional): The scale level to load.
            If 'highest', loads the finest scale (scale 0).
            If 'lowest', loads the coarsest scale (last available scale). Defaults to 0.
        load (bool, optional): Whether to compute the selected scale into memory.
            If False, returns a lazy Dask array. Defaults to True.

    Returns:
        np.ndarray or dask.array.Array: The requested image data, either as a NumPy array if `load=True`,
        or a Dask array if `load=False`.

    Raises:
        ValueError: If the requested `scale` does not exist in the data.

    Example:
        ```python
        import qim3d

        data = qim3d.io.import_ome_zarr("Escargot.zarr", scale=0, load=True)

        ```

    """

    # read the image data
    # store = parse_url(path, mode="r").store

    reader = Reader(parse_url(path))
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data

    log.info(f"Data contains {len(dask_data)} scales:")
    for i in np.arange(len(dask_data)):
        log.info(f"- Scale {i}: {dask_data[i].shape}")

    if scale == "highest":
        scale = 0

    if scale == "lowest":
        scale = len(dask_data) - 1

    if scale >= len(dask_data):
        raise ValueError(
            f"Scale {scale} does not exist in the data. Please choose a scale between 0 and {len(dask_data)-1}."
        )

    log.info(f"\nLoading scale {scale} with shape {dask_data[scale].shape}")

    if load:
        vol = dask_data[scale].compute()
    else:
        vol = dask_data[scale]

    return vol
