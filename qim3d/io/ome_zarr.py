"""
Exporting data to different formats.
"""

import os
import numpy as np
import zarr
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

import math
import shutil
from qim3d.utils.logger import log
from scipy.ndimage import zoom
from typing import Any, Callable, Iterator, List, Tuple, Union
import dask.array as da
from skimage.transform import (
    resize,
)


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
            downscale_ratio = (1 / self.downscale) ** (i + 1)
            rv.append(zoom(base, zoom=downscale_ratio, order=self.order))
            log.info(f"- Scale {i+1}: {rv[-1].shape}")
        return list(rv)

    def scaleZYXdask(self, base):
        """Downsample using :func:`scipy.ndimage.zoom`."""
        rv = [base]
        log.info(f"- Scale 0: {rv[-1].shape}")

        for i in range(self.max_layer):

            scaled_shape = tuple(
                base.shape[j] // (self.downscale ** (i + 1)) for j in range(3)
            )
            rv.append(dask_resize(base, scaled_shape, order=self.order))

            log.info(f"- Scale {i+1}: {rv[-1].shape}")
        return list(rv)


def export_ome_zarr(
    path,
    data,
    chunk_size=100,
    downsample_rate=2,
    order=0,
    replace=False,
    method="scaleZYX",
):
    """
    Export image data to OME-Zarr format with pyramidal downsampling.

    Automatically calculates the number of downsampled scales such that the smallest scale fits within the specified `chunk_size`.

    Args:
        path (str): The directory where the OME-Zarr data will be stored.
        data (np.ndarray): The image data to be exported.
        chunk_size (int, optional): The size of the chunks for storing data. Defaults to 100.
        downsample_rate (int, optional): Factor by which to downsample the data for each scale. Must be greater than 1. Defaults to 2.
        order (int, optional): Interpolation order to use when downsampling. Defaults to 0 (nearest-neighbor).
        replace (bool, optional): Whether to replace the existing directory if it already exists. Defaults to False.

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

    fmt = CurrentFormat()
    log.info("Creating a multi-scale pyramid")
    mip, axes = _create_mip(image=data, fmt=fmt, scaler=scaler, axes="zyx")

    log.info("Writing data to disk")
    write_multiscale(
        mip,
        group=root,
        fmt=fmt,
        axes=axes,
        name=None,
        compute=True,
    )

    log.info("All done!")
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
