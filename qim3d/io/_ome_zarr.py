"""
Exporting data to different formats.
"""

import math
import os
import shutil
from typing import List, Union

import dask.array as da
import numpy as np
import zarr
from ome_zarr import scale
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from ome_zarr.scale import dask_resize
from ome_zarr.writer import (
    CurrentFormat,
    write_multiscale,
)
from scipy.ndimage import zoom

from qim3d.utils import log
from qim3d.utils._progress_bar import OmeZarrExportProgressBar

ListOfArrayLike = Union[List[da.Array], List[np.ndarray]]
ArrayLike = Union[da.Array, np.ndarray]


class OMEScaler(
    scale.Scaler,
):

    """
    Scaler in the style of OME-Zarr.
    This is needed because their current zoom implementation is broken.
    """

    def __init__(
        self,
        order: int = 0,
        downscale: float = 2,
        max_layer: int = 5,
        method: str = 'scaleZYXdask',
    ):
        self.order = order
        self.downscale = downscale
        self.max_layer = max_layer
        self.method = method

    def scaleZYX(self, base: da.Array):
        """Downsample using :func:`scipy.ndimage.zoom`."""
        rv = [base]
        log.info(f'- Scale 0: {rv[-1].shape}')

        for i in range(self.max_layer):
            rv.append(zoom(rv[-1], zoom=1 / self.downscale, order=self.order))
            log.info(f'- Scale {i+1}: {rv[-1].shape}')

        return list(rv)

    def scaleZYXdask(self, base: da.Array):
        """
        Downsample a 3D volume using Dask and scipy.ndimage.zoom.

        This method performs multi-scale downsampling on a 3D dataset, generating image pyramids. It processes the data in chunks using Dask.

        Args:
            base (dask.array.core.array): The 3D array (volume) to be downsampled. Must be a Dask array for chunked processing.

        Returns:
            list of dask.array.core.Array: A list of downsampled volumes, where each element represents a different scale. The first element corresponds to the original resolution, and subsequent elements represent progressively downsampled versions.

        The downsampling process occurs scale by scale, using the following steps:
        - For each scale, the array is resized based on the downscale factor, computed as a function of the current scale level.
        - The `scipy.ndimage.zoom` function is used to perform interpolation, with chunk-wise processing handled by Dask's `map_blocks` function.
        - The output is rechunked to match the input volume's original chunk size.


        """

        def resize_zoom(vol: da.Array, scale_factors, order, scaled_shape):
            # Get the chunksize needed so that all the blocks match the new shape
            # This snippet comes from the original OME-Zarr-python library
            better_chunksize = tuple(
                np.maximum(
                    1, np.round(np.array(vol.chunksize) * scale_factors) / scale_factors
                ).astype(int)
            )

            log.debug(f'better chunk size: {better_chunksize}')

            # Compute the chunk size after the downscaling
            new_chunk_size = tuple(
                np.ceil(np.multiply(better_chunksize, scale_factors)).astype(int)
            )

            log.debug(
                f'orginal chunk size: {vol.chunksize}, chunk size after downscale: {new_chunk_size}'
            )

            def resize_chunk(chunk, scale_factors, order):
                # print(f"zoom factors: {scale_factors}")
                resized_chunk = zoom(
                    chunk,
                    zoom=scale_factors,
                    order=order,
                    mode='grid-constant',
                    grid_mode=True,
                )
                # print(f"resized chunk shape: {resized_chunk.shape}")

                return resized_chunk

            output_slices = tuple(slice(0, d) for d in scaled_shape)

            # Testing new shape
            predicted_shape = np.multiply(vol.shape, scale_factors)
            log.debug(f'predicted shape: {predicted_shape}')
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
        log.info(f'- Scale 0: {rv[-1].shape}')

        for i in range(self.max_layer):
            log.debug(f"\nScale {i+1}\n{'-'*32}")
            # Calculate the downscale factor for this scale
            downscale_factor = 1 / (self.downscale ** (i + 1))

            scaled_shape = tuple(
                np.ceil(np.multiply(base.shape, downscale_factor)).astype(int)
            )

            log.debug(f'target shape: {scaled_shape}')
            downscale_rate = tuple(np.divide(rv[-1].shape, scaled_shape).astype(float))
            log.debug(f'downscale rate: {downscale_rate}')
            scale_factors = tuple(np.divide(1, downscale_rate))
            log.debug(f'scale factors: {scale_factors}')

            log.debug('\nResizing volume chunk-wise')
            scaled_vol = resize_zoom(rv[-1], scale_factors, self.order, scaled_shape)
            rv.append(scaled_vol)

            log.info(f'- Scale {i+1}: {rv[-1].shape}')

        return list(rv)

    def scaleZYXdask_coarsen(self, base:da.core.Array):
        """
        Export 3D image data to OME-Zarr format using dask.coarsen
        """
        rv = [base]
        log.info(f'- Scale 0: {rv[-1].shape}')

        for i in range(self.max_layer):
            log.debug(f"\nScale {i+1}\n{'-'*32}")

            scaled = da.coarsen(np.mean, rv[-1], {0:2, 1:2, 2:2}, trim_excess=True)
            rv.append(scaled)
            log.info(f'- Scale {i+1}: {rv[-1].shape}')

        return list(rv)


    def scaleZYXdask_legacy(self, base):
        """Downsample using the original OME-Zarr python library"""

        rv = [base]
        log.info(f'- Scale 0: {rv[-1].shape}')

        for i in range(self.max_layer):
            scaled_shape = tuple(
                base.shape[j] // (self.downscale ** (i + 1)) for j in range(3)
            )

            scaled = dask_resize(base, scaled_shape, order=self.order)
            rv.append(scaled)

            log.info(f'- Scale {i+1}: {rv[-1].shape}')
        return list(rv)


def export_ome_zarr(
    path: str | os.PathLike,
    data: np.ndarray | da.core.Array,
    chunk_size: int = 256,
    downsample_rate: int = 2,
    order: int = 1,
    replace: bool = False,
    method: str = 'scaleZYXdask_coarsen',
    progress_bar: bool = True,
    progress_bar_repeat_time: str = 'auto',
) -> None:
    """
    Exports 3D data to the OME-Zarr (NGFF) format with multi-scale pyramidal levels.

    Generates a **Next Generation File Format (NGFF)** representation of the input volume.
    This format creates a multi-resolution pyramid (downsampled copies), allowing for efficient
    visualization and streaming of large datasets over networks or the cloud.

    **Key Features:**

    * **Chunking:** Data is divided into small blocks (`chunk_size`) for efficient random access.
    * **Pyramidal Levels:** Automatically calculates and generates lower-resolution levels
      until the dataset fits within a single chunk.
    * **Dask Integration:** efficiently handles larger-than-memory datasets by processing chunks in parallel.

    Args:
        path (str or os.PathLike):
            The destination directory path. The directory will be created as a Zarr group. (E.g. `"data.zarr"`).
        data (numpy.ndarray or dask.array.Array):
            The 3D image volume to export.
        chunk_size (int, optional):
            The size of the chunks (cubes) for storage (e.g., `256` creates 256x256x256 blocks).
            Smaller chunks improve access time for specific regions but increase file count.
        downsample_rate (int, optional):
            The reduction factor between pyramid levels. A rate of `2` means each level is
            half the size of the previous one.
        order (int, optional):
            The interpolation order for downsampling. `0` = Nearest Neighbor (faster) and `1` = Linear.
        replace (bool, optional):
            If `True`, deletes the existing directory at `path` before writing.
            If `False`, raises an error if the directory exists.
        method (str, optional):
            The downsampling strategy.
            `"scaleZYXdask_coarsen"` uses block averaging (faster, reduces noise).
            `"scaleZYXdask"` uses interpolation (slower, potentially sharper).
        progress_bar (bool, optional):
            If `True`, displays a progress bar tracking the chunk writing process.
        progress_bar_repeat_time (str or int, optional):
            Interval in seconds for updating the progress bar.

    Raises:
        ValueError: If `path` exists and `replace=False`.
        ValueError: If `downsample_rate` <= 1.

    Example:
        ```python
        import qim3d

        # Load a sample dataset
        downloader = qim3d.io.Downloader()
        data = downloader.Snail.Escargot(load_file=True)

        # Export to OME-Zarr with 2x downsampling per level
        qim3d.io.export_ome_zarr("Escargot.zarr", data, chunk_size=128, downsample_rate=2)
        ```
    """

    # Check if directory exists
    if os.path.exists(path):
        if replace:
            shutil.rmtree(path)
        else:
            raise ValueError(
                f'Directory {path} already exists. Use replace=True to overwrite.'
            )

    # Check if downsample_rate is valid
    if downsample_rate <= 1:
        raise ValueError('Downsample rate must be greater than 1.')

    log.info(f'Exporting data to OME-Zarr format at {path}')

    # Get the number of scales
    min_dim = np.max(np.shape(data))
    nscales = math.ceil(math.log(min_dim / chunk_size) / math.log(downsample_rate))
    log.info(f'Number of scales: {nscales + 1}')

    # Create scaler
    scaler = OMEScaler(
        downscale=downsample_rate, max_layer=nscales, method=method, order=order
    )

    # write the image data
    os.mkdir(path)
    store = parse_url(path, mode='w').store
    root = zarr.group(store=store)

    # Check if we want to process using Dask
    if 'dask' in method and not isinstance(data, da.Array):
        log.info('\nConverting input data to Dask array')
        data = da.from_array(data, chunks=(chunk_size, chunk_size, chunk_size))
        log.info(f' - shape...: {data.shape}\n - chunks..: {data.chunksize}\n')

    elif 'dask' in method and isinstance(data, da.Array):
        log.info('\nInput data will be rechunked')
        data = data.rechunk((chunk_size, chunk_size, chunk_size))
        log.info(f' - shape...: {data.shape}\n - chunks..: {data.chunksize}\n')

    log.info('Calculating the multi-scale pyramid')

    # Generate multi-scale pyramid
    mip = scaler.func(data)

    log.info('Writing data to disk')
    kwargs = dict(
        pyramid=mip,
        group=root,
        fmt=CurrentFormat(),
        axes='zyx',
        name=None,
        compute=True,
        storage_options=dict(chunks=(chunk_size, chunk_size, chunk_size)),
    )
    if progress_bar:

        # Get number of chunks for each shape and sum them together
        n_chunks = sum([np.prod(np.ceil(np.array(scaled_data.shape)/chunk_size)) for scaled_data in mip])

        with OmeZarrExportProgressBar(
            path=path, n_chunks=n_chunks, reapeat_time=progress_bar_repeat_time
        ):
            write_multiscale(**kwargs)
    else:
        write_multiscale(**kwargs)

    log.info('\nAll done!')

    return


def import_ome_zarr(
    path: str | os.PathLike, scale: int = 0, load: bool = True
) -> np.ndarray:
    """
    Imports or reads image data from an OME-Zarr (NGFF) container.

    Allows reading specific resolution levels from a multi-scale dataset. This is particularly
    useful for previewing large datasets by loading a coarse scale before fetching the full-resolution data.

    Args:
        path (str or os.PathLike):
            The path to the OME-Zarr file (directory).
        scale (int or str, optional):
            The resolution level to load.
            `0` is the full resolution (finest). Higher integers are progressively lower resolutions.
            Can also accept `'highest'` (alias for 0) or `'lowest'` (coarsest available scale).
        load (bool, optional):
            If `True`, reads the data into memory as a `numpy.ndarray`.
            If `False`, returns a `dask.array.Array` for lazy loading/processing.

    Returns:
        vol (numpy.ndarray or dask.array.Array):
            The requested image data.

            * **numpy.ndarray**: The full image data loaded into memory (if `load=True`).
            * **dask.array.Array**: A lazy-loaded Dask array connected to the Zarr store (if `load=False`).
    Raises:
        ValueError: If the requested `scale` index exceeds the available levels in the dataset.

    Example:
        ```python
        import qim3d

        # 1. Load the full resolution data into memory
        data = qim3d.io.import_ome_zarr("Escargot.zarr", scale=0, load=True)

        # 2. Lazy load the lowest resolution (thumbnail/preview)
        preview_lazy = qim3d.io.import_ome_zarr("Escargot.zarr", scale='lowest', load=False)
        print(preview_lazy.shape)
        ```
    """

    # read the image data
    # store = parse_url(path, mode="r").store

    reader = Reader(parse_url(path))
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data

    log.info(f'Data contains {len(dask_data)} scales:')
    for i in np.arange(len(dask_data)):
        log.info(f'- Scale {i}: {dask_data[i].shape}')

    if scale == 'highest':
        scale = 0

    if scale == 'lowest':
        scale = len(dask_data) - 1

    if scale >= len(dask_data):
        raise ValueError(
            f'Scale {scale} does not exist in the data. Please choose a scale between 0 and {len(dask_data)-1}.'
        )

    log.info(f'\nLoading scale {scale} with shape {dask_data[scale].shape}')

    if load:
        vol = dask_data[scale].compute()
    else:
        vol = dask_data[scale]

    return vol
