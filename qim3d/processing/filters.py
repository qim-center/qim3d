"""Provides filter functions and classes for image processing"""

from typing import Type, Union

import numpy as np
from scipy import ndimage
from skimage import morphology
import dask.array as da
import dask_image.ndfilters as dask_ndfilters

from qim3d.utils.logger import log

__all__ = [
    "Gaussian",
    "Median",
    "Maximum",
    "Minimum",
    "Pipeline",
    "Tophat",
    "gaussian",
    "median",
    "maximum",
    "minimum",
    "tophat",
]


class FilterBase:
    def __init__(self, dask=False, chunks="auto", *args, **kwargs):
        """
        Base class for image filters.

        Args:
            *args: Additional positional arguments for filter initialization.
            **kwargs: Additional keyword arguments for filter initialization.
        """
        self.args = args
        self.dask = dask
        self.chunks = chunks
        self.kwargs = kwargs

class Gaussian(FilterBase):
    def __call__(self, input):
        """
        Applies a Gaussian filter to the input.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume.
        """
        return gaussian(input, dask=self.dask, chunks=self.chunks, *self.args, **self.kwargs)


class Median(FilterBase):
    def __call__(self, input):
        """
        Applies a median filter to the input.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume.
        """
        return median(input, dask=self.dask, chunks=self.chunks, **self.kwargs)


class Maximum(FilterBase):
    def __call__(self, input):
        """
        Applies a maximum filter to the input.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume.
        """
        return maximum(input, dask=self.dask, chunks=self.chunks, **self.kwargs)


class Minimum(FilterBase):
    def __call__(self, input):
        """
        Applies a minimum filter to the input.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume.
        """
        return minimum(input, dask=self.dask, chunks=self.chunks, **self.kwargs)

class Tophat(FilterBase):
    def __call__(self, input):
        """
        Applies a tophat filter to the input.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume.
        """
        return tophat(input, dask=self.dask, chunks=self.chunks, **self.kwargs)


class Pipeline:
    """
    Example:
        ```python
        import qim3d
        from qim3d.processing import Pipeline, Median, Gaussian, Maximum, Minimum

        # Get data
        vol = qim3d.examples.fly_150x256x256

        # Show original
        qim3d.viz.slices(vol, axis=0, show=True)

        # Create filter pipeline
        pipeline = Pipeline(
            Median(size=5),
            Gaussian(sigma=3, dask = True)
        )

        # Append a third filter to the pipeline
        pipeline.append(Maximum(size=3))

        # Apply filter pipeline
        vol_filtered = pipeline(vol)

        # Show filtered
        qim3d.viz.slices(vol_filtered, axis=0)
        ```
        ![original volume](assets/screenshots/filter_original.png)
        ![filtered volume](assets/screenshots/filter_processed.png)
            
        """
    def __init__(self, *args: Type[FilterBase]):
        """
        Represents a sequence of image filters.

        Args:
            *args: Variable number of filter instances to be applied sequentially.
        """
        self.filters = {}

        for idx, fn in enumerate(args):
            self._add_filter(str(idx), fn)

    def _add_filter(self, name: str, fn: Type[FilterBase]):
        """
        Adds a filter to the sequence.

        Args:
            name: A string representing the name or identifier of the filter.
            fn: An instance of a FilterBase subclass.

        Raises:
            AssertionError: If `fn` is not an instance of the FilterBase class.
        """
        if not isinstance(fn, FilterBase):
            filter_names = [
                subclass.__name__ for subclass in FilterBase.__subclasses__()
            ]
            raise AssertionError(
                f"filters should be instances of one of the following classes: {filter_names}"
            )
        self.filters[name] = fn

    def append(self, fn: Type[FilterBase]):
        """
        Appends a filter to the end of the sequence.

        Args:
            fn: An instance of a FilterBase subclass to be appended.
        
        Example:
            ```python
            import qim3d
            from qim3d.processing import Pipeline, Maximum, Median

            # Create filter pipeline
            pipeline = Pipeline(
                Maximum(size=3, dask=True),
            )

            # Append a second filter to the pipeline
            pipeline.append(Median(size=5))
            ```
        """
        self._add_filter(str(len(self.filters)), fn)

    def __call__(self, input):
        """
        Applies the sequential filters to the input in order.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume after applying all sequential filters.
        """
        for fn in self.filters.values():
            input = fn(input)
        return input


def gaussian(vol, dask=False, chunks='auto', *args, **kwargs):
    """
    Applies a Gaussian filter to the input volume using scipy.ndimage.gaussian_filter or dask_image.ndfilters.gaussian_filter.

    Args:
        vol: The input image or volume.
        dask: Whether to use Dask for the Gaussian filter.
        chunks: Defines how to divide the array into blocks when using Dask. Can be an integer, tuple, size in bytes, or "auto" for automatic sizing.
        *args: Additional positional arguments for the Gaussian filter.
        **kwargs: Additional keyword arguments for the Gaussian filter.

    Returns:
        The filtered image or volume.
    """
    
    if dask:
        if not isinstance(vol, da.Array):
            vol = da.from_array(vol, chunks=chunks)
        dask_vol = dask_ndfilters.gaussian_filter(vol, *args, **kwargs)
        res = dask_vol.compute()
        return res
    else:
        res = ndimage.gaussian_filter(vol, *args, **kwargs)
        return res


def median(vol, dask=False, chunks='auto', **kwargs):
    """
    Applies a median filter to the input volume using scipy.ndimage.median_filter or dask_image.ndfilters.median_filter.

    Args:
        vol: The input image or volume.
        dask: Whether to use Dask for the median filter.
        chunks: Defines how to divide the array into blocks when using Dask. Can be an integer, tuple, size in bytes, or "auto" for automatic sizing.
        **kwargs: Additional keyword arguments for the median filter.

    Returns:
        The filtered image or volume.
    """
    if dask:
        if not isinstance(vol, da.Array):
            vol = da.from_array(vol, chunks=chunks)
        dask_vol = dask_ndfilters.median_filter(vol, **kwargs)
        res = dask_vol.compute()
        return res
    else:
        res = ndimage.median_filter(vol, **kwargs)
        return res


def maximum(vol, dask=False, chunks='auto', **kwargs):
    """
    Applies a maximum filter to the input volume using scipy.ndimage.maximum_filter or dask_image.ndfilters.maximum_filter.

    Args:
        vol: The input image or volume.
        dask: Whether to use Dask for the maximum filter.
        chunks: Defines how to divide the array into blocks when using Dask. Can be an integer, tuple, size in bytes, or "auto" for automatic sizing.
        **kwargs: Additional keyword arguments for the maximum filter.

    Returns:
        The filtered image or volume.
    """
    if dask:
        if not isinstance(vol, da.Array):
            vol = da.from_array(vol, chunks=chunks)
        dask_vol = dask_ndfilters.maximum_filter(vol, **kwargs)
        res = dask_vol.compute()
        return res
    else:
        res = ndimage.maximum_filter(vol, **kwargs)
        return res


def minimum(vol, dask=False, chunks='auto', **kwargs):
    """
    Applies a minimum filter to the input volume using scipy.ndimage.minimum_filter or dask_image.ndfilters.minimum_filter.

    Args:
        vol: The input image or volume.
        dask: Whether to use Dask for the minimum filter.
        chunks: Defines how to divide the array into blocks when using Dask. Can be an integer, tuple, size in bytes, or "auto" for automatic sizing.
        **kwargs: Additional keyword arguments for the minimum filter.

    Returns:
        The filtered image or volume.
    """
    if dask:
        if not isinstance(vol, da.Array):
            vol = da.from_array(vol, chunks=chunks)
        dask_vol = dask_ndfilters.minimum_filter(vol, **kwargs)
        res = dask_vol.compute()
        return res
    else:
        res = ndimage.minimum_filter(vol, **kwargs)
        return res

def tophat(vol, dask=False, chunks='auto', **kwargs):
    """
    Remove background from the volume.

    Args:
        vol: The volume to remove background from.
        radius: The radius of the structuring element (default: 3).
        background: Color of the background, 'dark' or 'bright' (default: 'dark'). If 'bright', volume will be inverted.
        dask: Whether to use Dask for the tophat filter (not supported, will default to SciPy).
        chunks: Defines how to divide the array into blocks when using Dask. Can be an integer, tuple, size in bytes, or "auto" for automatic sizing.
        **kwargs: Additional keyword arguments.

    Returns:
        vol: The volume with background removed.
    """

    radius = kwargs["radius"] if "radius" in kwargs else 3
    background = kwargs["background"] if "background" in kwargs else "dark"

    if dask:
        log.info("Dask not supported for tophat filter, switching to scipy.")
    
    if background == "bright":
        log.info("Bright background selected, volume will be temporarily inverted when applying white_tophat")
        vol = np.invert(vol)
    
    selem = morphology.ball(radius)
    vol = vol - morphology.white_tophat(vol, selem)

    if background == "bright":
        vol = np.invert(vol)
    
    return vol