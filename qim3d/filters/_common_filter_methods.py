"""Provides filter functions and classes for image processing"""

from typing import Type

import dask.array as da
import dask_image.ndfilters as dask_ndfilters
import numpy as np
from scipy import ndimage
from skimage import morphology

from qim3d.utils import log

class FilterBase:
    def __init__(self, *args, dask: bool = False, chunks: str = 'auto', **kwargs):
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
    def __init__(self, sigma: float, *args, **kwargs):
        """
        Gaussian filter initialization.

        Args:
            sigma (float): Standard deviation for Gaussian kernel.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Applies a Gaussian filter to the input.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume.

        """
        return gaussian(
            input,
            sigma=self.sigma,
            dask=self.dask,
            chunks=self.chunks,
            *self.args,
            **self.kwargs,
        )


class Median(FilterBase):
    def __init__(
        self, size: float = None, footprint: np.ndarray = None, *args, **kwargs
    ):
        """
        Median filter initialization.

        Args:
            size (float or tuple, optional): Filter size.
            footprint (np.ndarray, optional): The structuring element for filtering.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        """
        if size is None and footprint is None:
            raise ValueError("Either 'size' or 'footprint' must be provided.")
        super().__init__(*args, **kwargs)
        self.size = size
        self.footprint = footprint

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Applies a median filter to the input.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume.

        """
        return median(
            vol=input,
            size=self.size,
            footprint=self.footprint,
            dask=self.dask,
            chunks=self.chunks,
            **self.kwargs,
        )


class Maximum(FilterBase):
    def __init__(
        self, size: float = None, footprint: np.ndarray = None, *args, **kwargs
    ):
        """
        Maximum filter initialization.

        Args:
            size (float or tuple, optional): Filter size.
            footprint (np.ndarray, optional): The structuring element for filtering.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        """
        if size is None and footprint is None:
            raise ValueError("Either 'size' or 'footprint' must be provided.")
        super().__init__(*args, **kwargs)
        self.size = size
        self.footprint = footprint

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Applies a maximum filter to the input.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume.

        """
        return maximum(
            vol=input,
            size=self.size,
            footprint=self.footprint,
            dask=self.dask,
            chunks=self.chunks,
            **self.kwargs,
        )


class Minimum(FilterBase):
    def __init__(
        self, size: float = None, footprint: np.ndarray = None, *args, **kwargs
    ):
        """
        Minimum filter initialization.

        Args:
            size (float or tuple, optional): Filter size.
            footprint (np.ndarray, optional): The structuring element for filtering.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        """
        if size is None and footprint is None:
            raise ValueError("Either 'size' or 'footprint' must be provided.")
        super().__init__(*args, **kwargs)
        self.size = size
        self.footprint = footprint

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Applies a minimum filter to the input.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume.

        """
        return minimum(
            vol=input,
            size=self.size,
            footprint=self.footprint,
            dask=self.dask,
            chunks=self.chunks,
            **self.kwargs,
        )


class Tophat(FilterBase):
    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Applies a tophat filter to the input.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume.

        """
        return tophat(input, dask=self.dask, **self.kwargs)


class Pipeline:

    """
    Example:
        ```python
        import qim3d
        from qim3d.filters import Pipeline, Median, Gaussian, Maximum, Minimum

        # Get data
        vol = qim3d.examples.fly_150x256x256

        # Show original
        fig1 = qim3d.viz.slices_grid(vol, num_slices=5, display_figure=True)

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
        fig2 = qim3d.viz.slices_grid(vol_filtered, num_slices=5, display_figure=True)
        ```
        ![original volume](../../assets/screenshots/filter_original.png)
        ![filtered volume](../../assets/screenshots/filter_processed.png)

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
                f'filters should be instances of one of the following classes: {filter_names}'
            )
        self.filters[name] = fn

    def append(self, fn: FilterBase):
        """
        Appends a filter to the end of the sequence.

        Args:
            fn (FilterBase): An instance of a FilterBase subclass to be appended.

        Example:
            ```python
            import qim3d
            from qim3d.filters import Pipeline, Maximum, Median

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


def gaussian(
    vol: np.ndarray, sigma: float, dask: bool = False, chunks: str = 'auto', **kwargs
) -> np.ndarray:
    """
    Applies a Gaussian filter to the input volume using `scipy.ndimage.gaussian_filter` or `dask_image.ndfilters.gaussian_filter`.

    Args:
        vol (np.ndarray): The input image or volume.
        sigma (float or sequence of floats): The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.
        dask (bool, optional): Whether to use Dask for the Gaussian filter.
        chunks (int or tuple or "'auto'", optional): Defines how to divide the array into blocks when using Dask. Can be an integer, tuple, size in bytes, or "auto" for automatic sizing.
        **kwargs (Any): Additional keyword arguments for the Gaussian filter.

    Returns:
        filtered_vol (np.ndarray): The filtered image or volume.
    
    Example:
        ```python
        import qim3d

        # Apply filter
        vol = qim3d.examples.shell_225x128x128
        vol_filtered = qim3d.filters.gaussian(vol, sigma=3, dask=True)

        # Show original and filtered volumes
        qim3d.viz.slices_grid(vol, num_slices=5, display_figure=True)
        qim3d.viz.slices_grid(vol_filtered, num_slices=5, display_figure=True)
        ```
        ![gaussian-filter-before](../../assets/screenshots/gaussian_filter_original.png)
        ![gaussian-filter-after](../../assets/screenshots/gaussian_filter_processed.png)

    """

    if dask:
        if not isinstance(vol, da.Array):
            vol = da.from_array(vol, chunks=chunks)
        dask_vol = dask_ndfilters.gaussian_filter(vol, sigma, **kwargs)
        res = dask_vol.compute()
        return res
    else:
        res = ndimage.gaussian_filter(vol, sigma, **kwargs)
        return res


def median(
    vol: np.ndarray,
    size: float = None,
    footprint: np.ndarray = None,
    dask: bool = False,
    chunks: str = 'auto',
    **kwargs,
) -> np.ndarray:
    """
    Applies a median filter to the input volume using `scipy.ndimage.median_filter` or `dask_image.ndfilters.median_filter`.

    Args:
        vol (np.ndarray): The input image or volume.
        size (scalar or tuple, optional): Either size or footprint must be defined. size gives the shape that is taken from the input array, at every element position, to define the input to the filter function.
        footprint (np.ndarray, optional): Boolean array that specifies (implicitly) a shape, but also which of the elements within this shape will get passed to the filter function.
        dask (bool, optional): Whether to use Dask for the median filter.
        chunks (int or tuple or "'auto'", optional): Defines how to divide the array into blocks when using Dask. Can be an integer, tuple, size in bytes, or "auto" for automatic sizing.
        **kwargs (Any): Additional keyword arguments for the median filter.

    Returns:
        filtered_vol (np.ndarray): The filtered image or volume.

    Raises:
        RuntimeError: If neither size nor footprint is defined

    Example:
        ```python
        import qim3d

        # Generate a noisy volume
        vol = qim3d.generate.volume(noise_scale = 0.015)
        noisy_vol = qim3d.generate.background(background_shape = vol.shape, max_noise_value = 80, apply_method = 'add', apply_to = vol)

        # Apply filter
        vol_filtered = qim3d.filters.median(noisy_vol, size=5, dask=True)

        # Show original and filtered volumes
        qim3d.viz.slices_grid(noisy_vol, num_slices=5, slice_positions = [10, 31, 63, 95, 120], display_figure=True)
        qim3d.viz.slices_grid(vol_filtered, num_slices=5, slice_positions = [10, 31, 63, 95, 120], display_figure=True)
        ```
        ![median-filter-before](../../assets/screenshots/median_filter_original.png)
        ![median-filter-after](../../assets/screenshots/median_filter_processed.png)
    """
    if size is None:
        if footprint is None:
            raise RuntimeError('no footprint or filter size provided')

    if dask:
        if not isinstance(vol, da.Array):
            vol = da.from_array(vol, chunks=chunks)
        dask_vol = dask_ndfilters.median_filter(vol, size, footprint, **kwargs)
        res = dask_vol.compute()
        return res
    else:
        res = ndimage.median_filter(vol, size, footprint, **kwargs)
        return res


def maximum(
    vol: np.ndarray,
    size: float = None,
    footprint: np.ndarray = None,
    dask: bool = False,
    chunks: str = 'auto',
    **kwargs,
) -> np.ndarray:
    """
    Applies a maximum filter to the input volume using `scipy.ndimage.maximum_filter` or `dask_image.ndfilters.maximum_filter`.

    Args:
        vol (np.ndarray): The input image or volume.
        size (scalar or tuple, optional): Either size or footprint must be defined. size gives the shape that is taken from the input array, at every element position, to define the input to the filter function.
        footprint (np.ndarray, optional): Boolean array that specifies (implicitly) a shape, but also which of the elements within this shape will get passed to the filter function.
        dask (bool, optional): Whether to use Dask for the maximum filter.
        chunks (int or tuple or "'auto'", optional): Defines how to divide the array into blocks when using Dask. Can be an integer, tuple, size in bytes, or "auto" for automatic sizing.
        **kwargs (Any): Additional keyword arguments for the maximum filter.

    Returns:
        filtered_vol (np.ndarray): The filtered image or volume.

    Raises:
        RuntimeError: If neither size nor footprint is defined

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.fly_150x256x256
        vol_filtered = qim3d.filters.maximum(vol, size=6, dask=True)

        # Show original and filtered volumes
        qim3d.viz.slices_grid(vol, num_slices=5, display_figure=True)
        qim3d.viz.slices_grid(vol_filtered, num_slices=5, display_figure=True)
        ```
        ![maximum-filter-before](../../assets/screenshots/maximum_filter_original.png)
        ![maximum-filter-after](../../assets/screenshots/maximum_filter_processed.png)

    """
    if size is None:
        if footprint is None:
            raise RuntimeError('no footprint or filter size provided')

    if dask:
        if not isinstance(vol, da.Array):
            vol = da.from_array(vol, chunks=chunks)
        dask_vol = dask_ndfilters.maximum_filter(vol, size, footprint, **kwargs)
        res = dask_vol.compute()
        return res
    else:
        res = ndimage.maximum_filter(vol, size, footprint, **kwargs)
        return res


def minimum(
    vol: np.ndarray,
    size: float = None,
    footprint: np.ndarray = None,
    dask: bool = False,
    chunks: str = 'auto',
    **kwargs,
) -> np.ndarray:
    """
    Applies a minimum filter to the input volume using `scipy.ndimage.minimum_filter` or `dask_image.ndfilters.minimum_filter`.

    Args:
        vol (np.ndarray): The input image or volume.
        size (scalar or tuple, optional): Either size or footprint must be defined. size gives the shape that is taken from the input array, at every element position, to define the input to the filter function.
        footprint (np.ndarray, optional): Boolean array that specifies (implicitly) a shape, but also which of the elements within this shape will get passed to the filter function.
        dask (bool, optional): Whether to use Dask for the minimum filter.
        chunks (int or tuple or "'auto'", optional): Defines how to divide the array into blocks when using Dask. Can be an integer, tuple, size in bytes, or "auto" for automatic sizing.
        **kwargs (Any): Additional keyword arguments for the minimum filter.

    Returns:
        filtered_vol (np.ndarray): The filtered image or volume.

    Raises:
        RuntimeError: If neither size nor footprint is defined

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.shell_225x128x128
        vol_filtered = qim3d.filters.minimum(vol, size=3, dask=True)

        qim3d.viz.slices_grid(vol, num_slices=5, slice_positions = [10, 31, 63, 95, 120], display_figure=True)
        qim3d.viz.slices_grid(vol_filtered, num_slices=5, slice_positions = [10, 31, 63, 95, 120], display_figure=True)
        ```
        ![minimum-filter-before](../../assets/screenshots/minimum_filter_original.png)
        ![minimum-filter-after](../../assets/screenshots/minimum_filter_processed.png)

    """
    if size is None:
        if footprint is None:
            raise RuntimeError('no footprint or filter size provided')

    if dask:
        if not isinstance(vol, da.Array):
            vol = da.from_array(vol, chunks=chunks)
        dask_vol = dask_ndfilters.minimum_filter(vol, size, footprint, **kwargs)
        res = dask_vol.compute()
        return res
    else:
        res = ndimage.minimum_filter(vol, size, footprint, **kwargs)
        return res


def tophat(vol: np.ndarray, dask: bool = False, **kwargs):
    """
    Remove background from the volume.

    Args:
        vol (np.ndarray): The volume to remove background from.
        dask (bool, optional): Whether to use Dask for the tophat filter (not supported, will default to SciPy).
        **kwargs (Any): Additional keyword arguments.
            `radius` (float): The radius of the structuring element (default: 3).
            `background` (str): Color of the background, 'dark' or 'bright' (default: 'dark'). If 'bright', volume will be inverted.

    Returns:
        filtered_vol (np.ndarray): The volume with background removed.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.cement_128x128x128
        vol_filtered = qim3d.filters.tophat(vol, radius = 7, background = 'bright')

        qim3d.viz.slices_grid(vol, num_slices=5, slice_positions = [10, 31, 63, 95, 120], display_figure=True)
        qim3d.viz.slices_grid(vol_filtered, num_slices=5, slice_positions = [10, 31, 63, 95, 120], display_figure=True)
        ```
        ![tophat-filter-before](../../assets/screenshots/tophat_filter_original.png)
        ![tophat-filter-after](../../assets/screenshots/tophat_filter_processed.png)	

    """

    radius = kwargs['radius'] if 'radius' in kwargs else 3
    background = kwargs['background'] if 'background' in kwargs else 'dark'

    if dask:
        log.info('Dask not supported for tophat filter, switching to scipy.')

    if background == 'bright':
        log.info(
            'Bright background selected, volume will be temporarily inverted when applying white_tophat'
        )
        vol = np.invert(vol)

    selem = morphology.ball(radius)
    vol = vol - morphology.white_tophat(vol, selem)

    if background == 'bright':
        vol = np.invert(vol)

    return vol
