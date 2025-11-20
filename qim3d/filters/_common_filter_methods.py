"""Provides filter functions and classes for image processing"""

from typing import Type, Callable
import abc
import inspect

import dask.array as da
import dask_image.ndfilters as dask_ndfilters
import numpy as np
from scipy import ndimage
from skimage import morphology

from qim3d.utils import log

class FilterBase:
    def __init__(self, *args, dask: bool = False, chunks: str = 'auto',save_output:bool = False, **kwargs):
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
        self.save_output = save_output

    @abc.abstractmethod
    def __call__(self, input:np.ndarray) -> np.ndarray:
        pass

class Filter(FilterBase):
    def __init__(self, func:Callable, *args, **kwargs):
        self.func = func
        super().__init__(*args, **kwargs)

    def __call__(self, input):
        return self.func(input, *self.args, **self.kwargs)

class Threshold(FilterBase):
    def __init__(self, threshold:int|float, *args, **kwargs):
        self.threshold = threshold
        super().__init__(*args, **kwargs)

    def __call__(self, input:np.ndarray):
        return input > self.threshold

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

class Sobel(FilterBase):
    def __init__(self, *args, **kwargs):
        """
        Sobel filter initialization.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(*args, **kwargs)

    def __call__(self, input:np.ndarray) -> np.ndarray:
        """
        Applies a Sobel filter to the input.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume.

        """
        return sobel(input, self.dask)

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
    
class Normalize(FilterBase):
    def __call__(self, input:np.ndarray) -> np.ndarray:
        return normalize(input)


class Pipeline:
    """
    Creates an easy way to apply a lot of filters in order. Allows any callable
    that takes in one argument and returns one variable. Works with ```lambda``` or you can use ```Filter```
    class for better readability.
    Use keywaord ```save_output``` with any filter to save results in the middle of pipeline and use them later.
    
    Example:
        ```python
        import qim3d
        from qim3d.filters import Pipeline, Filter, Sobel, Threshold 
        from scipy import ndimage
        
        # Get data
        vol = qim3d.examples.bone_128x128x128.astype('int64')

        # Show original
        qim3d.viz.slices_grid(vol, num_slices=5)

        # Create filter pipeline
        pipeline = Pipeline(
            Sobel(save_output = True),
            Threshold(600),
        )

        # Add another filter to the pipeline
        pipeline.append(Filter(ndimage.binary_opening))

        # Apply the filter pipeline
        filtered_vol = pipeline(vol)

        # Show middle step
        qim3d.viz.slices_grid(pipeline.saved_outputs[0], num_slices=5)

        # Show filtered result
        qim3d.viz.slices_grid(filtered_vol, num_slices=5)
        ```

        ![original volume](../../assets/screenshots/pipeline_original.png)
        ![original volume](../../assets/screenshots/pipeline_middlestep.png)
        ![filtered volume](../../assets/screenshots/pipeline_processed.png)

    """

    def __init__(self, *args: Type[FilterBase]):
        """
        Represents a sequence of image filters.

        Args:
            *args: Variable number of filter instances to be applied sequentially.

        """
        self.filters = []
        self.saved_outputs = []

        for fn in args:
            self._add_filter(fn)

    def _add_filter(self, fn: Type[FilterBase]|Callable):
        """
        Adds a filter to the sequence. 

        Args:
            name: A string representing the name or identifier of the filter.
            fn: An instance of a FilterBase subclass.

        Raises:
            AssertionError: If `fn` is not an instance of the FilterBase class.

        """
        if not isinstance(fn, FilterBase):
            if not callable(fn):
                raise TypeError(f'Pipeline only accepts callable objects. Your object is of type "{type(fn)}".')
            signature = inspect.signature(fn)
            if signature.parameters != 1:
                raise TypeError(f'Pipeline only accepts callables that take one argument. Yours takes {signature.parameters}.')
        self.filters.append(fn)

    def append(self, fn: FilterBase|Callable):
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
        self._add_filter(fn)

    def __call__(self, input:np.ndarray):
        """
        Applies the sequential filters to the input in order.

        Args:
            input: The input image or volume.

        Returns:
            The filtered image or volume after applying all sequential filters.

        """
        self.saved_outputs = []
        for fn in self.filters:
            input = fn(input)
            if hasattr(fn, 'save_output') and fn.save_output:
                self.saved_outputs.append(input)
        return input

def normalize(vol:np.ndarray) -> np.ndarray:
    return 255 * int((vol - vol.min())/vol.max())

def gaussian(
    volume: np.ndarray, sigma: float, dask: bool = False, chunks: str = 'auto', **kwargs
) -> np.ndarray:
    """
    Applies a Gaussian filter to the input volume using `scipy.ndimage.gaussian_filter` or `dask_image.ndfilters.gaussian_filter`.

    Args:
        volume (np.ndarray): The input image or volume.
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
        qim3d.viz.slices_grid(vol, n_slices=5, display_figure=True)
        qim3d.viz.slices_grid(vol_filtered, n_slices=5, display_figure=True)
        ```
        ![gaussian-filter-before](../../assets/screenshots/gaussian_filter_original.png)
        ![gaussian-filter-after](../../assets/screenshots/gaussian_filter_processed.png)

    """

    if dask:
        if not isinstance(volume, da.Array):
            volume = da.from_array(volume, chunks=chunks)
        dask_volume = dask_ndfilters.gaussian_filter(volume, sigma, **kwargs)
        res = dask_volume.compute()
        return res
    else:
        res = ndimage.gaussian_filter(volume, sigma, **kwargs)
        return res
    
def sobel(vol:np.ndarray, dask:bool = False):
    """
    Applies scipy.ndimage.sobel filter along all three axes to find egdes.
    If the output looks like noise, the integers have overflown. Try changing 
    the dtype of your volume.
    
    Args:
        vol (np.ndarray): The input image or volume
        dask (bool, optional): Whether to use Dask for the median filter.

    Returns:
        filtered_vol (np.ndarray): The filtered image or volume

    Example:
        ```python
        import qim3d
        vol = qim3d.examples.bone_128x128x128.astype('int64')
        filtered_vol = qim3d.filters.sobel(vol)
        qim3d.viz.slices_grid(vol, num_slices=5)
        qim3d.viz.slices_grid(filtered_vol, num_slices=5)
        ```
        ![sobel-filter-before](../../assets/screenshots/sobel_filter_original.png)
        ![sobel-filter-after](../../assets/screenshots/pipeline_middlestep.png)
    """
    if dask:
        if not isinstance(vol, da.Array):
            vol = da.from_array()
    sob0 = ndimage.sobel(vol,0)
    sob1 = ndimage.sobel(vol,1)
    if vol.ndim == 3:
        sob2 = ndimage.sobel(vol,2)
        return np.sqrt(sob0**2 + sob1**2 + sob2**2)
    else:
        return np.sqrt(sob0**2 + sob1**2)



def median(
    volume: np.ndarray,
    size: float = None,
    footprint: np.ndarray = None,
    dask: bool = False,
    chunks: str = 'auto',
    **kwargs,
) -> np.ndarray:
    """
    Applies a median filter to the input volume using `scipy.ndimage.median_filter` or `dask_image.ndfilters.median_filter`.

    Args:
        volume (np.ndarray): The input image or volume.
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
        qim3d.viz.slices_grid(noisy_vol, n_slices=5, slice_positions = [10, 31, 63, 95, 120], display_figure=True)
        qim3d.viz.slices_grid(vol_filtered, n_slices=5, slice_positions = [10, 31, 63, 95, 120], display_figure=True)
        ```
        ![median-filter-before](../../assets/screenshots/median_filter_original.png)
        ![median-filter-after](../../assets/screenshots/median_filter_processed.png)
    """
    if size is None:
        if footprint is None:
            raise RuntimeError('no footprint or filter size provided')

    if dask:
        if not isinstance(volume, da.Array):
            volume = da.from_array(volume, chunks=chunks)
        dask_volume = dask_ndfilters.median_filter(volume, size, footprint, **kwargs)
        res = dask_volume.compute()
        return res
    else:
        res = ndimage.median_filter(volume, size, footprint, **kwargs)
        return res


def maximum(
    volume: np.ndarray,
    size: float = None,
    footprint: np.ndarray = None,
    dask: bool = False,
    chunks: str = 'auto',
    **kwargs,
) -> np.ndarray:
    """
    Applies a maximum filter to the input volume using `scipy.ndimage.maximum_filter` or `dask_image.ndfilters.maximum_filter`.

    Args:
        volume (np.ndarray): The input image or volume.
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
        qim3d.viz.slices_grid(vol, n_slices=5, display_figure=True)
        qim3d.viz.slices_grid(vol_filtered, n_slices=5, display_figure=True)
        ```
        ![maximum-filter-before](../../assets/screenshots/maximum_filter_original.png)
        ![maximum-filter-after](../../assets/screenshots/maximum_filter_processed.png)

    """
    if size is None:
        if footprint is None:
            raise RuntimeError('no footprint or filter size provided')

    if dask:
        if not isinstance(volume, da.Array):
            volume = da.from_array(volume, chunks=chunks)
        dask_volume = dask_ndfilters.maximum_filter(volume, size, footprint, **kwargs)
        res = dask_volume.compute()
        return res
    else:
        res = ndimage.maximum_filter(volume, size, footprint, **kwargs)
        return res


def minimum(
    volume: np.ndarray,
    size: float = None,
    footprint: np.ndarray = None,
    dask: bool = False,
    chunks: str = 'auto',
    **kwargs,
) -> np.ndarray:
    """
    Applies a minimum filter to the input volume using `scipy.ndimage.minimum_filter` or `dask_image.ndfilters.minimum_filter`.

    Args:
        volume (np.ndarray): The input image or volume.
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

        qim3d.viz.slices_grid(vol, n_slices=5, slice_positions = [10, 31, 63, 95, 120], display_figure=True)
        qim3d.viz.slices_grid(vol_filtered, n_slices=5, slice_positions = [10, 31, 63, 95, 120], display_figure=True)
        ```
        ![minimum-filter-before](../../assets/screenshots/minimum_filter_original.png)
        ![minimum-filter-after](../../assets/screenshots/minimum_filter_processed.png)

    """
    if size is None:
        if footprint is None:
            raise RuntimeError('no footprint or filter size provided')

    if dask:
        if not isinstance(volume, da.Array):
            volume = da.from_array(volume, chunks=chunks)
        dask_volume = dask_ndfilters.minimum_filter(volume, size, footprint, **kwargs)
        res = dask_volume.compute()
        return res
    else:
        res = ndimage.minimum_filter(volume, size, footprint, **kwargs)
        return res


def tophat(volume: np.ndarray, dask: bool = False, **kwargs):
    """
    Remove background from the volume.

    Args:
        volume (np.ndarray): The volume to remove background from.
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

        qim3d.viz.slices_grid(vol, n_slices=5, slice_positions = [10, 31, 63, 95, 120], display_figure=True)
        qim3d.viz.slices_grid(vol_filtered, n_slices=5, slice_positions = [10, 31, 63, 95, 120], display_figure=True)
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
        volume = np.invert(volume)

    selem = morphology.ball(radius)
    volume = volume - morphology.white_tophat(volume, selem)

    if background == 'bright':
        volume = np.invert(volume)

    return volume
