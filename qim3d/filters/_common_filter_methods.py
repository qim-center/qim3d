"""Provides filter functions and classes for image processing"""

import abc
import inspect
from collections.abc import Callable

import dask.array as da
import dask_image.ndfilters as dask_ndfilters
import numpy as np
from scipy import ndimage
from skimage import morphology

from qim3d.utils import log


class FilterBase:
    def __init__(
        self,
        *args,
        dask: bool = False,
        chunks: str = 'auto',
        save_output: bool = False,
        **kwargs,
    ):
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
    def __call__(self, input: np.ndarray) -> np.ndarray:
        pass


class Filter(FilterBase):
    def __init__(self, func: Callable, *args, **kwargs):
        self.func = func
        super().__init__(*args, **kwargs)

    def __call__(self, input):
        return self.func(input, *self.args, **self.kwargs)


class Threshold(FilterBase):
    def __init__(self, threshold: float, *args, **kwargs):
        self.threshold = threshold
        super().__init__(*args, **kwargs)

    def __call__(self, input: np.ndarray):
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

    def __call__(self, input: np.ndarray) -> np.ndarray:
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
            volume=input,
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
            volume=input,
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
            volume=input,
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
    def __call__(self, input: np.ndarray) -> np.ndarray:
        return normalize(input)


class Pipeline:
    """
    Orchestrates a sequential chain of image processing operations.

    This class allows you to build a reproducible workflow (or pipeline) by stacking multiple filters or functions together.

    The output of one step becomes the input to the next. It supports standard Python callables (functions, lambdas) and `qim3d.filters.Filter` objects. It is particularly useful for automating preprocessing routines, such as applying denoising, followed by thresholding, and finally morphological operations.

    **Key Features:**

    * **Flexibility:** Accepts any callable that takes a single argument (the image) and returns a modified result.
    * **Intermediate Results:** Access data from specific steps in the middle of the chain using the `save_output=True` flag on supported filters.
    * **Extensibility:** Add steps dynamically using the `.append()` method.

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

    def __init__(self, *args: type[FilterBase]):
        """
        Represents a sequence of image filters.

        Args:
            *args: Variable number of filter instances to be applied sequentially.

        """
        self.filters = []
        self.saved_outputs = []

        for fn in args:
            self._add_filter(fn)

    def _add_filter(self, fn: type[FilterBase] | Callable):
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
                raise TypeError(
                    f'Pipeline only accepts callable objects. Your object is of type "{type(fn)}".'
                )
            signature = inspect.signature(fn)
            if signature.parameters != 1:
                raise TypeError(
                    f'Pipeline only accepts callables that take one argument. Yours takes {signature.parameters}.'
                )
        self.filters.append(fn)

    def append(self, fn: FilterBase | Callable):
        """
        Adds a new processing step to the end of the pipeline.

        Args:
            fn (FilterBase or Callable): The filter or function to append.

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

    def __call__(self, input: np.ndarray):
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


def normalize(vol: np.ndarray) -> np.ndarray:
    return 255 * int((vol - vol.min()) / vol.max())


def gaussian(
    volume: np.ndarray, sigma: float, dask: bool = False, chunks: str = 'auto', **kwargs
) -> np.ndarray:
    """
    Applies a Gaussian blur to smooth the 3D volume and reduce noise.

    This function acts as a low-pass filter, effectively attenuating high-frequency noise (denoising) while blurring edges. The function wraps `scipy.ndimage.gaussian_filter` for standard processing and `dask_image.ndfilters.gaussian_filter` for parallelized, memory-efficient computation on large datasets.

    Args:
        volume (np.ndarray): The input 3D image stack.
        sigma (float or sequence of floats): The standard deviation for the Gaussian kernel. This controls the amount of blurring; higher values result in a smoother image. Can be a single float (isotropic) or a sequence (anisotropic) for each axis.
        dask (bool, optional): If `True`, utilizes Dask for parallel processing. Useful for large volumes that might not fit entirely in memory during processing. Defaults to `False`.
        chunks (int, tuple, or str, optional): The chunk size for Dask processing (e.g., 'auto'). Only used if `dask=True`.
        **kwargs (Any): Additional keyword arguments passed to the underlying Gaussian filter function (e.g., `mode`, `cval`).

    Returns:
        filtered_vol (np.ndarray):
            The smoothed volume.

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


def sobel(vol: np.ndarray, dask: bool = False):
    """
    Computes the magnitude of the intensity gradient using the Sobel operator to detect edges.

    This function approximates the gradient of the image intensity function. It applies the Sobel filter along each axis independently (Gx, Gy, Gz) and computes the Euclidean magnitude (sqrt(Gx^2 + Gy^2 + Gz^2)). High values in the output correspond to regions with sharp intensity changes, effectively highlighting boundaries and edges within the volume.

    Args:
        vol (np.ndarray): The input image or volume.
        dask (bool, optional): If `True`, utilizes Dask for parallel processing. Defaults to `False`.

    Returns:
        filtered_vol (np.ndarray):
            The gradient magnitude map.

    Note:
        **Data Type Warning:** The Sobel operator can produce negative values and large magnitudes. If the input `vol` is of an unsigned integer type (e.g., `uint8`, `uint16`), arithmetic overflow or wrapping may occur during calculation. It is strongly recommended to cast the input to a float or signed integer type (e.g., `int64` or `float32`) before applying this filter.

    Example:
        ```python
        import qim3d
        # Cast to int64 to prevent overflow during gradient calculation
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
    sob0 = ndimage.sobel(vol, 0)
    sob1 = ndimage.sobel(vol, 1)
    if vol.ndim == 3:
        sob2 = ndimage.sobel(vol, 2)
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
    Applies a median filter to remove impulse noise (salt-and-pepper) while preserving edges.

    This non-linear filter replaces each voxel with the median value of its neighbors. Unlike Gaussian blurring, which can fuzzy boundaries, the median filter is particularly effective at despeckling images (removing isolated bright or dark outliers) without blurring sharp transitions or structural details. It supports both standard in-memory processing and parallelized execution via Dask for large datasets.

    Args:
        volume (np.ndarray): The input 3D image stack or 2D image.
        size (int or tuple, optional): The shape of the neighborhood (kernel) used for the median calculation. Can be a single integer (e.g., `3` for a 3x3x3 box) or a tuple defining dimensions per axis. Must be provided if `footprint` is None.
        footprint (np.ndarray, optional): A boolean array defining a custom neighborhood shape. Only elements where `footprint` is `True` are considered for the median.
        dask (bool, optional): If `True`, utilizes Dask for parallel processing. Useful for large volumes that exceed memory limits. Defaults to `False`.
        chunks (int, tuple, or str, optional): The chunk size for Dask processing. Defaults to 'auto'.
        **kwargs (Any): Additional keyword arguments passed to the underlying filter function (e.g., `mode` for boundary handling).

    Returns:
        filtered_vol (np.ndarray):
            The denoised volume.

    Raises:
        RuntimeError: If neither `size` nor `footprint` is provided.

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
    Applies a maximum filter (grayscale dilation) to the input volume.

    This operation replaces the value of each voxel with the maximum value found within its defined neighborhood. It results in the expansion (dilation) of bright regions and the erosion of dark regions. It is commonly used for finding local peaks, reducing salt noise (dark spots in bright regions), or as the first step in morphological gradients.

    Args:
        volume (np.ndarray): The input image or volume.
        size (int or tuple, optional): The size of the neighborhood window (kernel). Must be provided if `footprint` is None.
        footprint (np.ndarray, optional): A boolean array defining the specific shape of the neighborhood. Elements where `footprint` is `True` are included in the maximum calculation.
        dask (bool, optional): If `True`, uses Dask for parallelized, chunked execution. Defaults to `False`.
        chunks (int, tuple, or str, optional): The chunk size for Dask arrays. Defaults to 'auto'.
        **kwargs (Any): Additional keyword arguments passed to the underlying filter function (e.g., `mode`, `cval`).

    Returns:
        filtered_vol (np.ndarray):
            The filtered volume with the same shape as the input.

    Raises:
        RuntimeError: If neither `size` nor `footprint` is defined.

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
    Applies a minimum filter (grayscale erosion) to the input volume.

    This operation replaces the value of each voxel with the minimum value found within its defined neighborhood. It results in the erosion of bright regions and the expansion (dilation) of dark regions. It is commonly used to remove salt noise (isolated bright pixels), darken the overall image, or separate touching objects in bright foregrounds.

    Args:
        volume (np.ndarray): The input image or volume.
        size (int or tuple, optional): The size of the neighborhood window (kernel). Must be provided if `footprint` is None.
        footprint (np.ndarray, optional): A boolean array defining the specific shape of the neighborhood. Elements where `footprint` is `True` are included in the minimum calculation.
        dask (bool, optional): If `True`, uses Dask for parallelized, chunked execution. Defaults to `False`.
        chunks (int, tuple, or str, optional): The chunk size for Dask arrays. Defaults to 'auto'.
        **kwargs (Any): Additional keyword arguments passed to the underlying filter function.

    Returns:
        filtered_vol (np.ndarray):
            The filtered volume with the same shape as the input.

    Raises:
        RuntimeError: If neither `size` nor `footprint` is defined.

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
    Applies a morphological opening (or closing) to estimate the background or smooth features.

    Despite the name, this function uses the Top-Hat transform intermediate to perform a Morphological Opening (for dark backgrounds) or Closing (for bright backgrounds).

    * **Dark Background** (Bright Objects): Performs an Opening. This removes bright features smaller than the `radius` (e.g., noise, small particles), effectively estimating the underlying background intensity.
    * **Bright Background** (Dark Objects): Performs a Closing. This fills in dark features smaller than the `radius`, estimating the background in inverted contrast.

    Args:
        volume (np.ndarray): The input image or volume.
        dask (bool, optional): Not currently supported for this filter; defaults to SciPy implementation.
        **kwargs (Any):
            * **radius** (float): The radius of the ball-shaped structuring element. Features smaller than this size will be removed/smoothed. Defaults to 3.
            * **background** (str): The brightness of the background relative to the objects.
                * `'dark'`: Use for bright objects on a dark background.
                * `'bright'`: Use for dark objects on a bright background. Defaults to `'dark'`.

    Returns:
        filtered_vol (np.ndarray):
            The processed volume (the background estimation).

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
