import logging
from typing import Literal

import numpy as np
import scipy.ndimage as ndi

_logger = logging.getLogger(__name__)


def _create_kernel(k: int | tuple | np.ndarray) -> np.ndarray:
    """
    Create a 3D kernel from various input types.

    Args:
        k (int | tuple | np.ndarray):
            - If int, returns a cubic kernel of shape (k,k,k).
            - If tuple of length 1, behaves as if given int.
            - If tuple of length 3, returns kernel with that shape.
            - If ndarray, returns the array if it has 3 dimensions.

    Returns:
        np.ndarray: A 3D kernel.

    """
    if isinstance(k, int):
        _logger.debug("Using int to generate np.ones((k,k,k))")
        return np.ones((k, k, k), dtype=bool)

    elif isinstance(k, tuple):
        if len(k) == 1 and isinstance(k[0], int):
            _logger.debug(
                "Using tuple with 1 element. Generating np.ones((k[0], k[0], k[0]))"
            )
            return np.ones((k[0], k[0], k[0]), dtype=bool)
        elif len(k) == 3 and all(isinstance(x, int) for x in k):
            _logger.debug(
                "Using tuple with 3 elements. Generating np.ones((k[0], k[1], k[2]))"
            )
            return np.ones((k[0], k[1], k[2]), dtype=bool)
        else:
            err = "Tuple input must be of length 1 or 3 with integer elements."
            raise ValueError(err)

    elif isinstance(k, np.ndarray):
        if k.ndim == 3:
            _logger.debug("Using provided ndarray with shape %s", k.shape)
            return k
        else:
            err = "ndarray kernel must be 3-dimensional."
            raise ValueError(err)

    else:
        err = "Kernel input must be int, tuple, or 3D np.ndarray."
        raise TypeError(err)


def dilate(
    volume: np.ndarray,
    kernel: int | tuple | np.ndarray,
    method: Literal["scipy.ndimage"] = "scipy.ndimage",
    **kwargs,
) -> np.ndarray:
    """
    Performs morphological dilation on a 3D volume.

    Dilation enlarges bright regions (foreground) and shrinks dark regions (background). It is commonly used to close small holes, connect disjoint features, or thicken object boundaries.

    Args:
        volume (np.ndarray): The input 3D volume.
        kernel (int, tuple or np.ndarray): The structuring element. An int gives a cube of that side length, a tuple of 3 ints a box of that shape, and a 3D array is used directly as the footprint.
        method (Literal["scipy.ndimage"], optional): The backend implementation to use. Currently only 'scipy.ndimage' is supported. Defaults to 'scipy.ndimage'.
        **kwargs (Any): Additional keyword arguments passed to the underlying method.

    Returns:
        dilated_vol (np.ndarray):
            The dilated volume.

    Example:
        <!-- k3d-example: dilation -->

        ```python
        import qim3d
        import numpy as np

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(noise_scale=0.025, seed=50)

        # Visualize synthetic volume
        qim3d.viz.volumetric(vol)
        ```
        <!-- k3d-output: dilation/original -->

        ```python
        # Apply dilation
        vol_dilated = qim3d.morphology.dilate(vol, kernel=(8,8,8), method='scipy.ndimage')

        # Visualize
        qim3d.viz.volumetric(vol_dilated)
        ```
        <!-- k3d-output: dilation/dilated -->
        <!-- /k3d-example -->
    """

    try:
        volume = np.asarray(volume)
    except TypeError as e:
        err = "Input volume must be array-like."
        raise TypeError(err) from e

    assert len(volume.shape) == 3, "Volume must be three-dimensional."

    if method == "scipy.ndimage":
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, "Kernel must a 3D np.ndarray."

        return ndi.grey_dilation(volume, footprint=kernel, **kwargs)

    else:
        err = f"Unknown method '{method}'. Supported: 'scipy.ndimage'."
        raise ValueError(err)


def erode(
    volume: np.ndarray,
    kernel: int | tuple | np.ndarray,
    method: Literal["scipy.ndimage"] = "scipy.ndimage",
    **kwargs,
) -> np.ndarray:
    """
    Performs morphological erosion on a 3D volume.

    Erosion shrinks bright regions (foreground) and enlarges dark regions (background). It is commonly used to remove small noise (salt noise), detach touching objects, or thin out features.

    Args:
        volume (np.ndarray): The input 3D volume.
        kernel (int, tuple or np.ndarray): The structuring element. An int gives a cube of that side length, a tuple of 3 ints a box of that shape, and a 3D array is used directly as the footprint.
        method (Literal["scipy.ndimage"], optional): The backend implementation to use. Currently only 'scipy.ndimage' is supported. Defaults to 'scipy.ndimage'.
        **kwargs (Any): Additional keyword arguments passed to the underlying method.

    Returns:
        eroded_vol (np.ndarray):
            The eroded volume.

    Example:
        <!-- k3d-example: erosion -->

        ```python
        import qim3d
        import numpy as np

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(noise_scale=0.025, seed=50)

        # Visualize synthetic volume
        qim3d.viz.volumetric(vol)
        ```
        <!-- k3d-output: erosion/original -->
        ```python
        # Apply erosion
        vol_eroded = qim3d.morphology.erode(vol, kernel=(10,10,10), method='scipy.ndimage')

        # Visualize
        qim3d.viz.volumetric(vol_eroded)
        ```
        <!-- k3d-output: erosion/eroded -->
        <!-- /k3d-example -->
    """

    try:
        volume = np.asarray(volume)
    except TypeError as e:
        err = "Input volume must be array-like."
        raise TypeError(err) from e

    assert len(volume.shape) == 3, "Volume must be three-dimensional."

    if method == "scipy.ndimage":
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, "Kernel must a 3D np.ndarray."

        return ndi.grey_erosion(volume, footprint=kernel, **kwargs)

    else:
        err = f"Unknown method '{method}'. Supported: 'scipy.ndimage'."
        raise ValueError(err)


def opening(
    volume: np.ndarray,
    kernel: int | tuple | np.ndarray,
    method: Literal["scipy.ndimage"] = "scipy.ndimage",
    **kwargs,
) -> np.ndarray:
    """
    Performs morphological opening on a 3D volume.

    Opening is defined as an **erosion** followed by a **dilation**. It is primarily used to remove small bright objects (salt noise) from the background while preserving the shape and size of larger objects. It smooths object contours by breaking narrow isthmuses and eliminating thin protrusions.

    Args:
        volume (np.ndarray): The input 3D volume.
        kernel (int, tuple or np.ndarray): The structuring element. An int gives a cube of that side length, a tuple of 3 ints a box of that shape, and a 3D array is used directly as the footprint.
        method (Literal["scipy.ndimage"], optional): The backend implementation to use. Currently only 'scipy.ndimage' is supported. Defaults to 'scipy.ndimage'.
        **kwargs (Any): Additional keyword arguments passed to the underlying method.

    Returns:
        opened_vol (np.ndarray):
            The opened volume.

    Example:
        <!-- k3d-example: opening -->

        ```python
        import qim3d
        import numpy as np

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(noise_scale=0.025, seed=50)

        # Add noise to the data
        vol_noised = qim3d.generate.background(
            background_shape=vol.shape,
            apply_method = 'add',
            apply_to = vol
        )

        # Visualize synthetic volume
        qim3d.viz.volumetric(vol_noised, grid_visible=True)
        ```

        <!-- k3d-output: opening/noisy -->

        ```python
        # Apply opening
        vol_opened = qim3d.morphology.opening(vol_noised, kernel=(6,6,6), method='scipy.ndimage')

        # Visualize
        qim3d.viz.volumetric(vol_opened)
        ```

        <!-- k3d-output: opening/opened -->
        <!-- /k3d-example -->
    """
    try:
        volume = np.asarray(volume)
    except TypeError as e:
        err = "Input volume must be array-like."
        raise TypeError(err) from e

    assert len(volume.shape) == 3, "Volume must be three-dimensional."

    if method == "scipy.ndimage":
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, "Kernel must a 3D np.ndarray."

        return ndi.grey_opening(volume, footprint=kernel, **kwargs)

    else:
        err = f"Unknown method '{method}'. Supported: 'scipy.ndimage'."
        raise ValueError(err)


def closing(
    volume: np.ndarray,
    kernel: int | tuple | np.ndarray,
    method: Literal["scipy.ndimage"] = "scipy.ndimage",
    **kwargs,
) -> np.ndarray:
    """
    Performs morphological closing on a 3D volume.

    Closing is defined as a **dilation** followed by an **erosion**. It is primarily used to fill small dark holes, cracks, or gaps within bright objects while preserving their overall shape and size. It smooths object contours by fusing narrow breaks and filling small depressions.

    Args:
        volume (np.ndarray): The input 3D volume.
        kernel (int, tuple or np.ndarray): The structuring element. An int gives a cube of that side length, a tuple of 3 ints a box of that shape, and a 3D array is used directly as the footprint.
        method (Literal["scipy.ndimage"], optional): The backend implementation to use. Currently only 'scipy.ndimage' is supported. Defaults to 'scipy.ndimage'.
        **kwargs (Any): Additional keyword arguments passed to the underlying method.

    Returns:
        closed_vol (np.ndarray):
            The closed volume.

    Example:
        <!-- k3d-example: closing -->

        ```python
        import qim3d
        import numpy as np

        # Generate a cube with a hole through it
        cube = np.zeros((110,110,110))
        cube[10:90, 10:90, 10:90] = 1
        cube[60:70,:,60:70]=0

        # Visualize synthetic volume
        qim3d.viz.volumetric(cube)
        ```
        <!-- k3d-output: closing/perforated-cube -->
        ```python
        # Apply closing
        cube_closed = qim3d.morphology.closing(cube, kernel=(15,15,15), method='scipy.ndimage')

        # Visualize
        qim3d.viz.volumetric(cube_closed)
        ```
        <!-- k3d-output: closing/closed-cube -->
        <!-- /k3d-example -->
    """

    try:
        volume = np.asarray(volume)
    except TypeError as e:
        err = "Input volume must be array-like."
        raise TypeError(err) from e

    assert len(volume.shape) == 3, "Volume must be three-dimensional."

    if method == "scipy.ndimage":
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, "Kernel must a 3D np.ndarray."

        return ndi.grey_closing(volume, footprint=kernel, **kwargs)

    else:
        err = f"Unknown method '{method}'. Supported: 'scipy.ndimage'."
        raise ValueError(err)


def black_tophat(
    volume: np.ndarray,
    kernel: int | tuple | np.ndarray,
    method: Literal["scipy.ndimage"] = "scipy.ndimage",
    **kwargs,
) -> np.ndarray:
    """
    Performs the black top-hat transform on a 3D volume.

    The black top-hat transform is defined as the difference between the morphological closing of the volume and the original volume (Closing - Input). It is used to extract dark features and valleys that are smaller than the structuring element (kernel) from a brighter background. This is particularly effective for background correction or isolating small dark structures in a non-uniformly lit image.

    Args:
        volume (np.ndarray): The input 3D volume.
        kernel (int, tuple or np.ndarray): The structuring element. An int gives a cube of that side length, a tuple of 3 ints a box of that shape, and a 3D array is used directly as the footprint.
        method (Literal["scipy.ndimage"], optional): The backend implementation to use. Currently only 'scipy.ndimage' is supported. Defaults to 'scipy.ndimage'.
        **kwargs (Any): Additional keyword arguments passed to the underlying method.

    Returns:
        bothat_vol (np.ndarray):
            The processed volume containing the extracted dark features.

    Example:
        <!-- k3d-example: black-tophat -->

        ```python
        import qim3d
        import numpy as np

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(noise_scale=0.025, seed=50)

        # Visualize synthetic volume
        qim3d.viz.volumetric(vol)
        ```
        <!-- k3d-output: black-tophat/original -->

        ```python
        # Apply the black top-hat to extract dark details
        vol_black = qim3d.morphology.black_tophat(vol, kernel=(10,10,10), method='scipy.ndimage')

        qim3d.viz.volumetric(vol_black)
        ```
        <!-- k3d-output: black-tophat/dark-details -->
        <!-- /k3d-example -->
    """

    try:
        volume = np.asarray(volume)
    except TypeError as e:
        err = "Input volume must be array-like."
        raise TypeError(err) from e

    assert len(volume.shape) == 3, "Volume must be three-dimensional."

    if method == "scipy.ndimage":
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, "Kernel must a 3D np.ndarray."

        return ndi.black_tophat(volume, footprint=kernel, **kwargs)

    else:
        err = f"Unknown method '{method}'. Supported: 'scipy.ndimage'."
        raise ValueError(err)


def white_tophat(
    volume: np.ndarray,
    kernel: int | tuple | np.ndarray,
    method: Literal["scipy.ndimage"] = "scipy.ndimage",
    **kwargs,
) -> np.ndarray:
    """
    Performs the white top-hat transform on a 3D volume.

    The white top-hat transform is defined as the difference between the original volume and its morphological opening (Input - Opening). It is used to extract bright features and peaks that are smaller than the structuring element (kernel) from a darker background. This is a powerful tool for background subtraction, enhancing small bright spots, or correcting uneven illumination in a volume.

    Args:
        volume (np.ndarray): The input 3D volume.
        kernel (int, tuple or np.ndarray): The structuring element. An int gives a cube of that side length, a tuple of 3 ints a box of that shape, and a 3D array is used directly as the footprint.
        method (Literal["scipy.ndimage"], optional): The backend implementation to use. Currently only 'scipy.ndimage' is supported. Defaults to 'scipy.ndimage'.
        **kwargs (Any): Additional keyword arguments passed to the underlying method.

    Returns:
        tophat_vol (np.ndarray):
            The processed volume containing the extracted bright features.

    Example:
        <!-- k3d-example: white-tophat -->

        ```python
        import qim3d
        import numpy as np

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(noise_scale=0.025, seed=50)

        # Visualize synthetic volume
        qim3d.viz.volumetric(vol)
        ```
        <!-- k3d-output: white-tophat/original -->

        ```python
        # Apply the white top-hat to extract bright details
        vol_white = qim3d.morphology.white_tophat(vol, kernel=(10,10,10), method='scipy.ndimage')

        qim3d.viz.volumetric(vol_white)
        ```
        <!-- k3d-output: white-tophat/bright-details -->
        <!-- /k3d-example -->
    """

    try:
        volume = np.asarray(volume)
    except TypeError as e:
        err = "Input volume must be array-like."
        raise TypeError(err) from e

    assert len(volume.shape) == 3, "Volume must be three-dimensional."

    if method == "scipy.ndimage":
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, "Kernel must a 3D np.ndarray."

        return ndi.white_tophat(volume, footprint=kernel, **kwargs)

    else:
        err = f"Unknown method '{method}'. Supported: 'scipy.ndimage'."
        raise ValueError(err)
