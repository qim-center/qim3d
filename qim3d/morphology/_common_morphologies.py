import numpy as np
import pygorpho as pg
import scipy.ndimage as ndi

from qim3d.utils import log


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
        log.debug('Using int to generate np.ones((k,k,k))')
        return np.ones((k, k, k), dtype=bool)

    elif isinstance(k, tuple):
        if len(k) == 1 and isinstance(k[0], int):
            log.debug(
                'Using tuple with 1 element. Generating np.ones((k[0], k[0], k[0]))'
            )
            return np.ones((k[0], k[0], k[0]), dtype=bool)
        elif len(k) == 3 and all(isinstance(x, int) for x in k):
            log.debug(
                'Using tuple with 3 elements. Generating np.ones((k[0], k[1], k[2]))'
            )
            return np.ones((k[0], k[1], k[2]), dtype=bool)
        else:
            err = 'Tuple input must be of length 1 or 3 with integer elements.'
            raise ValueError(err)

    elif isinstance(k, np.ndarray):
        if k.ndim == 3:
            log.debug('Using provided ndarray with shape %s', k.shape)
            return k
        else:
            err = 'ndarray kernel must be 3-dimensional.'
            raise ValueError(err)

    else:
        err = 'Kernel input must be int, tuple, or 3D np.ndarray.'
        raise TypeError(err)


def dilate(
    volume: np.ndarray,
    kernel: int | np.ndarray,
    method: str = 'pygorpho.linear',
    **kwargs,
) -> np.ndarray:
    """
    Performs morphological dilation on a 3D volume using CPU or GPU-accelerated methods.

    Dilation enlarges bright regions (foreground) and shrinks dark regions (background). It is commonly used to close small holes, connect disjoint features, or thicken object boundaries.

    This function supports efficient GPU acceleration using the `pygorpho` library, based on zonohedral approximations. If a GPU is not available, it is recommended to use the 'scipy.ndimage' method.

    Args:
        volume (np.ndarray): The input 3D volume.
        kernel (int or np.ndarray): The structuring element.
            * If method is 'pygorpho.linear': Must be an integer representing the radius of the ball-shaped kernel.
            * If method is 'pygorpho.flat' or 'scipy.ndimage': Must be a 3D numpy array defining the footprint.
        method (str, optional): The backend implementation to use. Defaults to 'pygorpho.linear'.
            * 'pygorpho.linear': GPU-accelerated. Best for large, spherical kernels.
            * 'pygorpho.flat': GPU-accelerated. Supports arbitrary kernel shapes.
            * 'scipy.ndimage': CPU-based. Standard implementation (slower for large volumes).
        **kwargs (Any): Additional keyword arguments passed to the underlying method.

    Returns:
        dilated_vol (np.ndarray):
            The dilated volume.

    !!! quote "Reference"
        The GPU methods implement the algorithms described in:
        [Zonohedral Approximation of Spherical Structuring Element for Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf).

    Example:
        ```python
        import qim3d
        import numpy as np

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(noise_scale=0.025, seed=50)

        # Visualize synthetic volume
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_original.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Apply dilation
        vol_dilated = qim3d.morphology.dilate(vol, kernel=(8,8,8), method='scipy.ndimage')

        # Visualize
        qim3d.viz.volumetric(vol_dilated)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_dilated.html" width="100%" height="500" frameborder="0"></iframe>

    """

    try:
        volume = np.asarray(volume)
    except TypeError as e:
        err = 'Input volume must be array-like.'
        raise TypeError(err) from e

    assert len(volume.shape) == 3, 'Volume must be three-dimensional.'

    if method == 'pygorpho.flat':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.dilate(volume, kernel, **kwargs)

    elif method == 'pygorpho.linear':
        assert isinstance(kernel, int), (
            'Kernel is generated within function and must therefore be an integer.'
        )

        linesteps, linelens = pg.strel.flat_ball_approx(kernel)

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.linear_dilate(volume, linesteps, linelens)

    elif method == 'scipy.ndimage':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        return ndi.grey_dilation(volume, footprint=kernel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def erode(
    volume: np.ndarray,
    kernel: int | np.ndarray,
    method: str = 'pygorpho.linear',
    **kwargs,
) -> np.ndarray:
    """
    Performs morphological erosion on a 3D volume using CPU or GPU-accelerated methods.

    Erosion shrinks bright regions (foreground) and enlarges dark regions (background). It is commonly used to remove small noise (salt noise), detach touching objects, or thin out features.

    This function supports efficient GPU acceleration using the `pygorpho` library. If a GPU is not available, it is recommended to use the 'scipy.ndimage' method.

    Args:
        volume (np.ndarray): The input 3D volume.
        kernel (int or np.ndarray): The structuring element.
            * If method is 'pygorpho.linear': Must be an integer representing the radius of the ball-shaped kernel.
            * If method is 'pygorpho.flat' or 'scipy.ndimage': Must be a 3D numpy array defining the footprint.
        method (str, optional): The backend implementation to use. Defaults to 'pygorpho.linear'.
            * 'pygorpho.linear': GPU-accelerated. Best for large, spherical kernels.
            * 'pygorpho.flat': GPU-accelerated. Supports arbitrary kernel shapes.
            * 'scipy.ndimage': CPU-based. Standard implementation (slower for large volumes).
        **kwargs (Any): Additional keyword arguments passed to the underlying method.

    Returns:
        eroded_vol (np.ndarray):
            The eroded volume.

    !!! quote "Reference"
        The GPU methods implement the algorithms described in:
        [Zonohedral Approximation of Spherical Structuring Element for Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf).

    Example:
        ```python
        import qim3d
        import numpy as np

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(noise_scale=0.025, seed=50)

        # Visualize synthetic volume
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_original.html" width="100%" height="500" frameborder="0"></iframe>
        ```python
        # Apply erosion
        vol_eroded = qim3d.morphology.erode(vol, kernel=(10,10,10), method='scipy.ndimage')

        # Visualize
        qim3d.viz.volumetric(vol_eroded)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_eroded.html" width="100%" height="500" frameborder="0"></iframe>

    """

    try:
        volume = np.asarray(volume)
    except TypeError as e:
        err = 'Input volume must be array-like.'
        raise TypeError(err) from e

    assert len(volume.shape) == 3, 'Volume must be three-dimensional.'

    if method == 'pygorpho.flat':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.erode(volume, kernel, **kwargs)

    elif method == 'pygorpho.linear':
        assert isinstance(kernel, int), (
            'Kernel is generated within function and must therefore be an integer.'
        )

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(kernel)
        return pg.flat.linear_erode(volume, linesteps, linelens, **kwargs)

    elif method == 'scipy.ndimage':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        return ndi.grey_erosion(volume, footprint=kernel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def opening(
    volume: np.ndarray,
    kernel: int | np.ndarray,
    method: str = 'pygorpho.linear',
    **kwargs,
) -> np.ndarray:
    """
    Performs morphological opening on a 3D volume using CPU or GPU-accelerated methods.

    Opening is defined as an **erosion** followed by a **dilation**. It is primarily used to remove small bright objects (salt noise) from the background while preserving the shape and size of larger objects. It smooths object contours by breaking narrow isthmuses and eliminating thin protrusions.

    This function supports efficient GPU acceleration using the `pygorpho` library. If a GPU is not available, it is recommended to use the [scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) method.

    Args:
        volume (np.ndarray): The input 3D volume.
        kernel (int or np.ndarray): The structuring element.
            * If method is 'pygorpho.linear': Must be an integer representing the radius of the ball-shaped kernel.
            * If method is 'pygorpho.flat' or 'scipy.ndimage': Must be a 3D numpy array defining the footprint.
        method (str, optional): The backend implementation to use. Defaults to 'pygorpho.linear'.
            * 'pygorpho.linear': GPU-accelerated. Best for large, spherical kernels.
            * 'pygorpho.flat': GPU-accelerated. Supports arbitrary kernel shapes.
            * 'scipy.ndimage': CPU-based. Standard implementation (slower for large volumes).
        **kwargs (Any): Additional keyword arguments passed to the underlying method.

    Returns:
        opened_vol (np.ndarray):
            The opened volume.

    !!! quote "Reference"
        The GPU methods implement the algorithms described in:
        [Zonohedral Approximation of Spherical Structuring Element for Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf).

    Example:
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

        <iframe src="https://platform.qim.dk/k3d/zonohedra_noised_volume.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Apply opening
        vol_opened = qim3d.morphology.opening(vol_noised, kernel=(6,6,6), method='scipy.ndimage')

        # Visualize
        qim3d.viz.volumetric(vol_opened)
        ```

        <iframe src="https://platform.qim.dk/k3d/zonohedra_opening.html" width="100%" height="500" frameborder="0"></iframe>

    """
    try:
        volume = np.asarray(volume)
    except TypeError as e:
        err = 'Input volume must be array-like.'
        raise TypeError(err) from e

    assert len(volume.shape) == 3, 'Volume must be three-dimensional.'

    if method == 'pygorpho.flat':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.open(volume, kernel, **kwargs)

    elif method == 'pygorpho.linear':
        assert isinstance(kernel, int), (
            'Kernel is generated within function and must therefore be an integer.'
        )

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(kernel)
        return pg.flat.linear_open(volume, linesteps, linelens, **kwargs)

    elif method == 'scipy.ndimage':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        return ndi.grey_opening(volume, footprint=kernel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def closing(
    volume: np.ndarray,
    kernel: int | np.ndarray,
    method: str = 'pygorpho.linear',
    **kwargs,
) -> np.ndarray:
    """
    Performs morphological closing on a 3D volume using CPU or GPU-accelerated methods.

    Closing is defined as a **dilation** followed by an **erosion**. It is primarily used to fill small dark holes, cracks, or gaps within bright objects while preserving their overall shape and size. It smooths object contours by fusing narrow breaks and filling small depressions.

    This function supports efficient GPU acceleration using the `pygorpho` library. If a GPU is not available, it is recommended to use the [scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) method.

    Args:
        volume (np.ndarray): The input 3D volume.
        kernel (int or np.ndarray): The structuring element.
            * If method is 'pygorpho.linear': Must be an integer representing the radius of the ball-shaped kernel.
            * If method is 'pygorpho.flat' or 'scipy.ndimage': Must be a 3D numpy array defining the footprint.
        method (str, optional): The backend implementation to use. Defaults to 'pygorpho.linear'.
            * 'pygorpho.linear': GPU-accelerated. Best for large, spherical kernels.
            * 'pygorpho.flat': GPU-accelerated. Supports arbitrary kernel shapes.
            * 'scipy.ndimage': CPU-based. Standard implementation (slower for large volumes).
        **kwargs (Any): Additional keyword arguments passed to the underlying method.

    Returns:
        closed_vol (np.ndarray):
            The closed volume.

    !!! quote "Reference"
        The GPU methods implement the algorithms described in:
        [Zonohedral Approximation of Spherical Structuring Element for Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf).

    Example:
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
        <iframe src="https://platform.qim.dk/k3d/zonohedra_cube.html" width="100%" height="500" frameborder="0"></iframe>
        ```python
        # Apply closing
        cube_closed = qim3d.morphology.closing(cube, kernel=(15,15,15), method='scipy.ndimage')

        # Visualize
        qim3d.viz.volumetric(cube_closed)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_cube_closed.html" width="100%" height="500" frameborder="0"></iframe>

    """

    try:
        volume = np.asarray(volume)
    except TypeError as e:
        err = 'Input volume must be array-like.'
        raise TypeError(err) from e

    assert len(volume.shape) == 3, 'Volume must be three-dimensional.'

    if method == 'pygorpho.flat':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.close(volume, kernel, **kwargs)

    elif method == 'pygorpho.linear':
        assert isinstance(kernel, int), (
            'Kernel is generated within function and must therefore be an integer.'
        )

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(kernel)
        return pg.flat.linear_close(volume, linesteps, linelens, **kwargs)

    elif method == 'scipy.ndimage':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        return ndi.grey_closing(volume, footprint=kernel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def black_tophat(
    volume: np.ndarray,
    kernel: int | np.ndarray,
    method: str = 'pygorpho.linear',
    **kwargs,
) -> np.ndarray:
    """
    Performs the black top-hat transform on a 3D volume.

    The black top-hat transform is defined as the difference between the morphological closing of the volume and the original volume (Closing - Input). It is used to extract dark features and valleys that are smaller than the structuring element (kernel) from a brighter background. This is particularly effective for background correction or isolating small dark structures in a non-uniformly lit image.

    This function supports efficient GPU acceleration using the `pygorpho` library. If a GPU is not available, it is recommended to use the [scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.black_tophat.html) method.

    Args:
        volume (np.ndarray): The input 3D volume.
        kernel (int or np.ndarray): The structuring element.
            * If method is 'pygorpho.linear': Must be an integer representing the radius of the ball-shaped kernel.
            * If method is 'pygorpho.flat' or 'scipy.ndimage': Must be a 3D numpy array defining the footprint.
        method (str, optional): The backend implementation to use. Defaults to 'pygorpho.linear'.
            * 'pygorpho.linear': GPU-accelerated. Best for large, spherical kernels.
            * 'pygorpho.flat': GPU-accelerated. Supports arbitrary kernel shapes.
            * 'scipy.ndimage': CPU-based. Standard implementation (slower for large volumes).
        **kwargs (Any): Additional keyword arguments passed to the underlying method.

    Returns:
        bothat_vol (np.ndarray):
            The processed volume containing the extracted dark features.

    !!! quote "Reference"
        The GPU methods implement the algorithms described in:
        [Zonohedral Approximation of Spherical Structuring Element for Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf).

    Example:
        ```python
        import qim3d
        import numpy as np

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(noise_scale=0.025, seed=50)

        # Visualize synthetic volume
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_original.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Apply the black top-hat to extract dark details
        vol_black = qim3d.morphology.black_tophat(vol, kernel=(10,10,10), method='scipy.ndimage')

        qim3d.viz.volumetric(vol_black)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_black_tophat.html" width="100%" height="500" frameborder="0"></iframe>

    """

    try:
        volume = np.asarray(volume)
    except TypeError as e:
        err = 'Input volume must be array-like.'
        raise TypeError(err) from e

    assert len(volume.shape) == 3, 'Volume must be three-dimensional.'

    if method == 'pygorpho.flat':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.bothat(volume, kernel, **kwargs)

    elif method == 'pygorpho.linear':
        assert isinstance(kernel, int), (
            'Kernel is generated within function and must therefore be an integer.'
        )

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(kernel)
        return pg.flat.bothat(volume, linesteps, linelens, **kwargs)

    elif method == 'scipy.ndimage':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        return ndi.black_tophat(volume, footprint=kernel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def white_tophat(
    volume: np.ndarray,
    kernel: int | np.ndarray,
    method: str = 'pygorpho.linear',
    **kwargs,
) -> np.ndarray:
    """
    Performs the white top-hat transform on a 3D volume.

    The white top-hat transform is defined as the difference between the original volume and its morphological opening (Input - Opening). It is used to extract bright features and peaks that are smaller than the structuring element (kernel) from a darker background. This is a powerful tool for background subtraction, enhancing small bright spots, or correcting uneven illumination in a volume.

    This function supports efficient GPU acceleration using the `pygorpho` library. If a GPU is not available, it is recommended to use the [scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.white_tophat.html) method.

    Args:
        volume (np.ndarray): The input 3D volume.
        kernel (int or np.ndarray): The structuring element.
            * If method is 'pygorpho.linear': Must be an integer representing the radius of the ball-shaped kernel.
            * If method is 'pygorpho.flat' or 'scipy.ndimage': Must be a 3D numpy array defining the footprint.
        method (str, optional): The backend implementation to use. Defaults to 'pygorpho.linear'.
            * 'pygorpho.linear': GPU-accelerated. Best for large, spherical kernels.
            * 'pygorpho.flat': GPU-accelerated. Supports arbitrary kernel shapes.
            * 'scipy.ndimage': CPU-based. Standard implementation (slower for large volumes).
        **kwargs (Any): Additional keyword arguments passed to the underlying method.

    Returns:
        tophat_vol (np.ndarray):
            The processed volume containing the extracted bright features.

    !!! quote "Reference"
        The GPU methods implement the algorithms described in:
        [Zonohedral Approximation of Spherical Structuring Element for Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf).

    Example:
        ```python
        import qim3d
        import numpy as np

        # Generate tubular synthetic blob
        vol = qim3d.generate.volume(noise_scale=0.025, seed=50)

        # Visualize synthetic volume
        qim3d.viz.volumetric(vol)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_original.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Apply the white top-hat to extract bright details
        vol_white = qim3d.morphology.white_tophat(vol, kernel=(10,10,10), method='scipy.ndimage')

        qim3d.viz.volumetric(vol_white)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_white_tophat.html" width="100%" height="500" frameborder="0"></iframe>

    """

    try:
        volume = np.asarray(volume)
    except TypeError as e:
        err = 'Input volume must be array-like.'
        raise TypeError(err) from e

    assert len(volume.shape) == 3, 'Volume must be three-dimensional.'

    if method == 'pygorpho.flat':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.tophat(volume, kernel, **kwargs)

    elif method == 'pygorpho.linear':
        assert isinstance(kernel, int), (
            'Kernel is generated within function and must therefore be an integer.'
        )

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(kernel)
        return pg.flat.tophat(volume, linesteps, linelens, **kwargs)

    elif method == 'scipy.ndimage':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        return ndi.white_tophat(volume, footprint=kernel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)
