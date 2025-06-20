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
    vol: np.ndarray, kernel: int | np.ndarray, method: str = 'pygorpho.linear', **kwargs
) -> np.ndarray:
    """
    Dilate an image. If method is either pygorpho.linear or pygorpho.flat, the dilation methods from [Zonohedral Approximation of Spherical Structuring Element for Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf) are used. These methods require a GPU, and we therefore recommend using the
    [scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) (scipy.ndimage) if no GPU is available on your current device.

    Args:
        vol (np.ndarray): The volume to dilate.
        kernel (int or np.ndarray): The structuring element/kernel to use while performing dilation. Note that the kernel should be 3D unless if the linear method is used. If this method is used, a kernel resembling a ball will be created with an integer radius.
        method (str, optional): Determines the method for dilation. Use either 'pygorpho.linear', 'pygorpho.flat' or 'scipy.ndimage'. Defaults to 'pygorpho.linear'.
        **kwargs (Any): Additional keyword arguments for the used method. See the documentation for more information.

    Returns:
        dilated_vol (np.ndarray): The dilated volume.


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
        vol = np.asarray(vol)
    except TypeError as e:
        err = 'Input volume must be array-like.'
        raise TypeError(err) from e

    assert len(vol.shape) == 3, 'Volume must be three-dimensional.'

    if method == 'pygorpho.flat':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.dilate(vol, kernel, **kwargs)

    elif method == 'pygorpho.linear':
        assert isinstance(
            kernel, int
        ), 'Kernel is generated within function and must therefore be an integer.'

        linesteps, linelens = pg.strel.flat_ball_approx(kernel)

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.linear_dilate(vol, linesteps, linelens)

    elif method == 'scipy.ndimage':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        return ndi.grey_dilation(vol, footprint=kernel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def erode(
    vol: np.ndarray, kernel: int | np.ndarray, method: str = 'pygorpho.linear', **kwargs
) -> np.ndarray:
    """
    Erode an image. If method is either pygorpho.linear or pygorpho.flat, the erosion methods from [Zonohedral Approximation of Spherical Structuring Element for Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf) are used. These methods require a GPU, and we therefore recommend using the [scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) (scipy.ndimage) if no GPU is available on your current device.

    Args:
            vol (np.ndarray): The volume to erode.
            kernel (int or np.ndarray): The structuring element/kernel to use while performing erosion. Note that the kernel should be 3D unless if the linear method is used. If this method is used, a kernel resembling a ball will be created with an integer radius.
            method (str, optional): Determines the method for erosion. Use either 'pygorpho.linear', 'pygorpho.flat' or 'scipy.ndimage'. Defaults to 'pygorpho.linear'.
            **kwargs (Any): Additional keyword arguments for the used method. See the documentation for more information.

    Returns:
            eroded_vol (np.ndarray): The eroded volume.


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
        vol = np.asarray(vol)
    except TypeError as e:
        err = 'Input volume must be array-like.'
        raise TypeError(err) from e

    assert len(vol.shape) == 3, 'Volume must be three-dimensional.'

    if method == 'pygorpho.flat':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.erode(vol, kernel, **kwargs)

    elif method == 'pygorpho.linear':
        assert isinstance(
            kernel, int
        ), 'Kernel is generated within function and must therefore be an integer.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(kernel)
        return pg.flat.linear_erode(vol, linesteps, linelens, **kwargs)

    elif method == 'scipy.ndimage':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        return ndi.grey_erosion(vol, footprint=kernel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def opening(
    vol: np.ndarray, kernel: int | np.ndarray, method: str = 'pygorpho.linear', **kwargs
) -> np.ndarray:
    """
    Morphologically open a volume.
    If method is either pygorpho.linear or pygorpho.flat, the open methods from [Zonohedral Approximation of Spherical Structuring Element for
    Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf) are used.
    These methods require a GPU, and we therefore recommend using the [scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) (scipy.ndimage) if no GPU is available on your current device.

    Args:
        vol (np.ndarray): The volume to open.
        kernel (int or np.ndarray): The structuring element/kernel to use while performing erosion. Note that the kernel should be 3D unless if the linear method is used. If this method is used, a kernel resembling a ball will be created with an integer radius.
        method (str, optional): Determines the method for opening. Use either 'pygorpho.linear', 'pygorpho.flat' or 'scipy.ndimage'. Defaults to 'pygorpho.linear'.
        **kwargs (Any): Additional keyword arguments for the used method. See the documentation for more information.

    Returns:
        eroded_vol (np.ndarray): The eroded volume.


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
        vol = np.asarray(vol)
    except TypeError as e:
        err = 'Input volume must be array-like.'
        raise TypeError(err) from e

    assert len(vol.shape) == 3, 'Volume must be three-dimensional.'

    if method == 'pygorpho.flat':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.open(vol, kernel, **kwargs)

    elif method == 'pygorpho.linear':
        assert isinstance(
            kernel, int
        ), 'Kernel is generated within function and must therefore be an integer.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(kernel)
        return pg.flat.linear_open(vol, linesteps, linelens, **kwargs)

    elif method == 'scipy.ndimage':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        return ndi.grey_opening(vol, footprint=kernel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def closing(
    vol: np.ndarray, kernel: int | np.ndarray, method: str = 'pygorpho.linear', **kwargs
) -> np.ndarray:
    """
    Morphologically close a volume.
    If method is either pygorpho.linear or pygorpho.flat, the close methods from [Zonohedral Approximation of Spherical Structuring Element for
    Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf) are used.
    These methods require a GPU, and we therefore recommend using the [scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) (scipy.ndimage) if no GPU is available on your current device.

    Args:
        vol (np.ndarray): The volume to be closed.
        kernel (int or np.ndarray): The structuring element/kernel to use while performing opening. Note that the kernel should be 3D unless if the linear method is used. If this method is used, a kernel resembling a ball will be created with an integer radius.
        method (str, optional): Determines the method for closing. Use either 'pygorpho.linear', 'pygorpho.flat' or 'scipy.ndimage'. Defaults to 'pygorpho.linear'.
        **kwargs (Any): Additional keyword arguments for the used method. See the documentation for more information.

    Returns:
        closed_vol (np.ndarray): The closed volume.


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
        vol = np.asarray(vol)
    except TypeError as e:
        err = 'Input volume must be array-like.'
        raise TypeError(err) from e

    assert len(vol.shape) == 3, 'Volume must be three-dimensional.'

    if method == 'pygorpho.flat':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.close(vol, kernel, **kwargs)

    elif method == 'pygorpho.linear':
        assert isinstance(
            kernel, int
        ), 'Kernel is generated within function and must therefore be an integer.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(kernel)
        return pg.flat.linear_close(vol, linesteps, linelens, **kwargs)

    elif method == 'scipy.ndimage':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        return ndi.grey_closing(vol, footprint=kernel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def black_tophat(
    vol: np.ndarray, kernel: int | np.ndarray, method: str = 'pygorpho.linear', **kwargs
) -> np.ndarray:
    """
    Perform black tophat operation on a volume.
    This operation is defined as bothat(x)=close(x)-x.
    If method is either pygorpho.linear or pygorpho.flat, the close methods from [Zonohedral Approximation of Spherical Structuring Element for
    Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf) are used.
    These methods require a GPU, and we therefore recommend using the [scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) (scipy.ndimage) if no GPU is available on your current device.

    Args:
        vol (np.ndarray): The volume to perform the black tophat on.
        kernel (int or np.ndarray): The structuring element/kernel to use while performing opening. Note that the kernel should be 3D unless if the linear method is used. If this method is used, a kernel resembling a ball will be created with an integer radius.
        method (str, optional): Determines the method for black tophat. Use either 'pygorpho.linear', 'pygorpho.flat' or 'scipy.ndimage'. Defaults to 'pygorpho.linear'.
        **kwargs (Any): Additional keyword arguments for the used method. See the documentation for more information.

    Returns:
        bothat_vol (np.ndarray): The morphed volume.


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
            # Apply the tophat
            vol_black = qim3d.morphology.black_tophat(vol, kernel=(10,10,10), method='scipy.ndimage')

            qim3d.viz.volumetric(vol_black)
            ```
            <iframe src="https://platform.qim.dk/k3d/zonohedra_black_tophat.html" width="100%" height="500" frameborder="0"></iframe>

    """

    try:
        vol = np.asarray(vol)
    except TypeError as e:
        err = 'Input volume must be array-like.'
        raise TypeError(err) from e

    assert len(vol.shape) == 3, 'Volume must be three-dimensional.'

    if method == 'pygorpho.flat':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.bothat(vol, kernel, **kwargs)

    elif method == 'pygorpho.linear':
        assert isinstance(
            kernel, int
        ), 'Kernel is generated within function and must therefore be an integer.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(kernel)
        return pg.flat.bothat(vol, linesteps, linelens, **kwargs)

    elif method == 'scipy.ndimage':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        return ndi.black_tophat(vol, footprint=kernel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def white_tophat(
    vol: np.ndarray, kernel: int | np.ndarray, method: str = 'pygorpho.linear', **kwargs
) -> np.ndarray:
    """
    Perform white tophat operation on a volume.
    This operation is defined as tophat(x)=x-open(x).
    If method is either pygorpho.linear or pygorpho.flat, the open methods from [Zonohedral Approximation of Spherical Structuring Element for
    Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf) are used.
    These methods require a GPU, and we therefore recommend using the [scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) (scipy.ndimage) if no GPU is available on your current device.

    Args:
        vol (np.ndarray): The volume to perform the white tophat on.
        kernel (int or np.ndarray): The structuring element/kernel to use while performing opening. Note that the kernel should be 3D unless if the linear method is used. If this method is used, a kernel resembling a ball will be created with an integer radius.
        method (str, optional): Determines the method for white tophat. Use either 'pygorpho.linear', 'pygorpho.flat' or 'scipy.ndimage'. Defaults to 'pygorpho.linear'.
        **kwargs (Any): Additional keyword arguments for the used method. See the documentation for more information.

    Returns:
        tophat_vol (np.ndarray): The morphed volume.


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
            # Apply tophat
            vol_white = qim3d.morphology.white_tophat(vol, kernel=(10,10,10), method='scipy.ndimage')

            qim3d.viz.volumetric(vol_white)
            ```
            <iframe src="https://platform.qim.dk/k3d/zonohedra_white_tophat.html" width="100%" height="500" frameborder="0"></iframe>

    """

    try:
        vol = np.asarray(vol)
    except TypeError as e:
        err = 'Input volume must be array-like.'
        raise TypeError(err) from e

    assert len(vol.shape) == 3, 'Volume must be three-dimensional.'

    if method == 'pygorpho.flat':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        return pg.flat.tophat(vol, kernel, **kwargs)

    elif method == 'pygorpho.linear':
        assert isinstance(
            kernel, int
        ), 'Kernel is generated within function and must therefore be an integer.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.ndimage.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(kernel)
        return pg.flat.tophat(vol, linesteps, linelens, **kwargs)

    elif method == 'scipy.ndimage':
        kernel = _create_kernel(kernel)
        assert kernel.ndim == 3, 'Kernel must a 3D np.ndarray.'

        return ndi.white_tophat(vol, footprint=kernel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)
