import numpy as np
import pygorpho as pg
import scipy.ndimage as ndi


def dilate(
    vol: np.ndarray, strel: int | np.ndarray, method: str = 'linear', **kwargs
) -> np.ndarray:
    """
    Dilate an image. If method is either linear or flat, the dilation methods from [Zonohedral Approximation of Spherical Structuring Element for Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf) are used. These methods require a GPU, and we therefore recommend using the
    [scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) if no GPU is available on your current device.

    Args:
        vol (np.ndarray): The volume to dilate.
        strel (int or np.ndarray): The structuring element to use while performing dilation. Note that the structuring element should be 3D unless if the linear method is used. If this method is used, a structuring element resembling a ball will be created with an integer radius.
        method (str, optional): Determines the method for dilation.
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
        qim3d.viz.volumetric(vol, grid_visible=True)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_original.html" width="100%" height="500" frameborder="0"></iframe>

        ```python
        # Pad volume to ensure dilation does not surpass boundaries
        p = 20
        vol_padded = qim3d.operations.pad(vol, x_axis=p, y_axis=p, z_axis=p)

        # Create structuring element and apply dilation
        s = 8
        strel = np.ones((s,s,s))
        vol_dilated = qim3d.morphology.dilate(vol_padded, strel, method='ndi')

        # Trim the padded slices
        vol_trimmed = qim3d.operations.trim(vol_dilated)

        # Pad it back to original size
        vol_final = qim3d.operations.pad_to(vol_trimmed, vol.shape)

        # Visualize
        qim3d.viz.volumetric(vol_final)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_dilated.html" width="100%" height="500" frameborder="0"></iframe>

    """

    if method == 'pg.flat' or method == 'pygorpho.flat' or method == 'flat':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.'
            raise RuntimeError(err)

        return pg.flat.dilate(vol, strel, **kwargs)

    elif method == 'pg.linear' or method == 'linear':
        assert isinstance(
            strel, int
        ), 'Structuring element is generated within function and must therefore be an integer.'

        linesteps, linelens = pg.strel.flat_ball_approx(strel)

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.'
            raise RuntimeError(err)

        return pg.flat.linear_dilate(vol, linesteps, linelens)

    elif method == 'ndi' or method == 'scipy' or method == 'ndimage':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        return ndi.grey_dilation(vol, footprint=strel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def erode(
    vol: np.ndarray, strel: int | np.ndarray, method: str = 'linear', **kwargs
) -> np.ndarray:
    """
    Erode an image. If method is either linear or flat, the erosion methods from [Zonohedral Approximation of Spherical Structuring Element for Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf) are used. These methods require a GPU, and we therefore recommend using the [scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) if no GPU is available on your current device.

    Args:
            vol (np.ndarray): The volume to erode.
            strel (int or np.ndarray): The structuring element to use while performing erosion. Note that the structuring element should be 3D unless if the linear method is used. If this method is used, a structuring element resembling a ball will be created with an integer radius.
            method (str, optional): Determines the method for erosion.
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
        qim3d.viz.volumetric(vol, grid_visible=True)
    ```
    <iframe src="https://platform.qim.dk/k3d/zonohedra_original.html" width="100%" height="500" frameborder="0"></iframe>
    ```python
        # Create structuring element and erode
        s = 6
        strel = np.ones((s,s,s))
        vol_eroded = qim3d.morphology.erode(vol, strel, method='ndi')

        # Visualize
        qim3d.viz.volumetric(vol_eroded)
    ```
    <iframe src="https://platform.qim.dk/k3d/zonohedra_eroded.html" width="100%" height="500" frameborder="0"></iframe>

    """

    if method == 'pg.flat' or method == 'pygorpho.flat' or method == 'flat':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.'
            raise RuntimeError(err)

        return pg.flat.erode(vol, strel, **kwargs)

    elif method == 'pg.linear' or method == 'linear':
        assert isinstance(
            strel, int
        ), 'Structuring element is generated within function and must therefore be an integer.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(strel)
        return pg.flat.linear_erode(vol, linesteps, linelens, **kwargs)

    elif method == 'ndi' or method == 'scipy' or method == 'ndimage':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        return ndi.grey_erosion(vol, footprint=strel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def opening(
    vol: np.ndarray, strel: int | np.ndarray, method: str = 'linear', **kwargs
) -> np.ndarray:
    """
    Morphologically open a volume.
    If method is either linear or flat, the open methods from [Zonohedral Approximation of Spherical Structuring Element for
    Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf) are used.
    These methods require a GPU, and we therefore recommend using the [scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) if no GPU is available on your current device.

    Args:
        vol (np.ndarray): The volume to open.
        strel (int or np.ndarray): The structuring element to use while performing erosion. Note that the structuring element should be 3D unless if the linear method is used. If this method is used, a structuring element resembling a ball will be created with an integer radius.
        method (str, optional): Determines the method for erosion.
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
        # Pad volume to ensure dilation does not surpass boundaries
        p = 20
        vol_padded = qim3d.operations.pad(vol_noised, x_axis=p, y_axis=p, z_axis=p)

        # Create structuring element and apply opening
        s = 6
        strel = np.ones((s,s,s))
        vol_opened = qim3d.morphology.opening(vol_padded, strel, method='ndi')

        # Trim the padded slices
        vol_trimmed = qim3d.operations.trim(vol_opened)

        # Pad it back to original size
        vol_final = qim3d.operations.pad_to(vol_trimmed, vol.shape)

        # Visualize
        qim3d.viz.volumetric(vol_final)
        ```

        <iframe src="https://platform.qim.dk/k3d/zonohedra_opening.html" width="100%" height="500" frameborder="0"></iframe>

    """

    if method == 'pg.flat' or method == 'pygorpho.flat' or method == 'flat':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.'
            raise RuntimeError(err)

        return pg.flat.open(vol, strel, **kwargs)

    elif method == 'pg.linear' or method == 'linear':
        assert isinstance(
            strel, int
        ), 'Structuring element is generated within function and must therefore be an integer.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(strel)
        return pg.flat.linear_open(vol, linesteps, linelens, **kwargs)

    elif method == 'ndi' or method == 'scipy' or method == 'ndimage':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        return ndi.grey_opening(vol, footprint=strel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def closing(
    vol: np.ndarray, strel: int | np.ndarray, method: str = 'linear', **kwargs
) -> np.ndarray:
    """
    Morphologically close a volume.
    If method is either linear or flat, the close methods from [Zonohedral Approximation of Spherical Structuring Element for
    Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf) are used.
    These methods require a GPU, and we therefore recommend using the [scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) if no GPU is available on your current device.

    Args:
        vol (np.ndarray): The volume to be closed.
        strel (int or np.ndarray): The structuring element to use while performing opening. Note that the structuring element should be 3D unless if the linear method is used. If this method is used, a structuring element resembling a ball will be created with an integer radius.
        method (str, optional): Determines the method for closing.
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
        qim3d.viz.volumetric(cube, grid_visible=True)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_cube.html" width="100%" height="500" frameborder="0"></iframe>
        ```python
        # Generate structuring element and apply closing
        s = 15
        strel = np.ones((s,s,s))
        cube_closed = qim3d.morphology.closing(cube, strel, method='ndi')

        # Visualize
        qim3d.viz.volumetric(cube_closed)
        ```
        <iframe src="https://platform.qim.dk/k3d/zonohedra_cube_closed.html" width="100%" height="500" frameborder="0"></iframe>

    """

    if method == 'pg.flat' or method == 'pygorpho.flat' or method == 'flat':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.'
            raise RuntimeError(err)

        return pg.flat.close(vol, strel, **kwargs)

    elif method == 'pg.linear' or method == 'linear':
        assert isinstance(
            strel, int
        ), 'Structuring element is generated within function and must therefore be an integer.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(strel)
        return pg.flat.linear_close(vol, linesteps, linelens, **kwargs)

    elif method == 'ndi' or method == 'scipy' or method == 'ndimage':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        return ndi.grey_closing(vol, footprint=strel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def black_tophat(
    vol: np.ndarray, strel: int | np.ndarray, method: str = 'linear', **kwargs
) -> np.ndarray:
    """
    Perform black tophat operation on a volume.
    This operation is defined as bothat(x)=close(x)-x.
    If method is either linear or flat, the close methods from [Zonohedral Approximation of Spherical Structuring Element for
    Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf) are used.
    These methods require a GPU, and we therefore recommend using the [scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) if no GPU is available on your current device.

    Args:
        vol (np.ndarray): The volume to perform the black tophat on.
        strel (int or np.ndarray): The structuring element to use while performing opening. Note that the structuring element should be 3D unless if the linear method is used. If this method is used, a structuring element resembling a ball will be created with an integer radius.
        method (str, optional): Determines the method for black tophat.
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
            qim3d.viz.volumetric(vol, grid_visible=True)
            ```
            <iframe src="https://platform.qim.dk/k3d/zonohedra_original.html" width="100%" height="500" frameborder="0"></iframe>
            ```python
            # Create structuring element and apply the tophat
            s = 10
            strel = np.ones((s,s,s))
            vol_black = qim3d.morphology.black_tophat(vol, strel, method='ndi')

            qim3d.viz.volumetric(vol_black)
            ```
            <iframe src="https://platform.qim.dk/k3d/zonohedra_black_tophat.html" width="100%" height="500" frameborder="0"></iframe>

    """

    if method == 'pg.flat' or method == 'pygorpho.flat' or method == 'flat':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.'
            raise RuntimeError(err)

        return pg.flat.bothat(vol, strel, **kwargs)

    elif method == 'pg.linear' or method == 'linear':
        assert isinstance(
            strel, int
        ), 'Structuring element is generated within function and must therefore be an integer.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(strel)
        return pg.flat.bothat(vol, linesteps, linelens, **kwargs)

    elif method == 'ndi' or method == 'scipy' or method == 'ndimage':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        return ndi.black_tophat(vol, footprint=strel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)


def white_tophat(
    vol: np.ndarray, strel: int | np.ndarray, method: str = 'linear', **kwargs
) -> np.ndarray:
    """
    Perform white tophat operation on a volume.
    This operation is defined as tophat(x)=x-open(x).
    If method is either linear or flat, the open methods from [Zonohedral Approximation of Spherical Structuring Element for
    Volumetric Morphology](https://backend.orbit.dtu.dk/ws/portalfiles/portal/172879029/SCIA19_Zonohedra.pdf) are used.
    These methods require a GPU, and we therefore recommend using the [scipy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) if no GPU is available on your current device.

    Args:
        vol (np.ndarray): The volume to perform the white tophat on.
        strel (int or np.ndarray): The structuring element to use while performing opening. Note that the structuring element should be 3D unless if the linear method is used. If this method is used, a structuring element resembling a ball will be created with an integer radius.
        method (str, optional): Determines the method for white tophat.
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
            qim3d.viz.volumetric(vol, grid_visible=True)
            ```
            <iframe src="https://platform.qim.dk/k3d/zonohedra_original.html" width="100%" height="500" frameborder="0"></iframe>

            ```python
            # Generate structuring element and apply tophat
            s = 10
            strel = np.ones((s,s,s))

            vol_white = qim3d.morphology.white_tophat(vol, strel, method='ndi')

            qim3d.viz.volumetric(vol_white)
            ```
            <iframe src="https://platform.qim.dk/k3d/zonohedra_white_tophat.html" width="100%" height="500" frameborder="0"></iframe>

    """

    if method == 'pg.flat' or method == 'pygorpho.flat' or method == 'flat':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.'
            raise RuntimeError(err)

        return pg.flat.tophat(vol, strel, **kwargs)

    elif method == 'pg.linear' or method == 'linear':
        assert isinstance(
            strel, int
        ), 'Structuring element is generated within function and must therefore be an integer.'

        if not pg.cuda.get_device_count():
            err = 'no CUDA device available. Use method=scipy.'
            raise RuntimeError(err)

        linesteps, linelens = pg.strel.flat_ball_approx(strel)
        return pg.flat.tophat(vol, linesteps, linelens, **kwargs)

    elif method == 'ndi' or method == 'scipy' or method == 'ndimage':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        return ndi.white_tophat(vol, footprint=strel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)
