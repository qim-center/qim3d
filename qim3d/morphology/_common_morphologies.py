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

        vol = qim3d.examples.cement_128x128x128
        fig1 = qim3d.viz.slices_grid(vol, value_min=0, value_max=255, num_slices=5, display_figure=True)
        ```

    """
    if not pg.cuda.get_device_count():
        err = 'no CUDA device available. Use method=scipy.'
        raise RuntimeError(err)

    if method == 'pg.flat' or method == 'pygorpho.flat' or method == 'flat':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'
        return pg.flat.dilate(vol, strel, **kwargs)

    elif method == 'pg.linear' or method == 'linear':
        assert isinstance(
            strel, int
        ), 'Structuring element is generated within function and must therefore be an integer.'

        linesteps, linelens = pg.strel.flat_ball_approx(strel)
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

            vol = qim3d.examples.cement_128x128x128
            fig1 = qim3d.viz.slices_grid(vol, value_min=0, value_max=255, num_slices=5, display_figure=True)
            ```

    """
    if not pg.cuda.get_device_count():
        err = 'no CUDA device available. Use method=scipy.'
        raise RuntimeError(err)

    if method == 'pg.flat' or method == 'pygorpho.flat' or method == 'flat':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'
        return pg.flat.erode(vol, strel, **kwargs)

    elif method == 'pg.linear' or method == 'linear':
        assert isinstance(
            strel, int
        ), 'Structuring element is generated within function and must therefore be an integer.'

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

            vol = qim3d.examples.cement_128x128x128
            fig1 = qim3d.viz.slices_grid(vol, value_min=0, value_max=255, num_slices=5, display_figure=True)
            ```

    """
    if not pg.cuda.get_device_count():
        err = 'no CUDA device available. Use method=scipy.'
        raise RuntimeError(err)

    if method == 'pg.flat' or method == 'pygorpho.flat' or method == 'flat':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'
        return pg.flat.open(vol, strel, **kwargs)

    elif method == 'pg.linear' or method == 'linear':
        assert isinstance(
            strel, int
        ), 'Structuring element is generated within function and must therefore be an integer.'

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

            Do something here
            ```

    """
    if not pg.cuda.get_device_count():
        err = 'no CUDA device available. Use method=scipy.'
        raise RuntimeError(err)

    if method == 'pg.flat' or method == 'pygorpho.flat' or method == 'flat':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'
        return pg.flat.close(vol, strel, **kwargs)

    elif method == 'pg.linear' or method == 'linear':
        assert isinstance(
            strel, int
        ), 'Structuring element is generated within function and must therefore be an integer.'

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

            Do something here
            ```

    """
    if not pg.cuda.get_device_count():
        err = 'no CUDA device available. Use method=scipy.'
        raise RuntimeError(err)

    if method == 'pg.flat' or method == 'pygorpho.flat' or method == 'flat':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'
        return pg.flat.bothat(vol, strel, **kwargs)

    elif method == 'pg.linear' or method == 'linear':
        assert isinstance(
            strel, int
        ), 'Structuring element is generated within function and must therefore be an integer.'

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

            Do something here
            ```

    """
    if not pg.cuda.get_device_count():
        err = 'no CUDA device available. Use method=scipy.'
        raise RuntimeError(err)

    if method == 'pg.flat' or method == 'pygorpho.flat' or method == 'flat':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'
        return pg.flat.tophat(vol, strel, **kwargs)

    elif method == 'pg.linear' or method == 'linear':
        assert isinstance(
            strel, int
        ), 'Structuring element is generated within function and must therefore be an integer.'

        linesteps, linelens = pg.strel.flat_ball_approx(strel)
        return pg.flat.tophat(vol, linesteps, linelens, **kwargs)

    elif method == 'ndi' or method == 'scipy' or method == 'ndimage':
        assert not isinstance(strel, int), 'Structuring element must a 3D np.ndarray.'
        assert strel.ndim == 3, 'Structuring element must a 3D np.ndarray.'

        return ndi.white_tophat(vol, footprint=strel, **kwargs)

    else:
        err = 'Unknown closing method.'
        raise ValueError(err)
