"""
Volumetric visualization using K3D

!!! quote "Reference"
    Volumetric visualization uses K3D:
    [Github page](https://github.com/K3D-tools/K3D-jupyter)

"""

import k3d
import numpy as np


def vol(img, aspectmode="data", show=True, save=False, grid_visible=False, cmap=None, **kwargs):
    """
    Visualizes a 3D volume using volumetric rendering.

    Args:
        img (numpy.ndarray): The input 3D image data. It should be a 3D numpy array.
        aspectmode (str, optional): Determines the proportions of the scene's axes. Defaults to "data".

            If `'data'`, the axes are drawn in proportion with the axes' ranges.
            If `'cube'`, the axes are drawn as a cube, regardless of the axes' ranges.
        show (bool, optional): If True, displays the visualization inline. Defaults to True.
        save (bool or str, optional): If True, saves the visualization as an HTML file.
            If a string is provided, it's interpreted as the file path where the HTML
            file will be saved. Defaults to False.
        grid_visible (bool, optional): If True, the grid is visible in the plot. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the `k3d.plot` function.

    Returns:
        plot (k3d.plot): If `show=False`, returns the K3D plot object.

    Raises:
        ValueError: If `aspectmode` is not `'data'` or `'cube'`.

    Example:
        Display a volume inline:

        ```python
        import qim3d

        vol = qim3d.examples.bone_128x128x128
        qim3d.viz.vol(vol) 
        ```
        <iframe src="https://platform.qim.dk/k3d/fima-bone_128x128x128-20240221113459.html" width="100%" height="500" frameborder="0"></iframe>

        Save a plot to an HTML file:

        ```python
        import qim3d
        vol = qim3d.examples.bone_128x128x128
        plot = qim3d.viz.vol(vol, show=False, save="plot.html")
        ```
        
    """

    if aspectmode.lower() not in ["data", "cube"]:
        raise ValueError("aspectmode should be either 'data' or 'cube'")

    plt_volume = k3d.volume(
        img.astype(np.float32),
        bounds=(
            [0, img.shape[0], 0, img.shape[1], 0, img.shape[2]]
            if aspectmode.lower() == "data"
            else None
        ),
        color_map=cmap,
    )
    plot = k3d.plot(grid_visible=grid_visible, **kwargs)
    plot += plt_volume

    if save:
        # Save html to disk
        with open(str(save), "w", encoding="utf-8") as fp:
            fp.write(plot.get_snapshot())

    if show:
        plot.display()
    else:
        return plot
