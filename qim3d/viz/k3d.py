"""
Volumetric visualization using K3D

!!! quote "Reference"
    Volumetric visualization uses K3D:
    [Github page](https://github.com/K3D-tools/K3D-jupyter)

"""

import k3d
import numpy as np

def vol(img, show=True, save=False):
    """
    Volumetric visualization of a given volume.

    Args:
        img (numpy.ndarray): The input 3D image data. It should be a 3D numpy array.
        show (bool, optional): If True, displays the visualization. Defaults to True.
        save (bool or str, optional): If True, saves the visualization as an HTML file. 
            If a string is provided, it's interpreted as the file path where the HTML 
            file will be saved. Defaults to False.

    Returns:
        k3d.plot: If show is False, returns the K3D plot object.

    Examples:
        ```python
        import qim3d
        vol = qim3d.examples.bone_128x128x128

        # shows the volume inline
        qim3d.viz.vol(vol) 

        # saves html plot to disk
        plot = qim3d.viz.vol(vol, show=False, save="plot.html")
        ```

    """
    plt_volume = k3d.volume(img.astype(np.float32))
    plot = k3d.plot()
    plot += plt_volume

    if save:
        # Save html to disk
        with open(str(save),'w', encoding="utf-8") as fp:
            fp.write(plot.get_snapshot())

    if show:
        plot.display()
    else:
        return plot