import matplotlib.pyplot as plt
import numpy as np

from qim3d.io.logger import log
from qim3d.processing.cc import CC
from qim3d.viz.colormaps import objects as qim3dCmap
import qim3d

def plot_cc(
    connected_components,
    component_indexs: list | tuple = None,
    max_cc_to_plot=32,
    overlay=None,
    crop=False,
    show=True,
    **kwargs,
) -> list[plt.Figure]:
    """
    Plots the connected components from a `qim3d.processing.cc.CC` object. If an overlay image is provided, the connected component will be masked to the overlay image.

    Parameters:
        connected_components (CC): The connected components object.
        component_indexs (list | tuple, optional): The components to plot. If None the first max_cc_to_plot=32 components will be plotted. Defaults to None.
        max_cc_to_plot (int, optional): The maximum number of connected components to plot. Defaults to 32.
        overlay (optional): Overlay image. Defaults to None.
        crop (bool, optional): Whether to crop the image to the cc. Defaults to False.
        show (bool, optional): Whether to show the figure. Defaults to True.
        **kwargs: Additional keyword arguments to pass to `qim3d.viz.slices`.

    Returns:
        figs (list[plt.Figure]): List of figures, if `show=False`.

    Example:
        ```python
        import qim3d
        vol = qim3d.examples.cement_128x128x128[50:150]
        vol_bin = vol<80
        cc = qim3d.processing.get_3d_cc(vol_bin)
        qim3d.viz.plot_cc(cc, crop=True, show=True, overlay=None, n_slices=5, component_indexs=[4,6,7])
        qim3d.viz.plot_cc(cc, crop=True, show=True, overlay=vol, n_slices=5, component_indexs=[4,6,7])
        ```
        ![plot_cc_no_overlay](assets/screenshots/plot_cc_no_overlay.png)
        ![plot_cc_overlay](assets/screenshots/plot_cc_overlay.png)
    """
    # if no components are given, plot the first max_cc_to_plot=32 components
    if component_indexs is None:
        if len(connected_components) > max_cc_to_plot:
            log.warning(
                f"More than {max_cc_to_plot} connected components found. Only the first {max_cc_to_plot} will be plotted. Change max_cc_to_plot to plot more components."
            )
        component_indexs = range(
            1, min(max_cc_to_plot + 1, len(connected_components) + 1)
        )
        
    figs = []
    for component in component_indexs:
        if overlay is not None:
            assert (overlay.shape == connected_components.shape), f"Overlay image must have the same shape as the connected components. overlay.shape=={overlay.shape} != connected_components.shape={connected_components.shape}."

            # plots overlay masked to connected component
            if crop:
                # Crop the overlay image based on the bounding box of the component
                bb = connected_components.get_bounding_box(component)[0]
                cc = connected_components.get_cc(component, crop=True)
                overlay_crop = overlay[bb]
                # use cc as mask for overlay_crop, where all values in cc set to 0 should be masked out, cc contains integers
                overlay_crop = np.where(cc == 0, 0, overlay_crop)
                fig = qim3d.viz.slices(overlay_crop, show=show, **kwargs)
            else:
                cc = connected_components.get_cc(component, crop=False)
                overlay_crop = np.where(cc == 0, 0, overlay)
                fig = qim3d.viz.slices(overlay_crop, show=show, **kwargs)
        else:
            # assigns discrete color map to each connected component if not given 
            if "cmap" not in kwargs:
                kwargs["cmap"] = qim3dCmap(len(component_indexs))
        
            # Plot the connected component without overlay
            fig = qim3d.viz.slices(connected_components.get_cc(component, crop=crop), show=show, **kwargs)

        figs.append(fig)

    if not show:
        return figs

    return