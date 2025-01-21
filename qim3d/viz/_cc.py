import matplotlib.pyplot as plt
import numpy as np
import qim3d
from qim3d.utils._logger import log
from qim3d.segmentation._connected_components import CC

def plot_cc(
    connected_components: CC,
    component_indexs: list | tuple = None,
    max_cc_to_plot: int = 32,
    overlay: np.ndarray = None,
    crop: bool = False,
    display_figure: bool = True,
    color_map: str = "viridis",
    value_min: float = None,
    value_max: float = None,
    **kwargs,
) -> list[plt.Figure]:
    """
    Plots the connected components from a `qim3d.processing.cc.CC` object. If an overlay image is provided, the connected component will be masked to the overlay image.

    Parameters:
        connected_components (CC): The connected components object.
        component_indexs (list or tuple, optional): The components to plot. If None the first max_cc_to_plot=32 components will be plotted. Defaults to None.
        max_cc_to_plot (int, optional): The maximum number of connected components to plot. Defaults to 32.
        overlay (np.ndarray or None, optional): Overlay image. Defaults to None.
        crop (bool, optional): Whether to crop the image to the cc. Defaults to False.
        display_figure (bool, optional): Whether to show the figure. Defaults to True.
        color_map (str, optional): Specifies the color map for the image. Defaults to "viridis".
        value_min (float or None, optional): Together with vmax define the data range the colormap covers. By default colormap covers the full range. Defaults to None.
        value_max (float or None, optional): Together with vmin define the data range the colormap covers. By default colormap covers the full range. Defaults to None
        **kwargs (Any): Additional keyword arguments to pass to `qim3d.viz.slices_grid`.

    Returns:
        figs (list[plt.Figure]): List of figures, if `display_figure=False`.

    Example:
        ```python
        import qim3d
        vol = qim3d.examples.cement_128x128x128[50:150]
        vol_bin = vol<80
        cc = qim3d.segmentation.get_3d_cc(vol_bin)
        qim3d.viz.plot_cc(cc, crop=True, display_figure=True, overlay=None, num_slices=5, component_indexs=[4,6,7])
        qim3d.viz.plot_cc(cc, crop=True, display_figure=True, overlay=vol, num_slices=5, component_indexs=[4,6,7])
        ```
        ![plot_cc_no_overlay](../../assets/screenshots/plot_cc_no_overlay.png)
        ![plot_cc_overlay](../../assets/screenshots/plot_cc_overlay.png)
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
            assert (
                overlay.shape == connected_components.shape
            ), f"Overlay image must have the same shape as the connected components. overlay.shape=={overlay.shape} != connected_components.shape={connected_components.shape}."

            # plots overlay masked to connected component
            if crop:
                # Crop the overlay image based on the bounding box of the component
                bb = connected_components.get_bounding_box(component)[0]
                cc = connected_components.get_cc(component, crop=True)
                overlay_crop = overlay[bb]
                # use cc as mask for overlay_crop, where all values in cc set to 0 should be masked out, cc contains integers
                overlay_crop = np.where(cc == 0, 0, overlay_crop)
            else:
                cc = connected_components.get_cc(component, crop=False)
                overlay_crop = np.where(cc == 0, 0, overlay)
            fig = qim3d.viz.slices_grid(
                overlay_crop, display_figure=display_figure, color_map=color_map, value_min=value_min, value_max=value_max, **kwargs
            )
        else:
            # assigns discrete color map to each connected component if not given
            if "color_map" not in kwargs:
                kwargs["color_map"] = qim3d.viz.colormaps.segmentation(len(component_indexs))

            # Plot the connected component without overlay
            fig = qim3d.viz.slices_grid(
                connected_components.get_cc(component, crop=crop), display_figure=display_figure, **kwargs
            )

        figs.append(fig)

    if not display_figure:
        return figs

    return
