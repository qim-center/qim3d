import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

import qim3d


def circles(
    blobs: tuple[float, float, float, float],
    volume: np.ndarray,
    alpha: float = 0.5,
    color: str = '#ff9900',
    **kwargs,
) -> widgets.interactive:
    """
    Visualizes detected blobs as circles overlaid on the volume slices.

    This function is primarily used to verify the results of blob detection algorithms. It takes a list of detected features (defined by their center coordinates and radius) and projects them onto the 2D slices of the volume. As you scroll through the stack, the circles dynamically resize to represent the cross-section of the 3D spherical blobs at that specific depth, providing an intuitive check for detection accuracy.

    Args:
        blobs (np.ndarray): A list or array of detected blobs. Each blob is expected to be a 4-tuple or array `(z, y, x, radius)`. This is typically the output from `qim3d.detection.blobs`.
        volume (np.ndarray): The 3D volume (image stack) on which the blobs were detected.
        alpha (float, optional): The transparency level of the filled circles (0.0 to 1.0). Defaults to 0.5.
        color (str, optional): The color of the circles, capable of accepting hex codes or standard color names. Defaults to "#ff9900" (orange).
        **kwargs (Any): Additional keyword arguments passed to the underlying `qim3d.viz.slices_grid` function (e.g., `vmin`, `vmax`).

    Returns:
        slicer_obj (widgets.interactive):
            An interactive widget with a slider to navigate through slices, showing the overlay of detected blobs.

    Example:
        ```python
        import qim3d
        import qim3d.detection

        # Get data
        vol = qim3d.examples.cement_128x128x128

        # Detect blobs, and get binary mask
        blobs, _ = qim3d.detection.blobs(
            vol,
            min_sigma=1,
            max_sigma=8,
            threshold=0.001,
            overlap=0.1,
            background="bright"
            )

        # Visualize detected blobs with circles method
        qim3d.viz.circles(blobs, vol, alpha=0.8, color='blue')
        ```
        ![blob detection](../../assets/screenshots/blob_detection.gif)

    """

    def _slicer(z_slice):
        clear_output(wait=True)
        fig = qim3d.viz.slices_grid(
            volume[z_slice : z_slice + 1],
            n_slices=1,
            colormap='gray',
            display_figure=False,
            display_positions=False,
            **kwargs,
        )
        # Add circles from deteced blobs
        for detected in blobs:
            z, y, x, s = detected
            if abs(z - z_slice) < s:  # The blob is in the slice
                # Adjust the radius based on the distance from the center of the sphere
                distance_from_center = abs(z - z_slice)
                angle = (
                    np.pi / 2 * (distance_from_center / s)
                )  # Angle varies from 0 at the center to pi/2 at the edge
                adjusted_radius = s * np.cos(angle)  # Radius follows a cosine curve

                if adjusted_radius > 0.5:
                    c = plt.Circle(
                        (x, y),
                        adjusted_radius,
                        color=color,
                        linewidth=0,
                        fill=True,
                        alpha=alpha,
                    )
                    fig.get_axes()[0].add_patch(c)

        display(fig)
        return fig

    position_slider = widgets.IntSlider(
        value=volume.shape[0] // 2,
        min=0,
        max=volume.shape[0] - 1,
        description='Slice',
        continuous_update=True,
    )
    slicer_obj = widgets.interactive(_slicer, z_slice=position_slider)
    slicer_obj.layout = widgets.Layout(align_items='flex-start')

    return slicer_obj
