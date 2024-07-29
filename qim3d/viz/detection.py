import matplotlib.pyplot as plt
from qim3d.utils.logger import log
import numpy as np
import ipywidgets as widgets
from IPython.display import clear_output, display
import qim3d


def circles(blobs, vol, alpha=0.5, color="#ff9900", **kwargs):
    """
    Plots the blobs found on a slice of the volume.

    This function takes in a 3D volume and a list of blobs (detected features)
    and plots the blobs on a specified slice of the volume. If no slice is specified,
    it defaults to the middle slice of the volume.

    Args:
        blobs (array-like): An array-like object of blobs, where each blob is represented
            as a 4-tuple (p, r, c, radius). Usually the result of `qim3d.processing.blob_detection(vol)`
        vol (array-like): The 3D volume on which to plot the blobs.
        alpha (float, optional): The transparency of the blobs. Defaults to 0.5.
        color (str, optional): The color of the blobs. Defaults to "#ff9900".
        **kwargs: Arbitrary keyword arguments for the `slices` function.

    Returns:
        slicer_obj (ipywidgets.interactive): An interactive widget for visualizing the blobs.

    """

    def _slicer(z_slice):
        clear_output(wait=True)
        fig = qim3d.viz.slices(
            vol,
            n_slices=1,
            position=z_slice,
            cmap="gray",
            show_position=False,
            **kwargs
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
        value=vol.shape[0] // 2,
        min=0,
        max=vol.shape[0] - 1,
        description="Slice",
        continuous_update=True,
    )
    slicer_obj = widgets.interactive(_slicer, z_slice=position_slider)
    slicer_obj.layout = widgets.Layout(align_items="flex-start")

    return slicer_obj
