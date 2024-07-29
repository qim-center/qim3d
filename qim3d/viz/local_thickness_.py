from qim3d.utils.logger import log
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple
import ipywidgets as widgets

def local_thickness(
    image: np.ndarray,
    image_lt: np.ndarray,
    max_projection: bool = False,
    axis: int = 0,
    slice_idx: Optional[Union[int, float]] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (15, 5),
) -> Union[plt.Figure, widgets.interactive]:
    """Visualizes the local thickness of a 2D or 3D image.

    Args:
        image (np.ndarray): 2D or 3D NumPy array representing the image/volume.
        image_lt (np.ndarray): 2D or 3D NumPy array representing the local thickness of the input
            image/volume.
        max_projection (bool, optional): If True, displays the maximum projection of the local
            thickness. Only used for 3D images. Defaults to False.
        axis (int, optional): The axis along which to visualize the local thickness.
            Unused for 2D images.
            Defaults to 0.
        slice_idx (int or float, optional): The initial slice to be visualized. The slice index
            can afterwards be changed. If value is an integer, it will be the index of the slice
            to be visualized. If value is a float between 0 and 1, it will be multiplied by the
            number of slices and rounded to the nearest integer. If None, the middle slice will
            be used for 3D images. Unused for 2D images. Defaults to None.
        show (bool, optional): If True, displays the plot (i.e. calls plt.show()). Defaults to False.
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (15, 5).

    Raises:
        ValueError: If the slice index is not an integer or a float between 0 and 1.

    Returns:
        If the input is 3D, returns an interactive widget. Otherwise, returns a matplotlib figure.

    Example:
        ```python
        import qim3d

        fly = qim3d.examples.fly_150x256x256 # 3D volume
        lt_fly = qim3d.processing.local_thickness(fly)
        qim3d.viz.local_thickness(fly, lt_fly, axis=0)
        ```
        ![local thickness 3d](assets/screenshots/local_thickness_3d.gif)

        
    """

    def _local_thickness(image, image_lt, show, figsize, axis=None, slice_idx=None):
        if slice_idx is not None:
            image = image.take(slice_idx, axis=axis)
            image_lt = image_lt.take(slice_idx, axis=axis)

        fig, axs = plt.subplots(1, 3, figsize=figsize, layout="constrained")

        axs[0].imshow(image, cmap="gray")
        axs[0].set_title("Original image")
        axs[0].axis("off")

        axs[1].imshow(image_lt, cmap="viridis")
        axs[1].set_title("Local thickness")
        axs[1].axis("off")

        plt.colorbar(
            axs[1].imshow(image_lt, cmap="viridis"), ax=axs[1], orientation="vertical"
        )

        axs[2].hist(image_lt[image_lt > 0].ravel(), bins=32, edgecolor="black")
        axs[2].set_title("Local thickness histogram")
        axs[2].set_xlabel("Local thickness")
        axs[2].set_ylabel("Count")

        if show:
            plt.show()

        plt.close()

        return fig

    # Get the middle slice if the input is 3D
    if len(image.shape) == 3:
        if max_projection:
            if slice_idx is not None:
                log.warning(
                    "slice_idx is not used for max_projection. It will be ignored."
                )
            image = image.max(axis=axis)
            image_lt = image_lt.max(axis=axis)
            return _local_thickness(image, image_lt, show, figsize)
        else:
            if slice_idx is None:
                slice_idx = image.shape[axis] // 2
            elif isinstance(slice_idx, float):
                if slice_idx < 0 or slice_idx > 1:
                    raise ValueError(
                        "Values of slice_idx of float type must be between 0 and 1."
                    )
                slice_idx = int(slice_idx * image.shape[0]) - 1
            slide_idx_slider = widgets.IntSlider(
                min=0,
                max=image.shape[axis] - 1,
                step=1,
                value=slice_idx,
                description="Slice index",
                layout=widgets.Layout(width="450px"),
            )
            widget_obj = widgets.interactive(
                _local_thickness,
                image=widgets.fixed(image),
                image_lt=widgets.fixed(image_lt),
                show=widgets.fixed(True),
                figsize=widgets.fixed(figsize),
                axis=widgets.fixed(axis),
                slice_idx=slide_idx_slider,
            )
            widget_obj.layout = widgets.Layout(align_items="center")
            if show:
                display(widget_obj)
            return widget_obj
    else:
        if max_projection:
            log.warning(
                "max_projection is only used for 3D images. It will be ignored."
            )
        if slice_idx is not None:
            log.warning("slice_idx is only used for 3D images. It will be ignored.")
        return _local_thickness(image, image_lt, show, figsize)