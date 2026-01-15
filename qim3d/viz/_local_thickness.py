from typing import Optional, Tuple, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from qim3d.utils._logger import log


def local_thickness(
    image: np.ndarray,
    image_lt: np.ndarray,
    max_projection: bool = False,
    axis: int = 0,
    slice_index: Optional[Union[int, float]] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (15, 5),
) -> Union[plt.Figure, widgets.interactive]:
    """
    Visualizes a local thickness map alongside the original image and a statistics histogram.

    This function provides a comprehensive view of structure width or pore size distribution. It displays a side-by-side comparison of the original data and the computed local thickness (heat map), where color intensity represents the diameter of the largest sphere that fits inside the structure at that point. It also includes a histogram to quantify the distribution of thickness values.

    For 3D volumes, the output can be either an interactive slice viewer or a static Maximum Intensity Projection (MIP).

    Args:
        image (np.ndarray): The original 2D or 3D input data (binary or grayscale).
        image_lt (np.ndarray): The computed local thickness map (must have the same shape as `image`). This is typically the output of `qim3d.processing.local_thickness`.
        max_projection (bool, optional): If `True` (and input is 3D), collapses the volume along the specified axis using maximum projection before plotting. Results in a static 2D figure. Defaults to `False`.
        axis (int, optional): The axis along which to slice or project the volume. Defaults to 0.
        slice_index (int or float, optional): The initial slice to display for 3D volumes.
            
            * **int**: The exact index of the slice.
            * **float**: A fraction between 0.0 and 1.0 (e.g., 0.5 for the middle).
            * **None**: Defaults to the middle slice.
        
        show (bool, optional): If `True`, explicitly calls `plt.show()` to render the plot immediately. Defaults to `False`.
        figsize (tuple[int, int], optional): The width and height of the figure in inches. Defaults to (15, 5).

    Returns:
        object (widgets.interactive or matplotlib.figure.Figure):
            The visualization object, depending on the input and parameters:

            * **widgets.interactive**: Returned if the input is 3D and `max_projection=False`. Contains a slider for slice navigation.
            * **matplotlib.figure.Figure**: Returned if the input is 2D or if `max_projection=True`.

    Raises:
        ValueError: If `slice_index` is a float outside the range [0, 1].

    Example:
        ```python
        import qim3d

        fly = qim3d.examples.fly_150x256x256
        lt_fly = qim3d.processing.local_thickness(fly)
        qim3d.viz.local_thickness(fly, lt_fly, axis=0)
        ```
        ![local thickness 3d](../../assets/screenshots/local_thickness_3d.gif)
    """

    def _local_thickness(image, image_lt, show, figsize, axis=None, slice_index=None):
        if slice_index is not None:
            image = image.take(slice_index, axis=axis)
            image_lt = image_lt.take(slice_index, axis=axis)

        fig, axs = plt.subplots(1, 3, figsize=figsize, layout='constrained')

        axs[0].imshow(image, cmap='gray')
        axs[0].set_title('Original image')
        axs[0].axis('off')

        axs[1].imshow(image_lt, cmap='viridis')
        axs[1].set_title('Local thickness')
        axs[1].axis('off')

        plt.colorbar(
            axs[1].imshow(image_lt, cmap='viridis'), ax=axs[1], orientation='vertical'
        )

        axs[2].hist(image_lt[image_lt > 0].ravel(), bins=32, edgecolor='black')
        axs[2].set_title('Local thickness histogram')
        axs[2].set_xlabel('Local thickness')
        axs[2].set_ylabel('Count')

        if show:
            plt.show()

        plt.close()

        return fig

    # Get the middle slice if the input is 3D
    if len(image.shape) == 3:
        if max_projection:
            if slice_index is not None:
                log.warning(
                    'slice_index is not used for max_projection. It will be ignored.'
                )
            image = image.max(axis=axis)
            image_lt = image_lt.max(axis=axis)
            return _local_thickness(image, image_lt, show, figsize)
        else:
            if slice_index is None:
                slice_index = image.shape[axis] // 2
            elif isinstance(slice_index, float):
                if slice_index < 0 or slice_index > 1:
                    raise ValueError(
                        'Values of slice_index of float type must be between 0 and 1.'
                    )
                slice_index = int(slice_index * image.shape[0]) - 1
            slice_index_slider = widgets.IntSlider(
                min=0,
                max=image.shape[axis] - 1,
                step=1,
                value=slice_index,
                description='Slice index',
                layout=widgets.Layout(width='450px'),
            )
            widget_obj = widgets.interactive(
                _local_thickness,
                image=widgets.fixed(image),
                image_lt=widgets.fixed(image_lt),
                show=widgets.fixed(True),
                figsize=widgets.fixed(figsize),
                axis=widgets.fixed(axis),
                slice_index=slice_index_slider,
            )
            widget_obj.layout = widgets.Layout(align_items='center')
            if show:
                display(widget_obj)
            return widget_obj
    else:
        if max_projection:
            log.warning(
                'max_projection is only used for 3D images. It will be ignored.'
            )
        if slice_index is not None:
            log.warning('slice_index is only used for 3D images. It will be ignored.')
        return _local_thickness(image, image_lt, show, figsize)
