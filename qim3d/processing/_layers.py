import numpy as np
from slgbuilder import GraphObject, MaxflowBuilder


def segment_layers(
    data: np.ndarray,
    inverted: bool = False,
    n_layers: int = 1,
    delta: float = 1,
    min_margin: int = 10,
    max_margin: int = None,
    wrap: bool = False,
) -> list:
    """
    Segments distinct layers in 2D or 3D data using graph-based optimal surface detection.

    This function identifies continuous surfaces (layers) within the volume that minimize a specific cost function. It uses a **Graph Cut (Max-Flow/Min-Cut)** algorithm to find the global optimum. This is particularly useful for detecting boundaries in geological strata, biological tissues (e.g., retinal layers), or material interfaces.

    It acts as a wrapper around the [slgbuilder](https://github.com/Skielex/slgbuilder) library.

    Args:
        data (np.ndarray): The 2D or 3D input array. The algorithm assumes layers are stacked along the first dimension (axis 0).
        inverted (bool, optional): If `True`, inverts the intensity of the image before processing. Use this if the boundaries you are looking for are dark instead of bright (or vice-versa, depending on the cost function logic). Defaults to `False`.
        n_layers (int, optional): The number of surfaces/boundaries to detect. Defaults to 1.
        delta (float, optional): The smoothness penalty. Higher values enforce smoother (stiffer) layer boundaries, resisting sudden changes in height. Defaults to 1.
        min_margin (int or None, optional): The minimum vertical distance (in pixels/voxels) allowed between consecutive layers. Used to prevent surfaces from crossing or collapsing onto each other. Defaults to 10.
        max_margin (int or None, optional): The maximum vertical distance allowed between consecutive layers. Defaults to `None`.
        wrap (bool, optional): If `True`, enforces a periodic boundary condition where the start and end of the layer (along the width) must connect. Useful for cylindrical or unwrapped data. Defaults to `False`.

    Returns:
        segmentations (list[np.ndarray]):
            A list of binary masks (0s and 1s), where each mask represents the region defined by a detected layer. The list length equals `n_layers`.

    Raises:
        TypeError: If `data` is not a numpy array or `n_layers` is not an integer.
        ValueError: If `n_layers` is less than 1 or `delta` is non-positive.

    Example:
        ```python
        import qim3d
        import matplotlib.pyplot as plt

        # Load data (using a 2D slice for this example)
        # In this image, we want to find 2 distinct boundaries
        layers_image = qim3d.io.load('layers3d.tif')[:,:,0]

        # Segment the layers
        layers = qim3d.processing.segment_layers(layers_image, n_layers=2)

        # Extract the line coordinates for plotting
        layer_lines = qim3d.processing.get_lines(layers)

        # Visualize
        plt.imshow(layers_image, cmap='gray')
        plt.axis('off')
        for layer_line in layer_lines:
            plt.plot(layer_line, linewidth=3, label='Detected Layer')
        plt.legend()
        plt.show()
        ```
        ![layer_segmentation](../../assets/screenshots/layers.png)
        ![layer_segmentation](../../assets/screenshots/segmented_layers.png)
    """
    if isinstance(data, np.ndarray):
        data = data.astype(np.int32)
        if inverted:
            data = ~data
    else:
        raise TypeError(
            f'Data has to be type np.ndarray. Your data is of type {type(data)}'
        )

    helper = MaxflowBuilder()
    if not isinstance(n_layers, int):
        raise TypeError(
            f'Number of layers has to be positive integer. You passed {type(n_layers)}'
        )

    if n_layers == 1:
        layer = GraphObject(data)
        helper.add_object(layer)
    elif n_layers > 1:
        layers = [GraphObject(data) for _ in range(n_layers)]
        helper.add_objects(layers)
        for i in range(len(layers) - 1):
            helper.add_layered_containment(
                layers[i], layers[i + 1], min_margin=min_margin, max_margin=max_margin
            )

    else:
        raise ValueError(
            f'Number of layers has to be positive integer. You passed {n_layers}'
        )

    helper.add_layered_boundary_cost()

    if delta > 1:
        delta = int(delta)
    elif delta <= 0:
        raise ValueError(f'Delta has to be positive number. You passed {delta}')
    helper.add_layered_smoothness(delta=delta, wrap=bool(wrap))
    helper.solve()
    if n_layers == 1:
        segmentations = [helper.what_segments(layer)]
    else:
        segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]

    return segmentations


def get_lines(segmentations: list[np.ndarray]) -> list:
    """
    Extracts 1D line coordinates from 2D binary segmentation masks.

    This utility function is designed to work with the output of `qim3d.processing.segment_layers`. It converts the binary masks (which split the image into "above" and "below" a layer) into a 1D array of height indices. This allows for easy plotting of the layer boundary using Matplotlib.

    Args:
        segmentations (list[np.ndarray]): A list of 2D binary arrays, typically returned by `segment_layers`. Each array should contain two classes (0 and 1) separated by a boundary.

    Returns:
        segmentation_lines (list[np.ndarray]):
            A list of 1D arrays. Each array contains the vertical index (y-coordinate) of the layer boundary for every horizontal position (x-coordinate).

    Example:
        ```python
        # Assuming 'layers' is the output from segment_layers
        lines = qim3d.processing.get_lines(layers)

        # Plotting the first layer
        plt.plot(lines[0], color='red')
        ```
    """
    segmentation_lines = [np.argmin(s, axis=0) - 0.5 for s in segmentations]
    return segmentation_lines
