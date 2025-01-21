import numpy as np
from slgbuilder import GraphObject 
from slgbuilder import MaxflowBuilder

def segment_layers(data: np.ndarray, 
                   inverted: bool = False, 
                   n_layers: int = 1, 
                   delta: float = 1, 
                   min_margin: int = 10, 
                   max_margin: int = None, 
                   wrap: bool = False
                   ) -> list:
    """
    Works on 2D and 3D data.
    Light one function wrapper around slgbuilder https://github.com/Skielex/slgbuilder to do layer segmentation
    Now uses only MaxflowBuilder for solving.

    Args:
        data (np.ndarray): 2D or 3D array on which it will be computed
        inverted (bool): If True, it will invert the brightness of the image. Defaults to False
        n_layers (int): Determines amount of layers to look for (result in a layer and background). Defaults to 1.
        delta (float): Patameter determining smoothness. Defaults to 1.
        min_margin (int or None): Parameter for minimum margin. If more layers are wanted, a margin is necessary to avoid layers being identical. Defaults to None.
        max_margin (int or None): Parameter for maximum margin. If more layers are wanted, a margin is necessary to avoid layers being identical. Defaults to None.
        wrap (bool): If True, starting and ending point of the border between layers are at the same level. Defaults to False.

    Returns:
        segmentations (list[np.ndarray]): list of numpy arrays, even if n_layers == 1, each array is only 0s and 1s, 1s segmenting this specific layer

    Raises:
        TypeError: If Data is not np.array, if n_layers is not integer.
        ValueError: If n_layers is less than 1, if delta is negative or zero

    Example:
        Example is only shown on 2D image, but segment_layers can also take 3D structures.
        ```python
        import qim3d

        layers_image = qim3d.io.load('layers3d.tif')[:,:,0]
        layers = qim3d.processing.segment_layers(layers_image, n_layers = 2)
        layer_lines = qim3d.processing.get_lines(layers)

        import matplotlib.pyplot as plt

        plt.imshow(layers_image, cmap='gray')
        plt.axis('off')
        for layer_line in layer_lines:
            plt.plot(layer_line, linewidth = 3)
        ```
        ![layer_segmentation](../../assets/screenshots/layers.png)
        ![layer_segmentation](../../assets/screenshots/segmented_layers.png)

    """
    if isinstance(data, np.ndarray):
        data = data.astype(np.int32)
        if inverted:
            data = ~data
    else:
        raise TypeError(F"Data has to be type np.ndarray. Your data is of type {type(data)}")
    
    helper = MaxflowBuilder()
    if not isinstance(n_layers, int):
        raise TypeError(F"Number of layers has to be positive integer. You passed {type(n_layers)}")
    
    if n_layers == 1:
        layer = GraphObject(data)
        helper.add_object(layer)
    elif n_layers > 1:
        layers = [GraphObject(data) for _ in range(n_layers)]
        helper.add_objects(layers)
        for i in range(len(layers)-1):
            helper.add_layered_containment(layers[i], layers[i+1], min_margin=min_margin, max_margin=max_margin) 

    else:
        raise ValueError(F"Number of layers has to be positive integer. You passed {n_layers}")
    
    helper.add_layered_boundary_cost()

    if delta > 1:
        delta = int(delta)
    elif delta <= 0:
        raise ValueError(F'Delta has to be positive number. You passed {delta}')
    helper.add_layered_smoothness(delta=delta, wrap = bool(wrap))
    helper.solve()
    if n_layers == 1:
        segmentations =[helper.what_segments(layer)]
    else:
        segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]

    return segmentations

def get_lines(segmentations:list[np.ndarray]) -> list:
    """
    Expects list of arrays where each array is 2D segmentation with only 2 classes. This function gets the border between those two
    so it could be plotted. Used with qim3d.processing.segment_layers

    Args:
        segmentations (list of arrays): List of arrays where each array is 2D segmentation with only 2 classes

    Returns:
        segmentation_lines (list): List of 1D numpy arrays
    """
    segmentation_lines = [np.argmin(s, axis=0) - 0.5 for s in segmentations]
    return segmentation_lines