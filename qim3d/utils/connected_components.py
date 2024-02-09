import numpy as np
import torch
from scipy.ndimage import find_objects, label


class ConnectedComponents:
    def __init__(self, connected_components, num_connected_components):
        """
        Initializes a ConnectedComponents object.

        Args:
            connected_components (np.ndarray): The connected components.
            num_connected_components (int): The number of connected components.
        """
        self._connected_components = connected_components
        self._num_connected_components = num_connected_components

    @property
    def connected_components(self):
        """
        Get the connected components.

        Returns:
            np.ndarray: The connected components.
        """
        return self._connected_components

    @property
    def num_connected_components(self):
        """
        Get the number of connected components.

        Returns:
            int: The number of connected components.
        """
        return self._num_connected_components

    def get_connected_component(self, index=None, crop=False):
        """
        Get the connected component with the given index, if index is None selects a random component.

        Args:
            index (int): The index of the connected component. If none selects a random component.
            crop (bool): If True, the volume is cropped to the bounding box of the connected component.

        Returns:
            np.ndarray: The connected component as a binary mask.
        """
        if index is None:
            volume =  self.connected_components == np.random.randint(
                1, self.num_connected_components + 1
            )
        else:
            assert 1 <= index <= self.num_connected_components, "Index out of range."
            volume = self.connected_components == index
            
        if crop:
            # As we index get_bounding_box element 0 will be the bounding box for the connected component at index
            bbox = self.get_bounding_box(index)[0] 
            volume = volume[bbox]
        
        return volume

    def get_bounding_box(self, index=None):
        """Get the bounding boxes of the connected components.

        Args:
            index (int, optional): The index of the connected component. If none selects all components.

        Returns:
            list: A list of bounding boxes.
        """

        if index:
            assert 1 <= index <= self.num_connected_components, "Index out of range."
            return find_objects(self.connected_components == index)
        else:
            return find_objects(self.connected_components)


def get_3d_connected_components(image: np.ndarray | torch.Tensor):
    """Get the connected components of a 3D binary image.

    Args:
        image (np.ndarray | torch.Tensor): An array-like object to be labeled. Any non-zero values in `input` are
            counted as features and zero values are considered the background.

    Returns:
        class: Returns class object of the connected components.
    """
    if image.ndim != 3:
        raise ValueError(
            f"Given array is not a volume! Current dimension: {image.ndim}"
        )

    connected_components, num_connected_components = label(image)
    return ConnectedComponents(connected_components, num_connected_components)
