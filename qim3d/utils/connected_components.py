import numpy as np
from scipy.ndimage import label

# TODO: implement find_objects and get_bounding_boxes methods

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

    def get_connected_component(self, index=None):
        """ 
        Get the connected component with the given index, if index is None selects a random component.

        Args:
            index (int): The index of the connected component. If none selects a random component.

        Returns:
            np.ndarray: The connected component as a binary mask.
        """
        if index is None:
            return self.connected_components == np.random.randint(1, self.num_connected_components + 1)
        else:
            assert 1 <= index <= self.num_connected_components, "Index out of range."
            return self.connected_components == index


def get_3d_connected_components(image):
    """Get the connected components of a 3D binary image.

    Args:
        image (np.ndarray): The 3D binary image.
        connectivity (int, optional): The connectivity of the connected components. Defaults to 1.

    Returns:
        class: Returns class object of the connected components.
    """
    connected_components, num_connected_components = label(image)
    return ConnectedComponents(connected_components, num_connected_components)
