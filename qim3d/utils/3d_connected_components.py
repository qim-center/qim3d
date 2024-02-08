import numpy as np
from scipy.ndimage import label


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
        assert 1 <= index <= self._num_connected_components, "Index out of range."

        if index:
            return self._connected_components == index
        else:
            return self._connected_components == np.random.randint(1, self._num_connected_components + 1)


def get_3d_connected_components(image, connectivity=1):
    """Get the connected components of a 3D binary image.

    Args:
        image (np.ndarray): The 3D binary image.
        connectivity (int, optional): The connectivity of the connected components. Defaults to 1.

    Returns:
        class: Returns class object of the connected components.
    """
    connected_components, num_connected_components = label(image, connectivity)
    return ConnectedComponents(connected_components, num_connected_components)
