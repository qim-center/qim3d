import numpy as np
from scipy.ndimage import find_objects, label
from qim3d.utils.logger import log


class CC:
    def __init__(self, connected_components, num_connected_components):
        """
        Initializes a ConnectedComponents object.

        Args:
            connected_components (np.ndarray): The connected components.
            num_connected_components (int): The number of connected components.
        """
        self._connected_components = connected_components
        self.cc_count = num_connected_components
        
        self.shape = connected_components.shape
    
    def __len__(self):
        """
        Returns the number of connected components in the object.
        """
        return self.cc_count

    def get_cc(self, index=None, crop=False):
        """
        Get the connected component with the given index, if index is None selects a random component.

        Args:
            index (int): The index of the connected component. 
                            If none returns all components.
                            If 'random' returns a random component.
            crop (bool): If True, the volume is cropped to the bounding box of the connected component.

        Returns:
            np.ndarray: The connected component as a binary mask.
        """
        if index is None:
            volume = self._connected_components
        elif index == "random":
            index = np.random.randint(1, self.cc_count + 1)
            volume = self._connected_components == index
        else:
            assert 1 <= index <= self.cc_count, "Index out of range. Needs to be in range [1, cc_count]."
            volume = self._connected_components == index
            
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
            assert 1 <= index <= self.cc_count, "Index out of range."
            return find_objects(self._connected_components == index)
        else:
            return find_objects(self._connected_components)


def get_3d_cc(image: np.ndarray) -> CC:
    """ Returns an object (CC) containing the connected components of the input volume. Use plot_cc to visualize the connected components.

    Args:
        image (np.ndarray): An array-like object to be labeled. Any non-zero values in `input` are
            counted as features and zero values are considered the background.

    Returns:
        CC: A ConnectedComponents object containing the connected components and the number of connected components.

    Example:
        ```python
        import qim3d
        vol = qim3d.examples.cement_128x128x128[50:150]<60
        cc = qim3d.processing.get_3d_cc(vol)
        ```
    """
    connected_components, num_connected_components = label(image)
    log.info(f"Total number of connected components found: {num_connected_components}")
    return CC(connected_components, num_connected_components)
