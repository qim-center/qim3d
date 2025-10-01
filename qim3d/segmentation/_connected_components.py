import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import find_objects, label, generate_binary_structure

from qim3d.utils._logger import log

class LabeledVolume:
    def __init__(self, labels: np.ndarray):
        self.labels = labels
        self.shape = labels.shape
        self._sizes = None
        self._count = self.labels.max()
    
    @property
    def sizes(self) -> np.ndarray:
        """Returns the sizes of the labels."""
        if self._sizes is None:
            self._sizes = np.bincount(self.labels.ravel())
        return self._sizes
    
    def __len__(self) -> int:
        """Number of non-background labels"""
        return self._count
    
    def filter_by_size(self, min_size: int = None, max_size: int = None) -> np.ndarray:
        """
        Extract a labels volume where only the labels with size within the chosen range are kept.

        Args:
            min_size: Lower bound of the range. Default value of None does not set a lower bound.
            max_size: Upper bound of the range. Default value of None does not set an upper bound.
        """
        keep = np.zeros(len(self) + 1, dtype=bool)
        for i, size in enumerate(self.sizes[1:], start=1):
            if (min_size is None or size >= min_size) and (max_size is None or size <= max_size):
                keep[i] = True
        mapping = np.arange(len(keep))
        mapping[~keep] = 0
        return mapping[self.labels]

    def filter_by_largest(self, n: int = 1) -> np.ndarray:
        """Extract a labels volume where only the largest (by size) n labels are kept."""
        keep = np.zeros(len(self) + 1, dtype=bool)
        largest_indices = np.argsort(self.sizes[1:])[-n:] + 1
        keep[largest_indices] = True
        mapping = np.arange(len(keep))
        mapping[~keep] = 0
        return mapping[self.labels]

    def sizes_histogram(self) -> None:
        """Plot the distribution of the sizes of the labels."""
        vals = self.sizes[1:]
        bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 50)
        plt.hist(vals, bins=bins, edgecolor='white', color='orange')
        plt.xscale('log')
        plt.xlabel('Size (number of pixels)')
        plt.ylabel('Frequency')
        plt.title('Histogram over the label sizes')
        plt.show()


class ConnectedComponents(LabeledVolume):
    def __init__(self, vol: np.ndarray, connectivity: int = 1):
        """
        Initializes a ConnectedComponents object.

        Args:
            vol (np.ndarray): Volume to compute connected components on.
            connectivity (int, optional): Controls the squared distance of connectivity. Can range from 1 to 3.

        """
        labels, count = label(vol, structure=generate_binary_structure(rank=3, connectivity=connectivity))
        super().__init__(labels)

    def get_cc(self, index: int | None = None, crop: bool = False) -> np.ndarray:
        """
        Get the connected component with the given index, if index is None selects all components.

        Args:
            index (int): The index of the connected component.
                            If none returns all components.
                            If 'random' returns a random component.
            crop (bool): If True, the volume is cropped to the bounding box of the connected component.

        Returns:
            np.ndarray: The connected component as a binary mask.

        """
        if index is None:
            volume = self.labels
        elif index == 'random':
            index = np.random.randint(1, len(self) + 1)
            volume = self.labels == index
        else:
            assert (
                1 <= index <= len(self)
            ), 'Index out of range. Needs to be in range [1, cc_count].'
            volume = self.labels == index

        if crop:
            # As we index get_bounding_box element 0 will be the bounding box for the connected component at index
            bbox = self.get_bounding_box(index)[0]
            volume = volume[bbox]

        return volume

    def get_bounding_box(self, index: int | None = None) -> list[tuple]:
        """
        Get the bounding boxes of the connected components.

        Args:
            index (int, optional): The index of the connected component. If none selects all components.

        Returns:
            list: A list of bounding boxes.

        """

        if index:
            assert 1 <= index <= len(self), 'Index out of range.'
            return find_objects((self.labels == index).astype(int))
        else:
            return find_objects(self.labels)

def connected_components(volume: np.ndarray, connectivity: int = 1) -> ConnectedComponents:
    """
    Computes connected components of a binary volume.

    Args:
        volume (np.ndarray): An array-like object to be labeled. Any non-zero values in `input` are
            counted as features and zero values are considered the background.
        connectivity (int, optional): Controls the squared distance of connectivity. Can range from 1 to 3.

    Returns:
        cc: A ConnectedComponents object containing the labeled volume and a number of useful methods and attributes.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.cement_128x128x128
        binary = qim3d.filters.gaussian(vol, sigma=2) < 60
        cc = qim3d.segmentation.connected_components(binary)
        color_map = qim3d.viz.colormaps.segmentation(len(cc), style='bright')
        qim3d.viz.slicer(cc.labels, slice_axis=1, color_map=color_map)
        ```
    
    Example: Show the largest connected components
        ```python
        import qim3d

        vol = qim3d.examples.cement_128x128x128
        binary = qim3d.filters.gaussian(vol, sigma=2) < 60
        cc = qim3d.segmentation.connected_components(binary)
        filtered = cc.filter_by_largest(5)

        color_map = qim3d.viz.colormaps.segmentation(len(cc), style='bright')
        qim3d.viz.volumetric(filtered, color_map=color_map, constant_opacity=True)
        ```
    
    Example: Filter the connected components by size
        ```python
        import qim3d

        vol = qim3d.examples.cement_128x128x128
        binary = qim3d.filters.gaussian(vol, sigma=2) < 60
        cc = qim3d.segmentation.connected_components(binary)
        
        # Show a histogram of the distribution of label sizes
        cc.sizes_histogram()

        # Based on the histogram, choose a range of sizes
        filtered = cc.filter_by_size(min_size=1e2, max_size=2e2)

        color_map = qim3d.viz.colormaps.segmentation(len(cc), style='bright')
        qim3d.viz.volumetric(filtered, color_map=color_map, constant_opacity=True)
        ```

    """
    cc = ConnectedComponents(volume, connectivity)
    return cc
