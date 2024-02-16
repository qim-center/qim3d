import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def watershed_segment(volume):
    """ Apply watershed algorithm to a 3D volume.

    Args:
        volume (np.array | torch.Tensor):  A 3D volume.

    Returns:
        np.array: Segmented watershed Connected Components.
    """
    volume = volume > 1
    distance = ndi.distance_transform_edt(volume)
    coords = peak_local_max(distance, footprint=np.ones((3,)*volume.ndim), labels=volume)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=volume)
    
    return labels