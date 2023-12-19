""" Provides a collection of visualisation functions for the Layers2d class."""
import matplotlib.pyplot as plt
import numpy as np
from qim3d.process import layers2d as l2d

def create_subplot_from_2d_arrays(data, m_rows = 1, n_cols = 1, figsize = (10, 10)):
    '''
    Creates a subplot from a collection of 2d arrays.
    
    Args:
        data (list of 2d numpy.ndarray): A list of 2d numpy.ndarray.
        m_rows (int): The number of rows in the subplot grid.
        n_cols (int): The number of columns in the subplot grid.

    Raises:
        ValueError: If the product of m_rows and n_cols is not equal to the number of 2d arrays in data.

    Notes:
    - Subplots are organized in a m rows x n columns Grid.
    - The total number of subplots is equal to the product of m_rows and n_cols.
    '''
    total = m_rows * n_cols
    
    if total != len(data):
        raise ValueError("The product of m_rows and n_cols must be equal to the number of 2d arrays in data.\nCurrently, m_rows * n_cols = {}, while arrays in data = {}".format(m_rows * n_cols, len(data)))
    
    pos_idx = range(1, total + 1)
    
    fig = plt.figure(figsize = figsize)
    
    for k in range(total):
        ax = fig.add_subplot(m_rows, n_cols, pos_idx[k])
        ax.imshow(data[k], cmap = "gray")

import os
from skimage.io import imread

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "qim3d", "img_examples", "slice_218x193.png")
    data = imread(path).astype(np.int32)
    
    l2d_obj = l2d.Layers2d()
    l2d_obj.prepare_update(
        data = data, 
        is_inverted=False,
        delta=1,
        min_margin=10,
        n_layers=3
        ) 
    l2d_obj.update()    
    
    data = []
    for i in range(len(l2d_obj.get_segmentations())):
        data.append(l2d_obj.get_segmentations()[i])
    
    # Create a subplot
    create_subplot_from_2d_arrays(data, m_rows = 1, n_cols = 3, figsize = (10, 10))
    
    # Display the plot
    plt.show()
    
