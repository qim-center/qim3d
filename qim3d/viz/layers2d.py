""" Provides a collection of visualisation functions for the Layers2d class."""
import matplotlib.pyplot as plt
import numpy as np
from qim3d.process import layers2d as l2d

def create_subplot_of_2d_arrays(data, m_rows = 1, n_cols = 1, figsize = None):
    '''
    Creates a `m x n` grid subplot from a collection of 2D arrays.
    
    Args:
        `data` (list of 2D numpy.ndarray): A list of 2d numpy.ndarray.
        `m_rows` (int): The number of rows in the subplot grid.
        `n_cols` (int): The number of columns in the subplot grid.

    Raises:
        ValueError: If the product of m_rows and n_cols is not equal to the number of 2d arrays in data.

    Notes:
    - Subplots are organized in a m rows x n columns Grid.
    - The total number of subplots is equal to the product of m_rows and n_cols.
    
    Returns:
        A tuple of (`fig`, `ax_list`), where fig is a matplotlib.pyplot.figure and ax_list is a list of matplotlib.pyplot.axes.
    '''
    total = m_rows * n_cols
    
    if total != len(data):
        raise ValueError("The product of m_rows and n_cols must be equal to the number of 2D arrays in data.\nCurrently, m_rows * n_cols = {}, while arrays in data = {}".format(m_rows * n_cols, len(data)))
    
    pos_idx = range(1, total + 1)
    
    if figsize is None:
        figsize = (m_rows * 10, n_cols * 10)
    fig = plt.figure(figsize = figsize)
    
    ax_list = []
    
    for k in range(total):
        ax_list.append(fig.add_subplot(m_rows, n_cols, pos_idx[k]))
        ax_list[k].imshow(data[k], cmap = "gray")
    
    plt.tight_layout()
    return fig, ax_list

def create_plot_of_2d_array(data, figsize = (10, 10)):
    '''
    Creates a plot of a 2D array.
    
    Args:
        `data` (list of 2D numpy.ndarray): A list of 2d numpy.ndarray.
        `figsize` (tuple of int): The figure size.
    Notes:
        - If data is not a list, it is converted to a list.
    Returns:
        A tuple of (`fig`, `ax`), where fig is a matplotlib.pyplot.figure and ax is a matplotlib.pyplot.axes.
    '''
    if not isinstance(data, list):
        data = [data]
    
    fig, ax_list = create_subplot_of_2d_arrays(data, figsize = figsize)
    return fig, ax_list[0]
    
def merge_multiple_segmentations_2d(segmentations):
    '''
    Merges multiple segmentations of a 2D image into a single image.
    
    Args:
        `segmenations` (list of numpy.ndarray): A list of 2D numpy.ndarray.
    Returns:
        A 2D numpy.ndarray representing the merged segmentations.
    '''
    if len(segmentations) == 0:
        raise ValueError("Segmentations must contain at least one segmentation.")
    if len(segmentations) == 1:
        return segmentations[0]
    else:
        return np.sum(segmentations, axis = 0)

def add_line_to_plot(axes, line, line_color = None):
    '''
    Adds a line to plot.
    
    Args:
        `axes` (matplotlib.pyplot.axes): A matplotlib.pyplot.axes.
        `line` (numpy.ndarray): A 1D numpy.ndarray.
    
    Notes:
        - The line is added on top of to the plot.
    '''
    if line_color is None:
        axes.plot(line)
    else:
        axes.plot(line, color = line_color)

def add_lines_to_plot(axes, lines, line_colors = None):
    '''
    Adds multiple lines to plot.
    
    Args:
        `axes` (matplotlib.pyplot.axes): A matplotlib.pyplot.axes.
        `lines` (list of numpy.ndarray): A list of 1D numpy.ndarray.
    
    Notes:
        - The lines are added on top of to the plot.
    '''
    if line_colors is None:
        for line in lines:
            axes.plot(line)
    else:
        for i in range(len(lines)):
            axes.plot(lines[i], color = line_colors[i])

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
        n_layers=4,
        ) 
    l2d_obj.update()    
    
    
    # Show how merge_multiple_segmentations_2d works:
    data_seg = []
    for i in range(len(l2d_obj.get_segmentations())):
        data_seg.append(merge_multiple_segmentations_2d(l2d_obj.get_segmentations()[:i+1]))
    
    # Show how create_plot_from_2d_arrays works:
    fig1, ax1 = create_plot_of_2d_array(data_seg[0])
    
    data_lines = []
    for i in range(len(l2d_obj.get_segmentation_lines())):
        data_lines.append(l2d_obj.get_segmentation_lines()[i])
    
    # Show how add_line_to_plot works:
    add_line_to_plot(ax1, data_lines[1])
    
    # Show how create_subplot_of_2d_arrays works:
    fig3, ax_list = create_subplot_of_2d_arrays(
            data_seg, 
            m_rows = 1, 
            n_cols = len(l2d_obj.get_segmentations())
            # m_rows = len(l2d_obj.get_segmentations()), 
            # n_cols = 1
        )
    
    # Show how add_lines_to_plot works:
    add_lines_to_plot(ax_list[1], data_lines[1:3])
    
    plt.show()
    