"""Class for layered surface segmentation in 2D images."""
import numpy as np
import os
from slgbuilder import GraphObject 
from slgbuilder import MaxflowBuilder

class Layers2d:
    """
    Create an object to store graphs for layered surface segmentations.

    Args:
        data (numpy.ndarray, optional): 2D image data.
        n_layers (int, optional): Number of layers. Defaults to 1.
        delta (int, optional): Smoothness parameter. Defaults to 1.
        min_margin (int, optional): Minimum margin between layers. Defaults to 10.
        inverted (bool, optional): Choose inverted data for segmentation. Defaults to False.

    Raises:
        TypeError: If `data` is not numpy.ndarray.
    
    Example:
        layers2d = Layers2d(data = np_arr, n_layers = 3, delta = 5, min_margin = 20)
    """
        
    def __init__(self, 
                 data = None, 
                 is_inverted = False,
                 n_layers = 1, 
                 delta = 1, 
                 min_margin = 10
                 ):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("Data must be a numpy.ndarray.")
        
        self.data = data
        self.is_inverted = is_inverted
        self.data_not_inverted = data
        self.data_inverted = ~data
        
        if self.is_inverted:
            self.data = self.data_inverted
        else:
            self.data = self.data_not_inverted
        
        self.n_layers = n_layers
        
        self.layers = []

        for i in range(self.n_layers):
            self.layers.append(GraphObject(self.data))
        
        self.helper = MaxflowBuilder()
        self.helper.add_objects(self.layers)
        self.helper.add_layered_boundary_cost()
        self.helper.add_layered_smoothness(delta = delta)
        
        for i in range(len(self.layers)-1):
            self.helper.add_layered_containment(
                outer_object = self.layers[i], 
                inner_object = self.layers[i + 1], 
                min_margin = min_margin
            ) 
        
        self.flow = self.helper.solve()
        
        self.segmentations = [self.helper.what_segments(l).astype(np.int32) for l in self.layers] 
        self.segmentation_lines = [np.argmin(s, axis = 0) - 0.5 for s in self.segmentations]

    def get_data(self):
        return self.data    
    
    def set_data(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy.ndarray.")
        self.data = data
    
    def get_is_inverted(self):
        return self.is_inverted
    
    def set_is_inverted(self, is_inverted):
        self.is_inverted = is_inverted
        
    def get_data_not_inverted(self):
        return self.data_not_inverted
    
    def set_data_not_inverted(self, data):
        self.data_not_inverted = data
    
    def get_data_inverted(self):
        return self.data_inverted
    
    def set_data_inverted(self, data):
        self.data_inverted = data
    
    def update_data_not_inverted(self):
        self.set_data_not_inverted(self.get_data())
    
    def update_data_inverted(self):
        self.set_data_inverted(~self.get_data())
    
    def update_data(self):
        if self.get_is_inverted():
            self.set_data(self.get_data_inverted())
        else:
            self.set_data(self.get_data_not_inverted())        
    
    def get_n_layers(self):
        return self.n_layers
    
    def set_n_layers(self, n_layers):
        self.n_layers = n_layers
    
    def get_layers(self):
        return self.layers
    
    def set_layers(self, layers):
        self.layers = layers
    
    def add_layer_to_layers(self):
        self.get_layers().append(GraphObject(self.get_data()))
    
    def add_n_layers_to_layers(self):
        for i in range(self.get_n_layers()):
            self.add_layer()
    
    def update_layers(self):
        self.set_layers([])
        self.add_n_layers_to_layers()
    
    


import matplotlib.pyplot as plt
import nibabel as nib
from skimage.io import imread
    
if __name__ == "__main__":        
    
    path = os.path.join(os.getcwd(), "qim3d", "img_examples", "slice_218x193.png")
    data = imread(path).astype(np.int32)

    layers2d = Layers2d(data = data, n_layers = 3)
    
    print(np.shape(layers2d.segmentations))
    
    print(np.shape(layers2d.segmentation_lines))
    
    # Draw results.
    plt.figure(figsize = (10, 10))
    ax = plt.subplot(1, 3, 1)
    ax.imshow(layers2d.data, cmap = "gray")

    ax = plt.subplot(1, 3, 2)
    ax.imshow(np.sum(layers2d.segmentations, axis = 0))

    ax = plt.subplot(1, 3, 3)
    ax.imshow(data, cmap = "gray")
    for line in layers2d.segmentation_lines:
        ax.plot(line)
    plt.show()