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
        self.n_layers = n_layers
        self.delta = delta
        self.min_margin = min_margin
        
        self.data_not_inverted = None
        self.data_inverted = None        
        self.layers = []
        self.helper = MaxflowBuilder()
        self.flow = None
        self.segmentations = []
        self.segmentation_lines = []


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
        
    def get_delta(self):
        return self.delta
    
    def set_delta(self, delta):
        self.delta = delta
    
    def get_min_margin(self):
        return self.min_margin
    
    def set_min_margin(self, min_margin):
        self.min_margin = min_margin    
    
    def get_data_not_inverted(self):
        return self.data_not_inverted
    
    def set_data_not_inverted(self, data_not_inverted):
        self.data_not_inverted = data_not_inverted
    
    def get_data_inverted(self):
        return self.data_inverted
    
    def set_data_inverted(self, data_inverted):
        self.data_inverted = data_inverted
    
    def update_data_not_inverted(self):
        self.set_data_not_inverted(self.get_data())
    
    def update_data_inverted(self):
        if self.get_data() is not None:
            self.set_data_inverted(~self.get_data())
        else:
            self.set_data_inverted(None)
    
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
        if self.get_data() is None:
            raise ValueError("Data must be set before adding a layer.")
        self.get_layers().append(GraphObject(self.get_data()))
    
    def add_n_layers_to_layers(self):
        for i in range(self.get_n_layers()):
            self.add_layer_to_layers()
    
    def update_layers(self):
        self.set_layers([])
        self.add_n_layers_to_layers()
    
    def get_helper(self):
        return self.helper
    
    def set_helper(self, helper):
        self.helper = helper
    
    def create_new_helper(self):
        self.set_helper(MaxflowBuilder())
    
    def add_objects_to_helper(self):
        self.get_helper().add_objects(self.get_layers())
    
    def add_layered_boundary_cost_to_helper(self):
        self.get_helper().add_layered_boundary_cost()
    
    def add_layered_smoothness_to_helper(self):
        self.get_helper().add_layered_smoothness(delta = self.get_delta())
    
    def add_a_layered_containment_to_helper(self, outer_object, inner_object):
        self.get_helper().add_layered_containment(
                outer_object = outer_object, 
                inner_object = inner_object, 
                min_margin = self.get_min_margin()
            )
    
    def add_all_layered_containments_to_helper(self):
        if len(self.get_layers()) < 1:
            raise ValueError("There must be at least 1 layer to add containment.")
        
        for i in range(self.get_n_layers()-1):
            self.add_a_layered_containment_to_helper(
                    outer_object  = self.get_layers()[i], 
                    inner_object = self.get_layers()[i + 1]
                )
    
    def get_flow(self):
        return self.flow
    
    def set_flow(self, flow):
        self.flow = flow
    
    def solve_helper(self):
        self.set_flow(self.get_helper().solve())

    def get_segmentations(self):
        return self.segmentations

    def set_segmentations(self, segmentations):
        self.segmentations = segmentations

    def add_segmentation_to_segmentations(self, layer, type = np.int32):
        self.get_segmentations().append(self.get_helper().what_segments(layer).astype(type))

    def add_all_segmentations_to_segmentations(self, type = np.int32):
        self.set_segmentations([])
        for l in self.get_layers():
            self.add_segmentation_to_segmentations(l, type = type)

    def get_segmentation_lines(self):
        return self.segmentation_lines
    
    def set_segmentation_lines(self, segmentation_lines):
        self.segmentation_lines = segmentation_lines
        
    def add_segmentation_line_to_segmentation_lines(self, segmentation):
        self.get_segmentation_lines().append(np.argmin(segmentation, axis = 0) - 0.5)
    
    def add_all_segmentation_lines_to_segmentation_lines(self):
        self.set_segmentation_lines([])
        for s in self.get_segmentations():
            self.add_segmentation_line_to_segmentation_lines(s)
    
    def update_helper(self, type = np.int32):
        self.add_objects_to_helper()
        self.add_layered_boundary_cost_to_helper()
        self.add_layered_smoothness_to_helper()
        self.add_all_layered_containments_to_helper()
        self.solve_helper()
        self.add_all_segmentations_to_segmentations(type = type)
        self.add_all_segmentation_lines_to_segmentation_lines()

    def update(self, type = np.int32):
        self.update_data_not_inverted()
        self.update_data_inverted()
        self.update_data()
        self.update_layers()
        self.create_new_helper()
        self.update_helper(type = type)

    def __repr__(self):
        return "data: %s\n, \nis_inverted: %s, \nn_layers: %s, \ndelta: %s, \nmin_margin: %s, \ndata_not_inverted: %s, \ndata_inverted: %s, \nlayers: %s, \nhelper: %s, \nflow: %s, \nsegmentations: %s, \nsegmentations_lines: %s" % (
            self.get_data(),
            self.get_is_inverted(),
            self.get_n_layers(),
            self.get_delta(),
            self.get_min_margin(),
            self.get_data_not_inverted(),
            self.get_data_inverted(),
            self.get_layers(),
            self.get_helper(),
            self.get_flow(),
            self.get_segmentations(),
            self.get_segmentation_lines()
        )

import matplotlib.pyplot as plt
from skimage.io import imread
    
if __name__ == "__main__":        
    
    path = os.path.join(os.getcwd(), "qim3d", "img_examples", "slice_218x193.png")
    data = imread(path).astype(np.int32)

    layers2d = Layers2d(data = data, n_layers = 3)
    layers2d.update()
    
    print(np.shape(layers2d.segmentations))
    print(np.shape(layers2d.segmentation_lines))

    print(layers2d.get_segmentations())

    # Draw results.
    def visulise():
        plt.figure(figsize = (10, 10))
        ax = plt.subplot(1, 3, 1)
        ax.imshow(layers2d.get_data(), cmap = "gray")

        ax = plt.subplot(1, 3, 2)
        ax.imshow(np.sum(layers2d.get_segmentations(), axis = 0))

        ax = plt.subplot(1, 3, 3)
        ax.imshow(data, cmap = "gray")
        for line in layers2d.get_segmentation_lines():
            ax.plot(line)
        plt.show()
    
    visulise()   