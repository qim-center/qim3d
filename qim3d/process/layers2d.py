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
        '''
        Create an object to store graphs for layered surface segmentations.\n
        - 'Data' must be a numpy.ndarray.\n
        - 'is_inverted' is a boolean which decides if the data is inverted or not.\n
        - 'n_layers' is the number of layers.\n
        - 'delta' is the smoothness parameter.\n
        - 'min_margin' is the minimum margin between layers.\n
        - 'data_not_inverted' is the original data.\n
        - 'data_inverted' is the inverted data.\n
        - 'layers' is a list of GraphObject objects.\n
        - 'helper' is a MaxflowBuilder object.\n
        - 'flow' is the result of the maxflow algorithm on the helper.\n
        - 'segmentations' is a list of segmentations.\n
        - 'segmentation_lines' is a list of segmentation lines.\n
        '''
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
        '''
        Sets data.\n
        - Data must be a numpy.ndarray.
        '''
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
        '''
        Updates data:\n
        - If 'is_inverted' is True, data is set to 'data_inverted'.\n
        - If 'is_inverted' is False, data is set to 'data_not_inverted'.
        '''
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
        '''
        Append a layer to layers.\n
        - Data must be set and not Nonetype before adding a layer.\n
        '''
        if self.get_data() is None:
            raise ValueError("Data must be set before adding a layer.")
        self.get_layers().append(GraphObject(self.get_data()))
    
    def add_n_layers_to_layers(self):
        '''
        Append n_layers to layers.
        '''
        for i in range(self.get_n_layers()):
            self.add_layer_to_layers()
    
    def update_layers(self):
        '''
        Updates layers:\n
        - Resets layers to empty list.\n
        - Appends n_layers to layers.
        '''
        self.set_layers([])
        self.add_n_layers_to_layers()
    
    def get_helper(self):
        return self.helper
    
    def set_helper(self, helper):
        self.helper = helper
    
    def create_new_helper(self):
        '''
        Creates a new helper MaxflowBuilder object.
        '''
        self.set_helper(MaxflowBuilder())
    
    def add_objects_to_helper(self):
        '''
        Adds layers as objects to the helper.
        '''
        self.get_helper().add_objects(self.get_layers())
    
    def add_layered_boundary_cost_to_helper(self):
        '''
        Adds layered boundary cost to the helper.
        '''
        self.get_helper().add_layered_boundary_cost()
    
    def add_layered_smoothness_to_helper(self):
        '''
        Adds layered smoothness to the helper.
        '''
        self.get_helper().add_layered_smoothness(delta = self.get_delta())
    
    def add_a_layered_containment_to_helper(self, outer_object, inner_object):
        '''
        Adds a layered containment to the helper.
        '''
        self.get_helper().add_layered_containment(
                outer_object = outer_object, 
                inner_object = inner_object, 
                min_margin = self.get_min_margin()
            )
    
    def add_all_layered_containments_to_helper(self):
        '''
        Adds all layered containments to the helper.\n
        n_layers most be at least 1.
        '''
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
        '''
        Solves maxflow of the helper and stores the result in self.flow.
        '''
        self.set_flow(self.get_helper().solve())

    def update_helper(self):
        '''
        Updates helper MaxflowBuilder object:\n
        - Adds to helper:
            - objects\n 
            - layered boundary cost\n 
            - layered smoothness\n 
            - all layered containments\n
        - Finally solves maxflow of the helper.
        '''
        self.add_objects_to_helper()
        self.add_layered_boundary_cost_to_helper()
        self.add_layered_smoothness_to_helper()
        self.add_all_layered_containments_to_helper()
        self.solve_helper()

    def get_segmentations(self):
        return self.segmentations

    def set_segmentations(self, segmentations):
        self.segmentations = segmentations

    def add_segmentation_to_segmentations(self, layer, type = np.int32):
        '''
        Adds a segmentation of a layer to segmentations.\n
        '''
        self.get_segmentations().append(self.get_helper().what_segments(layer).astype(type))

    def add_all_segmentations_to_segmentations(self, type = np.int32):
        '''
        Adds all segmentations to segmentations.\n
        - Resets segmentations to empty list.\n
        - Appends segmentations of all layers to segmentations.
        '''
        self.set_segmentations([])
        for l in self.get_layers():
            self.add_segmentation_to_segmentations(l, type = type)

    def get_segmentation_lines(self):
        return self.segmentation_lines
    
    def set_segmentation_lines(self, segmentation_lines):
        self.segmentation_lines = segmentation_lines
        
    def add_segmentation_line_to_segmentation_lines(self, segmentation):
        '''
        Adds a segmentation line to segmentation_lines.\n
        - A segmentation line is the minimum values along a given axis of a segmentation.\n
        - Each segmentation line is shifted by 0.5 to be in the middle of the pixel.
        '''
        self.get_segmentation_lines().append(np.argmin(segmentation, axis = 0) - 0.5)
    
    def add_all_segmentation_lines_to_segmentation_lines(self):
        '''
        Adds all segmentation lines to segmentation_lines.\n
        - Resets segmentation_lines to an empty list.\n
        - Appends segmentation lines of all segmentations to segmentation_lines.
        '''
        self.set_segmentation_lines([])
        for s in self.get_segmentations():
            self.add_segmentation_line_to_segmentation_lines(s)
        
    def update_semgmentations_and_semgmentation_lines(self, type = np.int32):
        '''
        Updates segmentations and segmentation_lines:\n
        - Adds all segmentations to segmentations.\n
        - Adds all segmentation lines to segmentation_lines.
        '''
        self.add_all_segmentations_to_segmentations(type = type)
        self.add_all_segmentation_lines_to_segmentation_lines()

    def prepare_update(self, 
                       data = None,
                       is_inverted = None,
                       n_layers = None,
                       delta = None,
                       min_margin = None
                       ):
        '''
        Prepare update of all fields of the object.\n
        - If a field is None, it is not updated.\n
        - If a field is not None, it is updated.
        '''
        if data is not None:
            self.set_data(data)
        if is_inverted is not None:
            self.set_is_inverted(is_inverted)
        if n_layers is not None:
            self.set_n_layers(n_layers)
        if delta is not None:
            self.set_delta(delta)
        if min_margin is not None:
            self.set_min_margin(min_margin)
    
    def update(self, type = np.int32):
        '''
        Update all fields of the object.
        '''
        self.update_data_not_inverted()
        self.update_data_inverted()
        self.update_data()
        self.update_layers()
        self.create_new_helper()
        self.update_helper()
        self.update_semgmentations_and_semgmentation_lines(type = type)

    def __repr__(self):
        '''
        Returns string representation of all fields of the object.
        '''
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
    # Draw results.
    def visulise(l2d = None):
        plt.figure(figsize = (10, 10))
        ax = plt.subplot(1, 3, 1)
        ax.imshow(l2d.get_data(), cmap = "gray")

        ax = plt.subplot(1, 3, 2)
        ax.imshow(np.sum(l2d.get_segmentations(), axis = 0))

        ax = plt.subplot(1, 3, 3)
        ax.imshow(data, cmap = "gray")
        for line in l2d.get_segmentation_lines():
            ax.plot(line)
        plt.show()
    
    
    path = os.path.join(os.getcwd(), "qim3d", "img_examples", "slice_218x193.png")
    data = imread(path).astype(np.int32)

    layers2d = Layers2d(data = data, n_layers = 3, delta = 1, min_margin = 10)
    layers2d.update()
    visulise(layers2d)
    
    layers2d.prepare_update(n_layers = 1)
    layers2d.update()
    visulise(layers2d)
    
    layers2d.prepare_update(is_inverted = True)
    layers2d.update()
    visulise(layers2d)