"""
The GUI can be launched directly from the command line:

```bash
qim3d gui --layers
```

Or launched from a python script

```python
import qim3d

layers = qim3d.gui.layers2d.Interface()
app = layers.launch()
```
![gui-layers](../../assets/screenshots/GUI-layers.png)

"""

import os

import gradio as gr
import numpy as np
from .interface import BaseInterface

# from qim3d.processing import layers2d as l2d
from qim3d.processing import segment_layers, get_lines
from qim3d.operations import overlay_rgb_images
from qim3d.io import load
from qim3d.viz._layers2d import image_with_lines
from typing import Dict, Any

#TODO figure out how not update anything and go through processing when there are no data loaded
# So user could play with the widgets but it doesnt throw error
# Right now its only bypassed with several if statements
# I opened an issue here https://github.com/gradio-app/gradio/issues/9273

X = 'X'
Y = 'Y'
Z = 'Z'
AXES = {X:2, Y:1, Z:0}

DEFAULT_PLOT_TYPE = 'Segmentation mask'
SEGMENTATION_COLORS = np.array([[0, 255, 255], # Cyan
                                [255, 195, 0], # Yellow Orange
                                [199, 0, 57], # Dark orange
                                [218, 247, 166], # Light green
                                [255, 0, 255], # Magenta
                                [65, 105, 225], # Royal blue
                                [138, 43, 226], # Blue violet
                                [255, 0, 0], #Red
                                ])

class Interface(BaseInterface):
    def __init__(self):
        super().__init__("Layered surfaces 2D", 1080)

        self.data = None
        # It important to keep the name of the attributes like this (including the capital letter) becuase of
        # accessing the attributes via __dict__
        self.X_slice = None
        self.Y_slice = None
        self.Z_slice = None
        self.X_segmentation = None
        self.Y_segmentation = None
        self.Z_segmentation = None

        self.plot_type = DEFAULT_PLOT_TYPE

        self.error = False



    def define_interface(self):
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                with gr.Row():
                    with gr.Column(scale=99, min_width=128):
                        base_path = gr.Textbox(
                            max_lines=1,
                            container=False,
                            label="Base path",
                            value=os.getcwd(),
                        )

                    with gr.Column(scale=1, min_width=36):
                        reload_base_path = gr.Button(value="⟳")

                explorer = gr.FileExplorer(
                    ignore_glob="*/.*",
                    root_dir=os.getcwd(),
                    label=os.getcwd(),
                    render=True,
                    file_count="single",
                    interactive=True,
                    height = 230,
                )

                        
                with gr.Group():
                    with gr.Row():
                        axis = gr.Radio(
                            choices = [Z, Y, X],
                            value = Z, 
                            label = 'Layer axis',
                            info = 'Specifies in which direction are the layers. The order of axes is ZYX',)
                    with gr.Row():
                        wrap = gr.Checkbox(
                            label = "Lines start and end at the same level.",
                            info = "Used when segmenting layers of unfolded image."
                        )
                        
                        is_inverted = gr.Checkbox(
                            label="Invert image before processing",
                            info="The algorithm effectively flips the gradient.",
                        ) 
                    
                    with gr.Row():
                        delta = gr.Slider(
                            minimum=0,
                            maximum=5,
                            value=0.75,
                            step=0.01,
                            interactive = True,
                            label="Delta value",
                            info="The lower the delta is, the more accurate the gradient calculation will be. However, the calculation takes longer to execute. Delta above 1 is rounded down to closest lower integer", 
                        )
                        
                    with gr.Row():
                        min_margin = gr.Slider(
                            minimum=1, 
                            maximum=50, 
                            value=10, 
                            step=1, 
                            interactive = True,
                            label="Min margin",
                            info="Minimum margin between layers to be detected in the image.",
                        )

                    with gr.Row():
                        n_layers = gr.Slider(
                            minimum=1,
                            maximum=len(SEGMENTATION_COLORS) - 1,
                            value=2,
                            step=1,
                            interactive=True,
                            label="Number of layers",
                            info="Number of layers to be detected in the image",
                        )                 

                # with gr.Row():
                #     btn_run = gr.Button("Run Layers2D", variant = 'primary')

            # Output panel: Plots
            """
            60em if plot is alone
            30em if two of them
            20em if all of them are visible

            When one slicing axis is made unvisible we want the other two images to be bigger
            For some reason, gradio changes their width but not their height. So we have to 
            change their height manually
            """

            self.heights = ['60em', '30em', '20em'] # em units are relative to the parent, 


            with gr.Column(scale=2,):
                # with gr.Row(): # Source image outputs
                #     input_image_kwargs = lambda axis: dict(
                #         show_label = True,
                #         label = F'Slice along {axis}-axis', 
                #         visible = True, 
                #         height = self.heights[2]
                #     )

                #     input_plot_x = gr.Image(**input_image_kwargs('X'))
                #     input_plot_y = gr.Image(**input_image_kwargs('Y'))
                #     input_plot_z = gr.Image(**input_image_kwargs('Z'))

                with gr.Row(): # Detected layers outputs
                    output_image_kwargs = lambda axis: dict(
                        show_label = True,
                        label = F'Detected layers {axis}-axis',
                        visible = True,
                        height = self.heights[2]
                    )
                    output_plot_x = gr.Image(**output_image_kwargs('X'))
                    output_plot_y = gr.Image(**output_image_kwargs('Y'))
                    output_plot_z = gr.Image(**output_image_kwargs('Z'))
                    
                with gr.Row(): # Axis position sliders
                    slider_kwargs = lambda axis: dict(
                        minimum = 0,
                        maximum = 1,
                        value = 0.5,
                        step = 0.01,
                        label = F'{axis} position',
                        info = F'The 3D image is sliced along {axis}-axis'
                    )
                    
                    x_pos = gr.Slider(**slider_kwargs('X'))                    
                    y_pos = gr.Slider(**slider_kwargs('Y'))
                    z_pos = gr.Slider(**slider_kwargs('Z'))

                with gr.Row():
                    x_check = gr.Checkbox(value = True, interactive=True, label = 'Show X slice')
                    y_check = gr.Checkbox(value = True, interactive=True, label = 'Show Y slice')
                    z_check = gr.Checkbox(value = True, interactive=True, label = 'Show Z slice')

                with gr.Row():
                    with gr.Group():
                        plot_type = gr.Radio(
                            choices= (DEFAULT_PLOT_TYPE, 'Segmentation lines',),
                            value = DEFAULT_PLOT_TYPE,
                            interactive = True,
                            show_label=False
                            )
                        
                        alpha = gr.Slider(
                            minimum=0,
                            maximum = 1,
                            step = 0.01,
                            label = 'Alpha value',
                            show_label=True,
                            value = 0.5,
                            visible = True,
                            interactive=True
                            )
                        
                        line_thickness = gr.Slider(
                            minimum=0.1,
                            maximum = 5,
                            value = 2,
                            label = 'Line thickness',
                            show_label = True,
                            visible = False,
                            interactive = True
                            )

                with gr.Row():
                    btn_run = gr.Button("Run Layers2D", variant = 'primary')


        positions = [x_pos, y_pos, z_pos]
        process_inputs = [axis, is_inverted, delta, min_margin, n_layers, wrap]
        plotting_inputs = [axis, alpha, line_thickness]
        # input_plots = [input_plot_x, input_plot_y, input_plot_z]
        output_plots = [output_plot_x, output_plot_y, output_plot_z]
        visibility_check_inputs = [x_check, y_check, z_check]

        spinner_loading = gr.Text("Loading data...", visible=False)
        spinner_running = gr.Text("Running pipeline...", visible=False)

        reload_base_path.click(
            fn=self.update_explorer,inputs=base_path, outputs=explorer)
        
        plot_type.change(
            self.change_plot_type, inputs = plot_type, outputs = [alpha, line_thickness]).then(
            fn = self.plot_output_img_all, inputs = plotting_inputs, outputs = output_plots
            )
        
        gr.on(
            triggers = [alpha.release, line_thickness.release],
            fn = self.plot_output_img_all, inputs = plotting_inputs, outputs = output_plots
        )

        """
        Difference between btn_run.click and the other triggers below is only loading the data.
        To make it easier to maintain, I created 'update_component' variable. Its value is completely
        unimportant. It exists only to be changed after loading the data which triggers further processing
        which is the same for button click and the other triggers
        """

        update_component = gr.State(True)

        btn_run.click(
            fn=self.set_spinner, inputs=spinner_loading, outputs=btn_run).then(
            fn=self.load_data, inputs = [base_path, explorer]).then(
            fn = lambda state: not state, inputs = update_component, outputs = update_component)
        
        gr.on(
            triggers= (axis.change, is_inverted.change, delta.release, min_margin.release, n_layers.release, update_component.change, wrap.change),
            fn=self.set_spinner, inputs = spinner_running, outputs=btn_run).then(
            fn=self.process_all, inputs = [*positions, *process_inputs]).then(
            # fn=self.plot_input_img_all, outputs = input_plots, show_progress='hidden').then(
            fn=self.plot_output_img_all, inputs =  plotting_inputs, outputs = output_plots, show_progress='hidden').then(
            fn=self.set_relaunch_button, inputs=[], outputs=btn_run)
        
        # Chnages visibility and sizes of the plots - gives user the option to see only some of the images and in bigger scale
        gr.on(
            triggers=[x_check.change, y_check.change, z_check.change],
            fn = self.change_row_visibility, inputs = visibility_check_inputs, outputs = positions).then(
            # fn = self.change_row_visibility, inputs = visibility_check_inputs, outputs = input_plots).then(
            fn = self.change_plot_size, inputs = visibility_check_inputs, outputs = output_plots)
        
        # for  axis, slider, input_plot, output_plot in zip(['x','y','z'], positions, input_plots, output_plots):
        for  axis, slider, output_plot in zip([X,Y,Z], positions, output_plots):
            slider.change(
                self.process_wrapper(axis), inputs = [slider, *process_inputs]).then( 
                # self.plot_input_img_wrapper(axis), outputs = input_plot).then(
                self.plot_output_img_wrapper(axis), inputs = plotting_inputs, outputs = output_plot)
            
        

    def change_plot_type(self, plot_type: str, ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        self.plot_type = plot_type
        if plot_type == 'Segmentation lines':
            return gr.update(visible = False), gr.update(visible = True)
        else:  
            return gr.update(visible = True), gr.update(visible = False)
        
    def change_plot_size(self, x_check: int, y_check: int, z_check: int) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Based on how many plots are we displaying (controlled by checkboxes in the bottom) we define
        also their height because gradio doesn't do it automatically. The values of heights were set just by eye.
        They are defines before defining the plot in 'define_interface'
        """
        index = x_check + y_check + z_check - 1
        height = self.heights[index] # also used to define heights of plots in the begining
        return gr.update(height = height, visible= x_check), gr.update(height = height, visible = y_check), gr.update(height = height, visible = z_check)

    def change_row_visibility(self, x_check: int, y_check: int, z_check: int):
        return self.change_visibility(x_check), self.change_visibility(y_check), self.change_visibility(z_check)
    
    def update_explorer(self, new_path: str):
        # Refresh the file explorer object
        new_path = os.path.expanduser(new_path)

        # In case we have a directory
        if os.path.isdir(new_path):
            return gr.update(root_dir=new_path, label=new_path)

        elif os.path.isfile(new_path):
            parent_dir = os.path.dirname(new_path)
            file_name = str(os.path.basename(new_path))
            return gr.update(root_dir=parent_dir, label=parent_dir, value=file_name)

        else:
            raise ValueError("Invalid path")

    def set_relaunch_button(self):
        return gr.update(value=f"Relaunch", interactive=True)

    def set_spinner(self, message: str):
        if self.error:
            return gr.Button()
        # spinner icon/shows the user something is happeing
        return gr.update(value=f"{message}", interactive=False)
    
    def load_data(self, base_path: str, explorer: str):
        if base_path and os.path.isfile(base_path):
            file_path = base_path
        elif explorer and os.path.isfile(explorer):
            file_path = explorer
        else:
            raise gr.Error("Invalid file path")

        try:
            self.data = qim3d.io.load(
                file_path,
                progress_bar=False
            )
        except Exception as error_message:
            raise gr.Error(
                f"Failed to load the image: {error_message}"
            ) from error_message
        
    def process_all(self, x_pos:float, y_pos:float, z_pos:float, axis:str, inverted:bool, delta:float, min_margin:int, n_layers:int, wrap:bool):
        self.process_wrapper(X)(x_pos, axis, inverted, delta, min_margin, n_layers, wrap)
        self.process_wrapper(Y)(y_pos, axis, inverted, delta, min_margin, n_layers, wrap)
        self.process_wrapper(Z)(z_pos, axis, inverted, delta, min_margin, n_layers, wrap)

    def process_wrapper(self, slicing_axis:str):
        """
        The function behaves the same in all 3 directions, however we have to know in which direction we are now.
        Thus we have this wrapper function, where we pass the slicing axis - in which axis are we indexing the data
            and we return a function working in that direction
        """
        slice_key = F'{slicing_axis}_slice'
        seg_key = F'{slicing_axis}_segmentation'
        slicing_axis_int = AXES[slicing_axis]

        def process(pos:float, segmenting_axis:str, inverted:bool, delta:float, min_margin:int, n_layers:int, wrap:bool):
            """
            Parameters:
            -----------
            pos: Relative position of a slice from data
            segmenting_axis: In which direction we want to detect layers
            inverted: If we want use inverted gradient
            delta: Smoothness parameter
            min_margin: What is the minimum distance between layers. If it was 0, all layers would be the same
            n_layers: How many layer boarders we want to find
            wrap: If True, the starting point and end point will be at the same level. Useful when segmenting unfolded images.
            """
            slice = self.get_slice(pos, slicing_axis_int)
            self.__dict__[slice_key] = slice

            if segmenting_axis == slicing_axis:
                self.__dict__[seg_key] = None
            else:
                
                if self.is_transposed(slicing_axis, segmenting_axis):
                    slice = np.rot90(slice)
                self.__dict__[seg_key] = qim3d.processing.segment_layers(slice, inverted = inverted, n_layers = n_layers, delta = delta, min_margin = min_margin, wrap = wrap)
        
        return process

    def is_transposed(self, slicing_axis:str, segmenting_axis:str):
        """
        Checks if the desired direction of segmentation is the same if the image would be submitted to segmentation as is. 
        If it is not, we have to rotate it before we put it to segmentation algorithm
        """
        remaining_axis = F"{X}{Y}{Z}".replace(slicing_axis, '').replace(segmenting_axis, '')
        return AXES[segmenting_axis] > AXES[remaining_axis]
    
    def get_slice(self, pos:float, axis:int):
        idx = int(pos * (self.data.shape[axis] - 1))
        return np.take(self.data, idx, axis = axis)
    
    # def plot_input_img_wrapper(self, axis:str):
    #     slice_key = F'{axis.lower()}_slice'
    #     def plot_input_img():
    #         slice = self.__dict__[slice_key]
    #         slice = slice + np.abs(np.min(slice))
    #         slice = slice / np.max(slice)
    #         return slice
    #     return plot_input_img

    # def plot_input_img_all(self):
    #     x_plot = self.plot_input_img_wrapper('x')()
    #     y_plot = self.plot_input_img_wrapper('y')()
    #     z_plot = self.plot_input_img_wrapper('z')()
    #     return x_plot, y_plot, z_plot
    
    def plot_output_img_wrapper(self, slicing_axis:str):
        slice_key = F'{slicing_axis}_slice'
        seg_key = F'{slicing_axis}_segmentation'

        def plot_output_img(segmenting_axis:str, alpha:float, line_thickness:float):
            slice = self.__dict__[slice_key]
            seg = self.__dict__[seg_key]

            if seg is None: # In case segmenting axis si the same as slicing axis
                return slice
            
            if self.plot_type == DEFAULT_PLOT_TYPE:
                n_layers = len(seg) + 1
                seg = np.sum(seg, axis = 0)
                seg = np.repeat(seg[..., None], 3, axis = -1)
                for i in range(n_layers):
                    seg[seg[:,:,0] == i, :] = SEGMENTATION_COLORS[i]

                if self.is_transposed(slicing_axis, segmenting_axis):
                    seg = np.rot90(seg, k = 3)
                # slice = 255 * (slice/np.max(slice))
                # return image_with_overlay(np.repeat(slice[..., None], 3, -1), seg, alpha) 
                return qim3d.operations.overlay_rgb_images(slice, seg, alpha)
            else:
                lines = qim3d.processing.get_lines(seg)
                if self.is_transposed(slicing_axis, segmenting_axis):
                    return qim3d.viz.image_with_lines(np.rot90(slice), lines, line_thickness).rotate(270, expand = True)
                else:
                    return qim3d.viz.image_with_lines(slice, lines, line_thickness)
            
        return plot_output_img
    
    def plot_output_img_all(self, segmenting_axis:str, alpha:float, line_thickness:float):
        x_output = self.plot_output_img_wrapper(X)(segmenting_axis, alpha, line_thickness)
        y_output = self.plot_output_img_wrapper(Y)(segmenting_axis, alpha, line_thickness)
        z_output = self.plot_output_img_wrapper(Z)(segmenting_axis, alpha, line_thickness)
        return x_output, y_output, z_output

if __name__ == "__main__":
    Interface().run_interface()
    