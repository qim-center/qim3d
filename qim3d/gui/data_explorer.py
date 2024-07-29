"""
The GUI can be launched directly from the command line:

```bash
qim3d gui --data-explorer
```

Or launched from a python script

```python
import qim3d

app = qim3d.gui.data_explorer.Interface()
app.launch()
```
"""

import datetime
import os
import re

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import outputformat as ouf

from qim3d.io import load
from qim3d.utils.logger import log
from qim3d.utils import misc

from qim3d.gui.interface import BaseInterface


class Interface(BaseInterface):
    def __init__(self,
                 verbose:bool = False,
                 figsize:int = 8,
                 display_saturation_percentile:int = 99,
                 nbins:int = 32):
        """
        Parameters:
        -----------
        show_header (bool, optional): If true, prints starting info into terminal. Default is False
        verbose (bool, optional): If true, prints info during session into terminal. Defualt is False.
        figsize (int, optional): Sets the size of plots displaying the slices. Default is 8.
        """
        super().__init__(
            title = "Data Explorer",
            height = 1024,
            width = 900,
            verbose = verbose
        )
        self.axis_dict = {"Z":0, "Y":1, "X":2}
        self.all_operations = [
            "Z Slicer",
            "Y Slicer", 
            "X Slicer",
            "Z max projection",
            "Z min projection",
            "Intensity histogram",
            "Data summary",
        ]
        self.calculated_operations = [] # For changing the visibility of results, we keep track what was calculated and thus will be displayed

        self.vol = None # The loaded volume

        # Plotting
        self.figsize = figsize

        # Projections and histogram
        self.min_percentile = None
        self.min_percentile = None
        self.display_saturation_percentile = display_saturation_percentile
        self.nbins = nbins
        self.projections_calculated = False

        # Spinner state - what phase after clicking run button are we in
        self.spinner_state = -1
        self.spinner_messages = ["Starting session...", "Loading data...", "Running pipeline...", "Relaunch"]
        # Error message that we want to show, for more details look inside function check error state
        self.error_message = None

    def define_interface(self, **kwargs):
        # File selection and parameters
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### File selection")
                with gr.Row():
                    with gr.Column(scale=99, min_width=128):
                        base_path = gr.Textbox(
                            max_lines=1,
                            container=False,
                            label="Base path",
                            value=os.getcwd(),
                        )
                    with gr.Column(scale=1, min_width=36):
                        reload_base_path = gr.Button(
                            value="⟳"
                        )
                explorer = gr.FileExplorer(
                    ignore_glob="*/.*",  # ignores hidden files
                    root_dir=os.getcwd(),
                    label=os.getcwd(),
                    render=True,
                    file_count="single",
                    interactive=True,
                    height = 320,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Parameters")
                cmap = gr.Dropdown(
                    value="viridis",
                    choices=plt.colormaps(),
                    label="Colormap",
                    interactive=True,
                )

                virtual_stack = gr.Checkbox(
                    value=False,
                    label="Virtual stack",
                    info="If checked, will use less memory by loading the images on demand.",
                )
                load_series = gr.Checkbox(
                    value=False,
                    label="Load series",
                    info="If checked, will load the whole series of images in the same folder as the selected file.",
                )
                series_contains = gr.Textbox(
                    label="Specify common part of file names for series",
                    value="",
                    visible=False,
                )

                dataset_name = gr.Textbox(
                    label="Dataset name (in case of H5 files, for example)",
                    value="exchange/data",
                )

                def toggle_show(checkbox):
                    return (
                        gr.update(visible=True)
                        if checkbox
                        else gr.update(visible=False)
                    )

                # Show series_contains only if load_series is checked
                load_series.change(toggle_show, load_series, series_contains)

            with gr.Column(scale=1):
                gr.Markdown("### Operations")
                operations = gr.CheckboxGroup(
                    choices=self.all_operations,
                    value=[self.all_operations[0], self.all_operations[-1]],
                    label=None,
                    container=False,
                    interactive=True,
                )
                with gr.Row():
                    btn_run = gr.Button(
                        value="Load & Run", variant = "primary",
                    )

        # Visualization and results
        with gr.Row():

            # Z Slicer
            with gr.Column(visible=False) as result_z_slicer:
                zslice_plot = gr.Plot(label="Z slice")
                zpos = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.01, label="Z position"
                )

            # Y Slicer
            with gr.Column(visible=False) as result_y_slicer:
                yslice_plot = gr.Plot(label="Y slice")

                ypos = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.01, label="Y position"
                )

            # X Slicer
            with gr.Column(visible=False) as result_x_slicer:
                xslice_plot = gr.Plot(label="X slice")

                xpos = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.01, label="X position"
                )
            # Z Max projection
            with gr.Column(visible=False) as result_z_max_projection:
                max_projection_plot = gr.Plot(
                    label="Z max projection",
                )

            # Z Min projection
            with gr.Column(visible=False) as result_z_min_projection:
                min_projection_plot = gr.Plot(
                    label="Z min projection",
                )

            # Intensity histogram
            with gr.Column(visible=False) as result_intensity_histogram:
                hist_plot = gr.Plot(label="Volume intensity histogram")

            # Text box with data summary
            with gr.Column(visible=False) as result_data_summary:
                data_summary = gr.Text(
                    lines=24,
                    label=None,
                    show_label=False,

                    value="Data summary",
                )
            ### Gradio objects lists


        ####################################
        #       EVENT LISTENERS
        ###################################
        pipeline_inputs = [operations, zpos, ypos, xpos, cmap]

        pipeline_outputs = [
            zslice_plot,
            yslice_plot,
            xslice_plot,
            max_projection_plot,
            min_projection_plot,
            hist_plot,
            data_summary,
        ]

        results = [
                result_z_slicer,
                result_y_slicer,
                result_x_slicer,
                result_z_max_projection,
                result_z_min_projection,
                result_intensity_histogram,
                result_data_summary,
            ]

        reload_base_path.click(fn=self.update_explorer,inputs=base_path, outputs=explorer)
        
        btn_run.click(
            fn=self.update_run_btn, inputs = [], outputs = btn_run).then(
            fn=self.start_session, inputs = [load_series, series_contains, explorer, base_path], outputs = []).then( 
            fn=self.update_run_btn, inputs = [], outputs = btn_run).then(
            fn=self.check_error_state, inputs = [], outputs = []).success(
            fn=self.load_data, inputs= [virtual_stack, dataset_name, series_contains], outputs= []).then(
            fn=self.update_run_btn, inputs = [], outputs = btn_run).then(
            fn=self.check_error_state, inputs = [], outputs = []).success(
            fn=self.run_operations, inputs = pipeline_inputs, outputs = pipeline_outputs).then(
            fn=self.update_run_btn, inputs = [], outputs = btn_run).then(
            fn=self.check_error_state, inputs = [], outputs = []).success(
            fn=self.show_results, inputs = operations, outputs = results) # results are columns of images and other component, not just the components

        """
        Gradio passes only the value to the function, not the whole component.
        That means we have no information about what slider out of those 3 was
        updated. This way, 3 different functions are created, one for each slider.
        The self.update_slice_wrapper returns a function.
        """
        sliders = [xpos, ypos, zpos]
        letters = ["X", "Y", "Z"]
        plots = [xslice_plot, yslice_plot, zslice_plot]
        for slider, letter, plot in zip(sliders, letters, plots):
            slider.change(fn = self.update_slice_wrapper(letter), inputs = [slider, cmap], outputs = plot, show_progress="hidden")

        
        # Immediate change without the need of pressing the relaunch button
        operations.change(fn=self.show_results, inputs = operations, outputs = results)
        cmap.change(fn=self.run_operations, inputs = pipeline_inputs, outputs = pipeline_outputs)

    def update_explorer(self, new_path):
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

    def update_run_btn(self):
        """
        When run_btn is clicked, it becomes uninteractive and displays which operation is now in progress
        When all operations are done, it becomes interactive again with 'Relaunch' label
        """
        self.spinner_state = (self.spinner_state + 1) % len(self.spinner_messages) if self.error_message is None else len(self.spinner_messages) - 1
        message = self.spinner_messages[self.spinner_state]
        interactive = (self.spinner_state == len(self.spinner_messages) - 1)
        return gr.update(
            value=f"{message}",
            interactive=interactive,
        )

    def check_error_state(self):
        """
        Raising gr.Error doesn't allow us to return anything and thus we can not update the Run button with 
        progress messages. We have to first update the button and then raise an Error so the button is interactive
        """
        if self.error_message is not None:
            error_message = self.error_message
            self.error_message = None
            raise gr.Error(error_message)
        
#######################################################
#
#       THE PIPELINE
#
#######################################################

    def start_session(self, load_series:bool, series_contains:str, explorer:str, base_path:str):
        self.projections_calculated = False # Probably new file was loaded, we would need new projections

        if load_series and series_contains == "":
            # Try to guess the common part of the file names
            try:
                filename = explorer.split("/")[-1]  # Extract filename from path
                series_contains = re.search(r"[^0-9]+", filename).group()
                gr.Info(f"Using '{series_contains}' as common file name part for loading.")
                self.series_contains = series_contains

            except:
                self.error_message = "For series, common part of file name must be provided in 'series_contains' field."
                

        # Get the file path from the explorer or base path
        # priority is given to the explorer if file is selected
        # else the base path is used
        if explorer and (os.path.isfile(explorer) or load_series):
            self.file_path = explorer

        elif base_path and (os.path.isfile(base_path) or load_series):
            self.file_path = base_path

        else:
            self.error_message = "Invalid file path"

        # If we are loading a series, we need to get the directory
        if load_series:
            self.file_path = os.path.dirname(self.file_path)


    def load_data(self, virtual_stack:bool, dataset_name:str, contains:str):
        try:
            self.vol = load(
                path = self.file_path,
                virtual_stack = virtual_stack,
                dataset_name = dataset_name,
                contains = contains
            )

            # Incase the data is 4D (RGB for example), we take the mean of the last dimension
            if self.vol.ndim == 4:
                self.vol = np.mean(self.vol, axis=-1)

            # The rest of the pipeline expects 3D data
            if self.vol.ndim != 3:
                self.error_message = F"Invalid data shape should be 3 dimensional, not shape: {self.vol.shape}"

        except Exception as error_message:
            self.error_message = F"Error when loading data: {error_message}"
    
    def run_operations(self, operations, *args):
        outputs = []
        self.calculated_operations = []
        for operation in self.all_operations:
            if operation in operations:
                log.info(f"Running {operation}")
                try:
                    outputs.append(self.run_operation(operation, *args))
                    self.calculated_operations.append(operation)

                except Exception as err:
                    self.error_message = F"Error while running operation '{operation}': {err}"

                    log.info(self.error_message)
                    outputs.append(gr.update())
            else:
                log.info(f"Skipping {operation}")
                outputs.append(gr.update())  
        
        return outputs

    def run_operation(self, operation:list, zpos:float, ypos:float, xpos:float, cmap:str, *args):
        match operation:
            case "Z Slicer":
                return self.update_slice_wrapper("Z")(zpos, cmap)
            case "Y Slicer":
                return self.update_slice_wrapper("Y")(ypos, cmap)
            case "X Slicer":
                return self.update_slice_wrapper("X")(xpos, cmap)
            case "Z max projection":
                return self.create_projections_figs()[0]
            case "Z min projection":
                return self.create_projections_figs()[1]
            case "Intensity histogram":
                # If the operations are run with the run_button, spinner_state == 2, 
                #   If we just changed cmap, spinner state would be 3 
                #   and we don't have to calculate histogram again
                #   That saves a lot of time as the histogram takes the most time to calculate
                return self.plot_histogram() if self.spinner_state == 2 else gr.update() 
            case "Data summary":
                return self.show_data_summary()
            case _:
                raise NotImplementedError(F"Operation '{operation} is not defined")

    def show_results(self, operations):
        update_list = []
        for operation in self.all_operations:
            if operation in operations and operation in self.calculated_operations:
                update_list.append(gr.update(visible=True))
            else:
                update_list.append(gr.update(visible=False))
        return update_list

#######################################################
#
#       CALCULATION OF IMAGES
#
#######################################################

    def create_img_fig(self, img, **kwargs):
        fig, ax = plt.subplots(figsize=(self.figsize, self.figsize))

        ax.imshow(img, interpolation="nearest", **kwargs)

        # Adjustments
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig

    def update_slice_wrapper(self, letter):
        def update_slice(position_slider:float, cmap:str):
            """
            position_slider: float from gradio slider, saying which relative slice we want to see
            cmap: string gradio drop down menu, saying what cmap we want to use for display
            """
            axis = self.axis_dict[letter]
            slice_index = int(position_slider * (self.vol.shape[axis] - 1))
            
            plt.close()
            plt.set_cmap(cmap)

            if self.min_percentile and self.max_percentile:
                vmin = self.min_percentile
                vmax = self.max_percentile
            else:
                vmin = None
                vmax = None

            # The axis we want to slice along is moved to be the last one, could also be the first one, it doesn't matter
            # Then we take out the slice defined in self.position for this axis
            slice_img = np.moveaxis(self.vol, axis, -1)[:,:,slice_index]

            fig_img = self.create_img_fig(slice_img, vmin = vmin, vmax = vmax)
            
            return gr.update(value = fig_img, label = f"{letter} Slice: {slice_index}", visible = True)
        return update_slice
    
    def vol_histogram(self, nbins, min_value, max_value):
        # Start histogram
        vol_hist = np.zeros(nbins)

        # Iterate over slices
        for zslice in self.vol:
            hist, bin_edges = np.histogram(
                zslice, bins=nbins, range=(min_value, max_value)
            )
            vol_hist += hist

        return vol_hist, bin_edges

    def plot_histogram(self):
        # The Histogram needs results from the projections
        if not self.projections_calculated:
            _ = self.get_projections()

        vol_hist, bin_edges = self.vol_histogram(self.nbins, self.min_value, self.max_value)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(bin_edges[:-1], vol_hist, width=np.diff(bin_edges), ec="white", align="edge")

        # Adjustments
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.set_yscale("log")

        return fig
    
    def create_projections_figs(self):
        if not self.projections_calculated:
            projections = self.get_projections()
            self.max_projection = projections[0]
            self.min_projection = projections[1]

        # Generate figures
        max_projection_fig = self.create_img_fig(
            self.max_projection,
            vmin=self.min_percentile,
            vmax=self.max_percentile,
        )
        min_projection_fig = self.create_img_fig(
            self.min_projection,
            vmin=self.min_percentile,
            vmax=self.max_percentile,
        )

        self.projections_calculated = True
        return max_projection_fig, min_projection_fig

    def get_projections(self):
        # Create arrays for iteration
        max_projection = np.zeros(np.shape(self.vol[0]))
        min_projection = np.ones(np.shape(self.vol[0])) * float("inf")
        intensity_sum = 0

        # Iterate over slices. This is needed in case of virtual stacks.
        for zslice in self.vol:
            max_projection = np.maximum(max_projection, zslice)
            min_projection = np.minimum(min_projection, zslice)
            intensity_sum += np.sum(zslice)

        self.min_value = np.min(min_projection)
        self.min_percentile = np.percentile(
            min_projection, 100 - self.display_saturation_percentile
        )
        self.max_value = np.max(max_projection)
        self.max_percentile = np.percentile(
            max_projection, self.display_saturation_percentile
        )

        self.intensity_sum = intensity_sum

        nvoxels = self.vol.shape[0] * self.vol.shape[1] * self.vol.shape[2]
        self.mean_intensity = intensity_sum / nvoxels

        return max_projection, min_projection

    def show_data_summary(self):
        summary_dict = {
            "Last modified": datetime.datetime.fromtimestamp(os.path.getmtime(self.file_path)).strftime("%Y-%m-%d %H:%M"),
            "File size": misc.sizeof(os.path.getsize(self.file_path)),
            "Z-size": str(self.vol.shape[self.axis_dict["Z"]]),
            "Y-size": str(self.vol.shape[self.axis_dict["Y"]]),
            "X-size": str(self.vol.shape[self.axis_dict["X"]]),
            "Data type": str(self.vol.dtype),
            "Min value": str(self.vol.min()),
            "Mean value": str(np.mean(self.vol)),
            "Max value": str(self.vol.max()),
        }

        display_dict = {k: v for k, v in summary_dict.items() if v is not None}
        return ouf.showdict(display_dict, return_str=True, title="Data summary")
        

if __name__ == "__main__":
    Interface().run_interface()
