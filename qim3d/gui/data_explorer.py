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
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import outputformat as ouf
import tifffile
from pathlib import Path

import qim3d
from qim3d.io import load
from qim3d.io.logger import log
from qim3d.utils import internal_tools
from .qim_theme import QimTheme


class Interface:
    def __init__(self):
        self.show_header = False
        self.verbose = False
        self.title = "Data Explorer"
        self.height = 1024
        self.width = 900
        self.operations = [
            "Z Slicer",
            "Y Slicer",
            "X Slicer",
            "Z max projection",
            "Z min projection",
            "Intensity histogram",
            "Data summary",
        ]

    def clear(self):
        """Used to reset outputs with the clear button"""
        return None

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

    def set_visible(self):
        return gr.update(visible=True)

    def set_spinner(self, message):
        return gr.update(
            value=f"{message}",
            interactive=False,
        )

    def set_relaunch_button(self):
        return gr.update(
            value=f"Relaunch",
            interactive=True,
        )

    def show_results(self, operations):
        update_list = []
        for operation in self.operations:
            if operation in operations:
                update_list.append(gr.update(visible=True))
            else:
                update_list.append(gr.update(visible=False))
        return update_list

    def create_interface(self, force_light_mode:bool = True):
        with gr.Blocks(theme = QimTheme(force_light_mode=force_light_mode), 
                       title = self.title) as gradio_interface:
            gr.Markdown("# Data Explorer")

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
                        choices=self.operations,
                        value=[self.operations[0], self.operations[-1]],
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

            session = gr.State([])
            pipeline = Pipeline()

            # Results
            results = [
                result_z_slicer,
                result_y_slicer,
                result_x_slicer,
                result_z_max_projection,
                result_z_min_projection,
                result_intensity_histogram,
                result_data_summary,
            ]
            # Inputs
            inputs = [
                operations,
                base_path,
                explorer,
                zpos,
                ypos,
                xpos,
                cmap,
                dataset_name,
                virtual_stack,
                load_series,
                series_contains,
            ]
            # Outputs
            outputs = [
                zslice_plot,
                yslice_plot,
                xslice_plot,
                max_projection_plot,
                min_projection_plot,
                hist_plot,
                data_summary,
            ]

            ### Listeners
            spinner_session = gr.Text("Starting session...", visible=False)
            spinner_loading = gr.Text("Loading data...", visible=False)
            spinner_operations = gr.Text("Running pipeline...", visible=False)
            # fmt: off
            reload_base_path.click(fn=self.update_explorer,inputs=base_path, outputs=explorer)
            
            btn_run.click(
                fn=self.set_spinner, inputs=spinner_session, outputs=btn_run).then(
                fn=self.start_session, inputs=inputs, outputs=session).then(
                fn=self.set_spinner, inputs=spinner_loading, outputs=btn_run).then(
                fn=pipeline.load_data, inputs=session, outputs=session).then(
                fn=self.set_spinner, inputs=spinner_operations, outputs=btn_run).then(
                fn=pipeline.run_pipeline, inputs=session, outputs=outputs).then(
                fn=self.show_results, inputs=operations, outputs=results).then(
                fn=self.set_relaunch_button, inputs=[], outputs=btn_run)

                

            zpos.change(
                fn=self.update_zpos, inputs=[session, zpos], outputs=[session, zslice_plot]).success(
                fn=pipeline.create_zslice_fig, inputs=[], outputs=zslice_plot,show_progress="hidden")
            ypos.change(
                fn=self.update_ypos, inputs=[session, ypos], outputs=[session, yslice_plot]).success(
                fn=pipeline.create_yslice_fig, inputs=[], outputs=yslice_plot,show_progress="hidden")
            
            xpos.change(
                fn=self.update_xpos, inputs=[session, xpos], outputs=[session, xslice_plot]).success(
                fn=pipeline.create_xslice_fig, inputs=[], outputs=xslice_plot,show_progress="hidden")

            # fmt: on

        return gradio_interface

    def start_session(self, *args):
        # Starts a new session dictionary
        session = Session()
        # Tells rest of the pipeline if the session failed at some point prior, so should skip the rest
        session.failed = False
        session.all_operations = Interface().operations
        session.operations = args[0]
        session.base_path = args[1]
        session.explorer = args[2]
        session.zpos = args[3]
        session.ypos = args[4]
        session.xpos = args[5]
        session.cmap = args[6]
        session.dataset_name = args[7]
        session.virtual_stack = args[8]
        session.load_series = args[9]
        session.series_contains = args[10]

        if session.load_series and session.series_contains == "":

            # Try to guess the common part of the file names
            try:
                filename = session.explorer.split("/")[-1]  # Extract filename from path
                series_contains = re.search(r"[^0-9]+", filename).group()
                gr.Info(f"Using '{series_contains}' as common file name part for loading.")
                session.series_contains = series_contains

            except:
                session.failed = True
                raise gr.Error(
                    "For series, common part of file name must be provided in 'series_contains' field."
                )

        # Get the file path from the explorer or base path
        # priority is given to the explorer if file is selected
        # else the base path is used
        if session.explorer and (
            os.path.isfile(session.explorer) or session.load_series
        ):
            session.file_path = session.explorer
        elif session.base_path and (
            os.path.isfile(session.base_path) or session.load_series
        ):
            session.file_path = session.base_path
        else:
            session.failed = True
            raise gr.Error("Invalid file path")

        # If we are loading a series, we need to get the directory
        if session.load_series:
            session.file_path = os.path.dirname(session.file_path)

        return session

    def update_zpos(self, session, zpos):
        session.zpos = zpos
        session.zslice_from_zpos()

        return session, gr.update(label=f"Z slice: {session.zslice}")

    def update_ypos(self, session, ypos):
        session.ypos = ypos
        session.yslice_from_ypos()

        return session, gr.update(label=f"Y slice: {session.yslice}")

    def update_xpos(self, session, xpos):
        session.xpos = xpos
        session.xslice_from_xpos()

        return session, gr.update(label=f"X slice: {session.xslice}")

    def launch(self,force_light_mode:bool = True, **kwargs):
        # Show header
        if self.show_header:
            internal_tools.gradio_header(self.title, self.port)

        # Create gradio interfaces 
        interface = self.create_interface(force_light_mode=force_light_mode)

        # Set gradio verbose level
        if self.verbose:
            quiet = False
        else:
            quiet = True

        interface.launch(
            quiet=quiet,
            height=self.height,
            width=self.width,
            favicon_path = Path(qim3d.__file__).parents[0] / "gui/images/qim_platform-icon.svg",
            **kwargs,
        )


class Session:
    def __init__(self):
        self.virtual_stack = False
        self.file_path = None
        self.failed = None
        self.vol = None
        self.zpos = 0.5
        self.ypos = 0.5
        self.xpos = 0.5
        self.cmap = "viridis"
        self.dataset_name = None
        self.error_message = None
        self.file_path = None
        self.max_projection = None
        self.min_projection = None
        self.projections_calculated = False
        # Volume info
        self.zsize = None
        self.ysize = None
        self.xsize = None
        self.data_type = None
        self.axes = None
        self.last_modified = None
        self.file_size = None
        self.min_percentile = None
        self.max_percentile = None
        self.min_value = None
        self.max_value = None
        self.intensity_sum = None
        self.mean_intensity = None

        # Histogram
        self.nbins = 32

    def zslice_from_zpos(self):
        self.zslice = int(self.zpos * (self.zsize - 1))

        return self.zslice

    def yslice_from_ypos(self):
        self.yslice = int(self.ypos * (self.ysize - 1))

        return self.yslice

    def xslice_from_xpos(self):
        self.xslice = int(self.xpos * (self.xsize - 1))

        return self.xslice


class Pipeline:
    def __init__(self):
        self.figsize = 8  # Used for matplotlig figure size
        self.display_saturation_percentile = 99
        self.verbose = False
        self.session = None

    def load_data(self, session):
        # skip loading if session failed at some point prior
        if session.failed:
            return session

        try:
            session.vol = load(
                session.file_path,
                virtual_stack=session.virtual_stack,
                dataset_name=session.dataset_name,
                contains=session.series_contains,
            )

            # Incase the data is 4D (RGB for example), we take the mean of the last dimension
            if session.vol.ndim == 4:
                session.vol = np.mean(session.vol, axis=-1)

            # The rest of the pipeline expects 3D data
            if session.vol.ndim != 3:
                raise gr.Error(
                    f"Invalid data shape should be 3 dimensional, not shape: {session.vol.shape}"
                )
        except Exception as error_message:
            raise gr.Error(
                f"Failed to load the image: {error_message}"
            ) from error_message

        session = self.get_data_info(session)

        return session

    def get_data_info(self, session):
        first_slice = session.vol[0]

        # Get info
        session.zsize = len(session.vol)
        session.ysize, session.xsize = first_slice.shape
        session.data_type = str(first_slice.dtype)
        session.last_modified = datetime.datetime.fromtimestamp(
            os.path.getmtime(session.file_path)
        ).strftime("%Y-%m-%d %H:%M")
        session.file_size = os.path.getsize(session.file_path)

        return session

    def run_pipeline(self, session):
        # skip loading if session failed at some point prior
        if session.failed:
            return []

        self.session = session
        outputs = []
        log.info(session.all_operations)
        for operation in session.all_operations:
            if operation in session.operations:
                outputs.append(self.run_operation(operation))

            else:
                log.info(f"Skipping {operation}")
                outputs.append(None)

        return outputs

    def run_operation(self, operation):
        log.info(f"Running {operation}")

        if operation == "Data summary":
            return self.show_data_summary()

        if operation == "Z Slicer":
            return self.create_zslice_fig()

        if operation == "Y Slicer":
            return self.create_yslice_fig()

        if operation == "X Slicer":
            return self.create_xslice_fig()

        if operation == "Z max projection":
            return self.create_projections_figs()[0]

        if operation == "Z min projection":
            return self.create_projections_figs()[1]

        if operation == "Intensity histogram":
            return self.plot_vol_histogram()

        # In case nothing was triggered, raise error
        raise ValueError("Unknown operation")

    def show_data_summary(self):
        # Get info from Tiff file

        summary_dict = {
            "Last modified": self.session.last_modified,
            "File size": internal_tools.sizeof(self.session.file_size),
            "Z-size": str(self.session.zsize),
            "Y-size": str(self.session.ysize),
            "X-size": str(self.session.xsize),
            "Data type": self.session.data_type,
            "Min value": self.session.min_value,
            "Mean value": self.session.mean_intensity,
            "Max value": self.session.max_value,
        }

        display_dict = {k: v for k, v in summary_dict.items() if v is not None}
        return ouf.showdict(display_dict, return_str=True, title="Data summary")

    def create_zslice_fig(self):
        slice_fig = self.create_slice_fig("z")

        return slice_fig

    def create_yslice_fig(self):
        slice_fig = self.create_slice_fig("y")

        return slice_fig

    def create_xslice_fig(self):
        slice_fig = self.create_slice_fig("x")

        return slice_fig

    def create_slice_fig(self, axis):
        plt.close()
        vol = self.session.vol
        plt.set_cmap(self.session.cmap)

        zslice = self.session.zslice_from_zpos()
        yslice = self.session.yslice_from_ypos()
        xslice = self.session.xslice_from_xpos()

        # Check if we something to use as vmin and vmax
        if self.session.min_percentile and self.session.max_percentile:
            vmin = self.session.min_percentile
            vmax = self.session.max_percentile
        else:
            vmin = None
            vmax = None

        if axis == "z":
            slice_fig = self._zslice_fig(vol, zslice, vmin=vmin, vmax=vmax)
        if axis == "y":
            slice_fig = self._yslice_fig(vol, yslice, vmin=vmin, vmax=vmax)
        if axis == "x":
            slice_fig = self._xslice_fig(vol, xslice, vmin=vmin, vmax=vmax)

        return slice_fig

    def _zslice_fig(self, vol, slice, **kwargs):
        fig = self.create_img_fig(vol[slice, :, :], **kwargs)

        return fig

    def _yslice_fig(self, vol, slice, **kwargs):
        fig = self.create_img_fig(vol[:, slice, :], **kwargs)

        return fig

    def _xslice_fig(self, vol, slice, **kwargs):
        fig = self.create_img_fig(vol[:, :, slice], **kwargs)

        return fig

    def create_img_fig(self, img, **kwargs):
        fig, ax = plt.subplots(figsize=(self.figsize, self.figsize))

        ax.imshow(img, interpolation="nearest", **kwargs)

        # Adjustments
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig

    def create_projections_figs(self):
        vol = self.session.vol

        if not self.session.projections_calculated:
            projections = self.get_projections(vol)
            self.session.max_projection = projections[0]
            self.session.min_projection = projections[1]

        # Generate figures
        max_projection_fig = self.create_img_fig(
            self.session.max_projection,
            vmin=self.session.min_percentile,
            vmax=self.session.max_percentile,
        )
        min_projection_fig = self.create_img_fig(
            self.session.min_projection,
            vmin=self.session.min_percentile,
            vmax=self.session.max_percentile,
        )

        self.session.projections_calculated = True
        return max_projection_fig, min_projection_fig

    def get_projections(self, vol):
        # Create arrays for iteration
        max_projection = np.zeros(np.shape(vol[0]))
        min_projection = np.ones(np.shape(vol[0])) * float("inf")
        intensity_sum = 0

        # Iterate over slices. This is needed in case of virtual stacks.
        for zslice in vol:
            max_projection = np.maximum(max_projection, zslice)
            min_projection = np.minimum(min_projection, zslice)
            intensity_sum += np.sum(zslice)

        self.session.min_value = np.min(min_projection)
        self.session.min_percentile = np.percentile(
            min_projection, 100 - self.display_saturation_percentile
        )
        self.session.max_value = np.max(max_projection)
        self.session.max_percentile = np.percentile(
            max_projection, self.display_saturation_percentile
        )

        self.session.intensity_sum = intensity_sum

        nvoxels = self.session.zsize * self.session.ysize * self.session.xsize
        self.session.mean_intensity = intensity_sum / nvoxels

        return max_projection, min_projection

    def plot_vol_histogram(self):

        # The Histogram needs results from the projections
        if not self.session.projections_calculated:
            _ = self.get_projections(self.session.vol)

        vol_hist, bin_edges = self.vol_histogram(
            self.session.vol,
            self.session.nbins,
            self.session.min_value,
            self.session.max_value,
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(
            bin_edges[:-1], vol_hist, width=np.diff(bin_edges), ec="white", align="edge"
        )

        # Adjustments
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.set_yscale("log")

        # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig

    def vol_histogram(self, vol, nbins, min_value, max_value):
        # Start histogram
        vol_hist = np.zeros(nbins)

        # Iterate over slices
        for zslice in vol:
            hist, bin_edges = np.histogram(
                zslice, bins=nbins, range=(min_value, max_value)
            )
            vol_hist += hist

        return vol_hist, bin_edges


def run_interface(host="0.0.0.0"):
    gradio_interface = Interface().create_interface()
    internal_tools.run_gradio_app(gradio_interface, host)


if __name__ == "__main__":
    # Creates interface
    run_interface()
