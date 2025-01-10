"""
The GUI can be launched directly from the command line:

```bash
qim3d gui --iso3d
```

Or launched from a python script

```python
import qim3d

app = qim3d.gui.iso3d.Interface()
app.launch()
```

"""
import os

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from scipy import ndimage

import qim3d
from qim3d.utils._logger import log
from qim3d.gui.interface import InterfaceWithExamples


#TODO img in launch should be self.img
class Interface(InterfaceWithExamples):
    def __init__(self,
                 verbose:bool = False,
                 plot_height:int = 768,
                 img = None):
        
        super().__init__(title = "Isosurfaces for 3D visualization",
                         height = 1024,
                         width = 960,
                         verbose = verbose)

        self.interface = None
        self.img = img
        self.plot_height = plot_height

    def load_data(self, gradiofile: gr.File):
        try:
            self.vol = qim3d.io.load(gradiofile.name)
            assert self.vol.ndim == 3
        except AttributeError:
            raise gr.Error("You have to select a file")
        except ValueError:
            raise gr.Error("Unsupported file format")
        except AssertionError:
            raise gr.Error(F"File has to be 3D structure. Your structure has {self.vol.ndim} dimension{'' if self.vol.ndim == 1 else 's'}")

    def resize_vol(self, display_size: int):
        """Resizes the loaded volume to the display size"""

        # Get original size
        original_Z, original_Y, original_X = np.shape(self.vol)
        max_size = np.max([original_Z, original_Y, original_X])
        if self.verbose:
            log.info(f"\nOriginal volume: {original_Z, original_Y, original_X}")

        # Resize for display
        self.vol = ndimage.zoom(
            input=self.vol,
            zoom = display_size / max_size,
            order=0,
            prefilter=False,
        )

        self.display_size_z, self.display_size_y, self.display_size_x = np.shape(
            self.vol
        )
        if self.verbose:
            log.info(
                f"Resized volume: {self.display_size_z, self.display_size_y, self.display_size_x}"
            )

    def save_fig(self, fig: go.Figure, filename: str):
        # Write Plotly figure to disk
        fig.write_html(filename)

    def create_fig(self, 
        gradio_file: gr.File,
        display_size: int ,
        opacity: float,
        opacityscale: str,
        only_wireframe: bool,
        min_value: float,
        max_value: float,
        surface_count: int,
        colormap: str,
        show_colorbar: bool,
        reversescale: bool,
        flip_z: bool,
        show_axis: bool,
        show_ticks: bool,
        show_caps: bool,
        show_z_slice: bool,
        slice_z_location: int,
        show_y_slice: bool,
        slice_y_location: int,
        show_x_slice: bool,
        slice_x_location: int,
        ) -> tuple[go.Figure, str]:

        # Load volume
        self.load_data(gradio_file)

        # Resize data for display size
        self.resize_vol(display_size)

        # Flip Z
        if flip_z:
            self.vol = np.flip(self.vol, axis=0)

        # Create 3D grid
        Z, Y, X = np.mgrid[
            0 : self.display_size_z, 0 : self.display_size_y, 0 : self.display_size_x
        ]

        if only_wireframe:
            surface_fill = 0.2
        else:
            surface_fill = 1.0

        fig = go.Figure(
            go.Volume(
                z = Z.flatten(),
                y = Y.flatten(),
                x = X.flatten(),
                value = self.vol.flatten(),
                isomin = min_value * np.max(self.vol),
                isomax = max_value * np.max(self.vol),
                cmin = np.min(self.vol),
                cmax = np.max(self.vol),
                opacity = opacity,
                opacityscale = opacityscale,
                surface_count = surface_count,
                colorscale = colormap,
                slices_z = dict(
                    show = show_z_slice,
                    locations = [int(self.display_size_z * slice_z_location)],
                ),
                slices_y = dict(
                    show = show_y_slice,
                    locations=[int(self.display_size_y * slice_y_location)],
                ),
                slices_x = dict(
                    show = show_x_slice,
                    locations = [int(self.display_size_x * slice_x_location)],
                ),
                surface = dict(fill=surface_fill),
                caps = dict(
                    x_show = show_caps,
                    y_show = show_caps,
                    z_show = show_caps,
                ),
                showscale = show_colorbar,
                colorbar=dict(
                    thickness=8, outlinecolor="#fff", len=0.5, orientation="h"
                ),
                reversescale = reversescale,
                hoverinfo = "skip",
            )
        )

        fig.update_layout(
            scene_xaxis_showticklabels = show_ticks,
            scene_yaxis_showticklabels = show_ticks,
            scene_zaxis_showticklabels = show_ticks,
            scene_xaxis_visible = show_axis,
            scene_yaxis_visible = show_axis,
            scene_zaxis_visible = show_axis,
            scene_aspectmode="data",
            height=self.plot_height,
            hovermode=False,
            scene_camera_eye=dict(x=2.0, y=-2.0, z=1.5),
        )

        filename = "iso3d.html"
        self.save_fig(fig, filename)

        return fig, filename
    
    def remove_unused_file(self):
        # Remove localthickness.tif file from working directory
        # as it otherwise is not deleted
        os.remove("iso3d.html")

    def define_interface(self, **kwargs):

        gr.Markdown(
                """
                This tool uses Plotly Volume (https://plotly.com/python/3d-volume-plots/) to create iso surfaces from voxels based on their intensity levels.
                To optimize performance when generating visualizations, set the number of voxels (_display resolution_) and isosurfaces (_total surfaces_) to lower levels. 
                """
            )

        with gr.Row():
            # Input and parameters column
            with gr.Column(scale=1, min_width=320):
                with gr.Tab("Input"):
                    # File loader
                    gradio_file = gr.File(
                        show_label=False
                    )
                with gr.Tab("Examples"):
                    gr.Examples(examples=self.img_examples, inputs=gradio_file)

                # Run button
                with gr.Row():
                    with gr.Column(scale=3, min_width=64):
                        btn_run = gr.Button(
                            value="Run 3D visualization", variant = "primary"
                        )
                    with gr.Column(scale=1, min_width=64):
                        btn_clear = gr.Button(
                            value="Clear", variant = "stop"
                        )

                with gr.Tab("Display"):
                    # Display options

                    display_size = gr.Slider(
                        32,
                        128,
                        step=4,
                        label="Display resolution",
                        info="Number of voxels for the largest dimension",
                        value=64,
                    )
                    surface_count = gr.Slider(
                        2, 16, step=1, label="Total iso-surfaces", value=6
                    )

                    show_caps = gr.Checkbox(value=False, label="Show surface caps")

                    with gr.Row():
                        opacityscale = gr.Dropdown(
                            choices=["uniform", "extremes", "min", "max"],
                            value="uniform",
                            label="Opacity scale",
                            info="Handles opacity acording to voxel value",
                        )

                        opacity = gr.Slider(
                            0.0, 1.0, step=0.1, label="Max opacity", value=0.4
                        )
                    with gr.Row():
                        min_value = gr.Slider(
                            0.0, 1.0, step=0.05, label="Min value", value=0.1
                        )
                        max_value = gr.Slider(
                            0.0, 1.0, step=0.05, label="Max value", value=1
                        )

                with gr.Tab("Slices") as slices:
                    show_z_slice = gr.Checkbox(value=False, label="Show Z slice")
                    slice_z_location = gr.Slider(
                        0.0, 1.0, step=0.05, value=0.5, label="Position"
                    )

                    show_y_slice = gr.Checkbox(value=False, label="Show Y slice")
                    slice_y_location = gr.Slider(
                        0.0, 1.0, step=0.05, value=0.5, label="Position"
                    )

                    show_x_slice = gr.Checkbox(value=False, label="Show X slice")
                    slice_x_location = gr.Slider(
                        0.0, 1.0, step=0.05, value=0.5, label="Position"
                    )

                with gr.Tab("Misc"):
                    with gr.Row():
                        colormap = gr.Dropdown(
                            choices=[
                                "Blackbody",
                                "Bluered",
                                "Blues",
                                "Cividis",
                                "Earth",
                                "Electric",
                                "Greens",
                                "Greys",
                                "Hot",
                                "Jet",
                                "Magma",
                                "Picnic",
                                "Portland",
                                "Rainbow",
                                "RdBu",
                                "Reds",
                                "Viridis",
                                "YlGnBu",
                                "YlOrRd",
                            ],
                            value="Magma",
                            label="Colormap",
                        )

                        show_colorbar = gr.Checkbox(
                            value=False, label="Show color scale"
                        )
                        reversescale = gr.Checkbox(
                            value=False, label="Reverse color scale"
                        )
                    flip_z = gr.Checkbox(value=True, label="Flip Z axis")
                    show_axis = gr.Checkbox(value=True, label="Show axis")
                    show_ticks = gr.Checkbox(value=False, label="Show ticks")
                    only_wireframe = gr.Checkbox(
                        value=False, label="Only wireframe"
                    )

                # Inputs for gradio
                inputs = [
                    gradio_file,
                    display_size,
                    opacity,
                    opacityscale,
                    only_wireframe,
                    min_value,
                    max_value,
                    surface_count,
                    colormap,
                    show_colorbar,
                    reversescale,
                    flip_z,
                    show_axis,
                    show_ticks,
                    show_caps,
                    show_z_slice,
                    slice_z_location,
                    show_y_slice,
                    slice_y_location,
                    show_x_slice,
                    slice_x_location,
                ]

            # Output display column
            with gr.Column(scale=4):
                volvizplot = gr.Plot(show_label=False)

                plot_download = gr.File(
                    interactive=False,
                    label="Download interactive plot",
                    show_label=True,
                    visible=False,
                )

            outputs = [volvizplot, plot_download]

        #####################################
        # Listeners
        #####################################

        # Clear button
        for gr_obj in outputs:
            btn_clear.click(fn=self.clear, inputs=None, outputs=gr_obj)
        # Run button
        # fmt: off
        btn_run.click(
            fn=self.create_fig, inputs = inputs, outputs = outputs).success(
            fn=self.remove_unused_file).success(
            fn=self.set_visible, inputs=None, outputs=plot_download)

if __name__ == "__main__":
    Interface().run_interface()