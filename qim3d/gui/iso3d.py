import gradio as gr
import numpy as np
import os
from qim3d.utils import internal_tools
from qim3d.io import DataLoader
from qim3d.io.logger import log
import plotly.graph_objects as go
from scipy import ndimage


class Interface:
    def __init__(self):
        self.show_header = False
        self.verbose = False
        self.title = "Isosurfaces for 3D visualization"
        self.interface = None
        self.plot_height = 768
        self.height = 1024
        self.width = 960

        # Data examples
        current_dir = os.path.dirname(os.path.abspath(__file__))
        examples_dir = ["..", "img_examples"]
        examples = [
            "blobs_256x256x256.tif",
            "fly_150x256x256.tif",
            "cement_128x128x128.tif",
            "NT_10x200x100.tif",
            "NT_128x128x128.tif",
            "shell_225x128x128.tif",
            "bone_128x128x128.tif",
        ]
        self.img_examples = []
        for example in examples:
            self.img_examples.append(
                [os.path.join(current_dir, *examples_dir, example)]
            )

        # CSS path
        self.css_path = os.path.join(current_dir, "..", "css", "gradio.css")

    def clear(self):
        """Used to reset the plot with the clear button"""
        return None

    def load_data(self, filepath):
        # TODO: Add support for multiple files
        self.vol = DataLoader().load_tiff(filepath)

    def resize_vol(self):
        """Resizes the loaded volume to the display size"""

        # Get original size
        original_Z, original_Y, original_X = np.shape(self.vol)
        max_size = np.max([original_Z, original_Y, original_X])
        if self.verbose:
            log.info(f"\nOriginal volume: {original_Z, original_Y, original_X}")

        # Resize for display
        self.vol = ndimage.zoom(
            input=self.vol,
            zoom=self.display_size / max_size,
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

    def save_fig(self, fig, filename):
        # Write Plotly figure to disk
        fig.write_html(filename)

    def create_fig(self):
        # Load volume
        self.load_data(self.gradio_file.name)

        # Resize data for display size
        self.resize_vol()

        # Flip Z
        if self.flip_z:
            self.vol = np.flip(self.vol, axis=0)

        # Create 3D grid
        Z, Y, X = np.mgrid[
            0 : self.display_size_z, 0 : self.display_size_y, 0 : self.display_size_x
        ]

        if self.only_wireframe:
            surface_fill = 0.2
        else:
            surface_fill = 1.0

        fig = go.Figure(
            go.Volume(
                z=Z.flatten(),
                y=Y.flatten(),
                x=X.flatten(),
                value=self.vol.flatten(),
                isomin=self.min_value * np.max(self.vol),
                isomax=self.max_value * np.max(self.vol),
                cmin=np.min(self.vol),
                cmax=np.max(self.vol),
                opacity=self.opacity,
                opacityscale=self.opacityscale,
                surface_count=self.surface_count,
                colorscale=self.colormap,
                slices_z=dict(
                    show=self.show_z_slice,
                    locations=[int(self.display_size_z * self.slice_z_location)],
                ),
                slices_y=dict(
                    show=self.show_y_slice,
                    locations=[int(self.display_size_y * self.slice_y_location)],
                ),
                slices_x=dict(
                    show=self.show_x_slice,
                    locations=[int(self.display_size_x * self.slice_x_location)],
                ),
                surface=dict(fill=surface_fill),
                caps=dict(
                    x_show=self.show_caps,
                    y_show=self.show_caps,
                    z_show=self.show_caps,
                ),
                showscale=self.show_colorbar,
                colorbar=dict(
                    thickness=8, outlinecolor="#fff", len=0.5, orientation="h"
                ),
                reversescale=self.reversescale,
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            scene_xaxis_showticklabels=self.show_ticks,
            scene_yaxis_showticklabels=self.show_ticks,
            scene_zaxis_showticklabels=self.show_ticks,
            scene_xaxis_visible=self.show_axis,
            scene_yaxis_visible=self.show_axis,
            scene_zaxis_visible=self.show_axis,
            scene_aspectmode="data",
            height=self.plot_height,
            hovermode=False,
            scene_camera_eye=dict(x=2.0, y=-2.0, z=1.5),
        )

        return fig

    def process(self, *args):
        # Get args passed by Gradio
        # TODO: solve this in an automated way
        # Could Gradio pass kwargs instead of args?
        self.gradio_file = args[0]
        self.display_size = args[1]
        self.opacity = args[2]
        self.opacityscale = args[3]
        self.only_wireframe = args[4]
        self.min_value = args[5]
        self.max_value = args[6]
        self.surface_count = args[7]
        self.colormap = args[8]
        self.show_colorbar = args[9]
        self.reversescale = args[10]
        self.flip_z = args[11]
        self.show_axis = args[12]
        self.show_ticks = args[13]
        self.show_caps = args[14]
        self.show_z_slice = args[15]
        self.slice_z_location = args[16]
        self.show_y_slice = args[17]
        self.slice_y_location = args[18]
        self.show_x_slice = args[19]
        self.slice_x_location = args[20]

        # Create output figure
        fig = self.create_fig()

        # Save it to disk
        self.save_fig(fig, "iso3d.html")

        return fig, "iso3d.html"
    
    def remove_unused_file(self):
        # Remove localthickness.tif file from working directory
        # as it otherwise is not deleted
        os.remove("iso3d.html")

    def create_interface(self):
        # Create gradio app

        with gr.Blocks(css=self.css_path) as gradio_interface:
            if self.show_header:
                gr.Markdown(
                    """
                    # 3D Visualization (isosurfaces)
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
                            show_label=False, elem_classes="file-input h-128"
                        )
                    with gr.Tab("Examples"):
                        gr.Examples(examples=self.img_examples, inputs=gradio_file)

                    # Run button
                    with gr.Row():
                        with gr.Column(scale=3, min_width=64):
                            btn_run = gr.Button(
                                value="Run 3D visualization", elem_classes="btn btn-run"
                            )
                        with gr.Column(scale=1, min_width=64):
                            btn_clear = gr.Button(
                                value="Clear", elem_classes="btn btn-clear"
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
                            elem_classes="",
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

                    with gr.Tab("Slices"):
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
                        elem_classes="w-256",
                    )

                outputs = [volvizplot, plot_download]

            # Session for user data
            session = gr.State([])

            # Listeners

            # Clear button
            for gr_obj in outputs:
                btn_clear.click(fn=self.clear, inputs=None, outputs=gr_obj)
            # Run button
            # fmt: off
            btn_run.click(
                fn=self.process, inputs=inputs, outputs=outputs).success(
                fn=self.remove_unused_file).success(
                fn=self.make_visible, inputs=None, outputs=plot_download)
            # fmt: on

        return gradio_interface

    def make_visible(self):
        return gr.update(visible=True)

    def launch(self, **kwargs):
        # Show header
        if self.show_header:
            internal_tools.gradio_header(self.title, self.port)

        # Create gradio interface
        self.interface = self.create_interface()

        # Set gradio verbose level
        if self.verbose:
            quiet = False
        else:
            quiet = True

        self.interface.launch(
            quiet=quiet,
            height=self.height,
            width=self.width,
            **kwargs,
        )


def run_interface(host = "0.0.0.0"):
    gradio_interface = Interface().create_interface()
    internal_tools.run_gradio_app(gradio_interface,host)


if __name__ == "__main__":
    # Creates interface
    run_interface()