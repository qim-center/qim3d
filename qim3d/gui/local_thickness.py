import gradio as gr
import numpy as np
import os
from qim3d.tools import internal_tools
from qim3d.io import DataLoader
import tifffile
import plotly.express as px
from scipy import ndimage
import outputformat as ouf
import plotly.graph_objects as go
import localthickness as lt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Interface:
    def __init__(self):
        self.show_header = False
        self.verbose = False
        self.title = "Local thickness"
        self.plot_height = 768
        self.height = 1024
        self.width = 960

        # Data examples
        current_dir = os.path.dirname(os.path.abspath(__file__))
        examples_dir = ["..", "..", "img_examples"]
        examples = [
            "blobs_256x256x256.tif",
            "cement_128x128x128.tif",
            "bone_128x128x128.tif",
            "NT_10x200x100.tif",
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

    def make_visible(self):
        return gr.update(visible=True)

    def start_session(self, *args):
        session = Session()
        session.verbose = self.verbose
        session.interface = "gradio"

        # Get the args passed by gradio
        session.data = args[0]
        session.lt_scale = args[1]
        session.threshold = args[2]
        session.dark_objects = args[3]
        session.flip_z = args[4]
        session.nbins = args[5]
        session.display_size_input = args[6]
        session.surface_count_input = args[7]
        session.display_size_output = args[8]
        session.surface_count_output = args[9]
        session.reversescale = args[10]
        session.show_caps = args[11]

        return session

    def launch(self, **kwargs):
        # Show header
        if self.show_header:
            internal_tools.gradio_header(self.title, self.port)

        # Create gradio interfaces
        interface = self.create_interface()

        # Set gradio verbose level
        if self.verbose:
            quiet = False
        else:
            quiet = True

        interface.launch(
            quiet=quiet,
            height=self.height,
            width=self.width,
            **kwargs,
        )

    def create_interface(self):
        with gr.Blocks(css=self.css_path) as gradio_interface:
            gr.Markdown(
                "# 3D Local thickness \n Interface for _Fast local thickness in 3D and 2D_ (https://github.com/vedranaa/local-thickness)"
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    with gr.Tab("Input"):
                        data = gr.File(
                            show_label=False, elem_classes="file-input h-128"
                        )
                    with gr.Tab("Examples"):
                        gr.Examples(examples=self.img_examples, inputs=data)

                    # Run button
                    with gr.Row():
                        with gr.Column(scale=3, min_width=64):
                            btn = gr.Button(
                                "Run local thickness", elem_classes="btn btn-run"
                            )
                        with gr.Column(scale=1, min_width=64):
                            btn_clear = gr.Button("Clear", elem_classes="btn btn-clear")

                    with gr.Tab("Parameters"):
                        gr.Markdown(
                            "It is possible to scale down the image before processing. Lower values will make the algorithm run faster, but decreases the accuracy of results."
                        )
                        lt_scale = gr.Slider(
                            0.1, 1.0, label="Scale", value=0.5, step=0.1
                        )

                        with gr.Row():
                            threshold = gr.Slider(
                                0.0,
                                1.0,
                                value=0.5,
                                step=0.05,
                                label="Threshold",
                                info="Local thickness uses a binary image, so a threshold value is needed.",
                            )

                        dark_objects = gr.Checkbox(value=False, label="Dark objects")

                    with gr.Tab("Display"):
                        with gr.Row():
                            gr.Markdown("Input display")

                            display_size_input = gr.Slider(
                                16,
                                64,
                                step=4,
                                label="Display resolution",
                                info="Number of voxels for the largest dimension",
                                value=32,
                            )

                            surface_count_input = gr.Slider(
                                2, 16, step=1, label="Total iso-surfaces", value=4
                            )
                        with gr.Row():
                            gr.Markdown("Output display")

                            display_size_output = gr.Slider(
                                32,
                                128,
                                step=4,
                                label="Display resolution",
                                info="Number of voxels for the largest dimension",
                                value=64,
                            )

                            surface_count_output = gr.Slider(
                                2, 16, step=1, label="Total iso-surfaces", value=12
                            )

                        reversescale = gr.Checkbox(
                            value=False, label="Reverse color scale"
                        )

                        show_caps = gr.Checkbox(value=True, label="Show surface caps")

                        flip_z = gr.Checkbox(value=True, label="Flip Z axis")

                        gr.Markdown("Thickness histogram options")
                        nbins = gr.Slider(
                            5, 50, value=25, step=1, label="Histogram bins"
                        )

                    inputs = [
                        data,
                        lt_scale,
                        threshold,
                        dark_objects,
                        flip_z,
                        nbins,
                        display_size_input,
                        surface_count_input,
                        display_size_output,
                        surface_count_output,
                        reversescale,
                        show_caps,
                    ]

                with gr.Column(scale=4):
                    with gr.Row():
                        with gr.Column(min_width=256):
                            input_vol = gr.Plot(
                                show_label=True, label="Original volume", visible=True
                            )

                        with gr.Column(min_width=256):
                            binary_vol = gr.Plot(
                                show_label=True, label="Binary volume", visible=True
                            )

                    with gr.Row():
                        with gr.Column(min_width=256):
                            output_vol = gr.Plot(
                                show_label=True,
                                label="Local thickness volume",
                                visible=True,
                            )
                        with gr.Column(min_width=256):
                            histogram = gr.Plot(
                                show_label=True,
                                label="Thickness histogram",
                                visible=True,
                            )
                    with gr.Row():
                        with gr.Column():
                            lt_output = gr.File(
                                interactive=False,
                                show_label=True,
                                label="Output file",
                                visible=False,
                                elem_classes="w-320",
                            )

            # Pipelines
            pipeline = Pipeline()
            pipeline.verbose = self.verbose

            # Session
            session = gr.State([])

            # Ouput gradio objects
            outputs = [input_vol, output_vol, binary_vol, histogram, lt_output]

            # Clear button
            for gr_obj in outputs:
                btn_clear.click(fn=self.clear, inputs=None, outputs=gr_obj)

            # Run button
            # fmt: off
            btn.click(
                fn=self.start_session, inputs=inputs, outputs=session).success(
                fn=pipeline.process_input, inputs=session, outputs=session).success(
                fn=pipeline.prepare_volume, inputs=session,outputs=session).success(
                fn=pipeline.input_viz, inputs=session, outputs=input_vol).success(
                fn=pipeline.make_binary, inputs=session, outputs=session).success(
                fn=pipeline.binary_viz, inputs=session, outputs=binary_vol).success(
                fn=pipeline.compute_localthickness, inputs=session, outputs=session).success(
                fn=pipeline.prepare_output_for_display, inputs=session, outputs=session).success(
                fn=pipeline.output_viz, inputs=session, outputs=output_vol).success(
                fn=pipeline.thickness_histogram, inputs=session, outputs=histogram).success(
                fn=pipeline.save_lt, inputs=session, outputs=lt_output).success(
                fn=self.make_visible, inputs=None, outputs=lt_output)

            # fmt: on

        return gradio_interface


class Session:
    def __init__(self):
        self.interface = None
        self.verbose = None
        self.show_ticks = False
        self.show_axis = True

        # Args from gradio
        self.data = None
        self.vol = None
        self.lt_scale = None
        self.threshold = 0.5
        self.dark_objects = False
        self.flip_z = True
        self.nbins = 25
        self.display_size_input = 32
        self.surface_count_input = 4
        self.display_size_output = 64
        self.surface_count_output = 12
        self.reversescale = False
        self.show_caps = True

        # From pipeline
        self.vol_input_display = None
        self.input_display_size_z = None
        self.input_display_size_y = None
        self.input_display_size_x = None
        self.Zgrid = None
        self.Ygrid = None
        self.Xgrid = None
        self.vol_binary = None
        self.vol_binary_display = None
        self.vol_thickness = None
        self.output_display_size_z = None
        self.output_display_size_y = None
        self.output_display_size_x = None
        self.Zgrid_output = None
        self.Ygrid_output = None
        self.Xgrid_output = None

    def get_vol_info(self):
        self.original_shape = np.shape(self.vol)
        self.original_Z = self.original_shape[0]
        self.original_Y = self.original_shape[1]
        self.original_X = self.original_shape[2]
        self.max_size = np.max(self.original_shape)

        if self.verbose:
            print(f"Original volume shape:{self.original_shape}")
            print(f"Original Z: {self.original_Z}")
            print(f"Original Y: {self.original_Y}")
            print(f"Original X: {self.original_X}")
            print(f"Max size: {self.max_size}")


class Pipeline:
    def process_input(self, session):
        # Load volume
        session.vol = DataLoader().load(session.data.name)

        if session.dark_objects:
            session.vol = np.invert(session.vol)

        if session.flip_z:
            session.vol = np.flip(session.vol, axis=0)

        return session

    def prepare_volume(self, session):
        # Get volume shape
        session.get_vol_info()

        # Resize for display
        session.vol_input_display = ndimage.zoom(
            input=session.vol,
            zoom=(session.display_size_input / session.max_size),
            prefilter=False,
            order=0,
        )

        display_shape = np.shape(session.vol_input_display)
        session.input_display_size_z = display_shape[0]
        session.input_display_size_y = display_shape[1]
        session.input_display_size_x = display_shape[2]

        # Create 3D grid
        session.Zgrid, session.Ygrid, session.Xgrid = np.mgrid[
            0 : session.input_display_size_z,
            0 : session.input_display_size_y,
            0 : session.input_display_size_x,
        ]

        return session

    def input_viz(self, session):
        # Generate input visualization
        data = go.Volume(
            z=session.Zgrid.flatten(),
            y=session.Ygrid.flatten(),
            x=session.Xgrid.flatten(),
            value=session.vol_input_display.flatten(),
            opacity=0.3,
            isomin=0.05 * np.max(session.vol_input_display),
            isomax=1.0 * np.max(session.vol_input_display),
            cmin=np.min(session.vol_input_display),
            cmax=np.max(session.vol_input_display),
            opacityscale="uniform",
            surface_count=session.surface_count_input,
            caps=dict(
                x_show=session.show_caps,
                y_show=session.show_caps,
                z_show=session.show_caps,
            ),
            showscale=False,
            reversescale=session.reversescale,
        )

        fig = go.Figure(data)
        fig.update_layout(
            scene_aspectmode="data",
            scene_xaxis_showticklabels=session.show_ticks,
            scene_yaxis_showticklabels=session.show_ticks,
            scene_zaxis_showticklabels=session.show_ticks,
            scene_xaxis_visible=session.show_axis,
            scene_yaxis_visible=session.show_axis,
            scene_zaxis_visible=session.show_axis,
            hovermode=False,
            scene_camera_eye=dict(x=1.5, y=-1.5, z=1.2),
        )

        return fig

    def make_binary(self, session):
        # Make a binary volume
        # Nothing fancy, but we could add new features here
        session.vol_binary = session.vol > (session.threshold * np.max(session.vol))

        session.vol_binary_display = ndimage.zoom(
            input=session.vol_binary * 255,
            zoom=(session.display_size_input / session.max_size),
            prefilter=False,
            order=0,
        )

        return session

    def binary_viz(self, session):
        # Generate input visualization
        data = go.Volume(
            z=session.Zgrid.flatten(),
            y=session.Ygrid.flatten(),
            x=session.Xgrid.flatten(),
            value=session.vol_binary_display.flatten(),
            opacity=0.3,
            isomin=0.99 * np.max(session.vol_binary_display),
            isomax=1.0 * np.max(session.vol_binary_display),
            cmin=np.min(session.vol_binary_display),
            cmax=np.max(session.vol_binary_display),
            opacityscale="max",
            surface_count=2,
            caps=dict(
                x_show=session.show_caps,
                y_show=session.show_caps,
                z_show=session.show_caps,
            ),
            showscale=False,
            reversescale=session.reversescale,
            colorscale="Greys",
        )

        fig = go.Figure(data)
        fig.update_layout(
            scene_aspectmode="data",
            scene_xaxis_showticklabels=session.show_ticks,
            scene_yaxis_showticklabels=session.show_ticks,
            scene_zaxis_showticklabels=session.show_ticks,
            scene_xaxis_visible=session.show_axis,
            scene_yaxis_visible=session.show_axis,
            scene_zaxis_visible=session.show_axis,
            hovermode=False,
            scene_camera_eye=dict(x=1.5, y=-1.5, z=1.2),
        )

        return fig

    def compute_localthickness(self, session):
        session.vol_thickness = lt.local_thickness(session.vol_binary, session.lt_scale)

        return session

    def prepare_output_for_display(self, session):
        # Display Local thickness
        session.vol_output_display = ndimage.zoom(
            input=session.vol_thickness,
            zoom=(session.display_size_output / session.max_size),
            prefilter=False,
            order=0,
        )

        output_display_shape = np.shape(session.vol_output_display)
        session.output_display_size_z = output_display_shape[0]
        session.output_display_size_y = output_display_shape[1]
        session.output_display_size_x = output_display_shape[2]

        session.Zgrid_output, session.Ygrid_output, session.Xgrid_output = np.mgrid[
            0 : session.output_display_size_z,
            0 : session.output_display_size_y,
            0 : session.output_display_size_x,
        ]

        return session

    def output_viz(self, session):
        # Generate input visualization
        data = go.Volume(
            z=session.Zgrid_output.flatten(),
            y=session.Ygrid_output.flatten(),
            x=session.Xgrid_output.flatten(),
            value=session.vol_output_display.flatten(),
            opacity=0.3,
            isomin=0.05 * np.max(session.vol_output_display),
            isomax=1.0 * np.max(session.vol_output_display),
            cmin=np.min(session.vol_output_display),
            cmax=np.max(session.vol_output_display),
            opacityscale="uniform",
            surface_count=session.surface_count_input,
            caps=dict(
                x_show=session.show_caps,
                y_show=session.show_caps,
                z_show=session.show_caps,
            ),
            showscale=False,
            reversescale=session.reversescale,
        )

        fig = go.Figure(data)
        fig.update_layout(
            scene_aspectmode="data",
            scene_xaxis_showticklabels=session.show_ticks,
            scene_yaxis_showticklabels=session.show_ticks,
            scene_zaxis_showticklabels=session.show_ticks,
            scene_xaxis_visible=session.show_axis,
            scene_yaxis_visible=session.show_axis,
            scene_zaxis_visible=session.show_axis,
            hovermode=False,
            scene_camera_eye=dict(x=1.5, y=-1.5, z=1.2),
        )

        return fig

    def thickness_histogram(self, session):
        # Ignore zero thickness
        non_zero_values = session.vol_thickness[session.vol_thickness > 0]

        # Calculate histogram
        vol_hist, bin_edges = np.histogram(non_zero_values, session.nbins)

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

        return fig

    def save_lt(self, session):
        filename = "localthickness.tif"
        # Save output image in a temp space
        tifffile.imwrite(filename, session.vol_thickness)

        return filename


def gradio_fn(
    gradio_file,
    lt_scale,
    threshold,
    dark_objects,
    flip_z,
    nbins,
    display_size_input,
    surface_count_input,
    display_size_output,
    surface_count_output,
    reversescale,
    show_caps,
    show_ticks=False,
    show_axis=True,
):
    # Some cleanup
    vol_input = None
    vol_input_display = None
    vol_output = None
    vol_output_display = None
    data = None

    return fig_input, fig_output, fig_binary, fig_hist, "localthickness.tif"


if __name__ == "__main__":
    app = Interface()
    app.show_header = True
    app.launch(server_name="0.0.0.0", show_error=True)
