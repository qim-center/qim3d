import gradio as gr
import numpy as np
import os
from qim.tools import internal_tools
from qim.io import DataLoader
import tifffile
import plotly.express as px
from scipy import ndimage
import outputformat as ouf
import plotly.graph_objects as go
import localthickness as lt
from app import apptools


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

        # Get the args passed by gradio
        session.interface = "gradio"
        session.data = args[0]
        session.threshold = args[1]
        session.dark_objects = args[2]
        session.flip_z = args[3]
        session.nbins = args[4]
        session.display_size_input = args[5]
        session.surface_count_input = args[6]
        session.display_size_output = args[7]
        session.surface_count_output = args[8]
        session.reversescale = args[9]
        session.show_caps = args[10]
    
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
                        data = gr.File(show_label=False, elem_classes="file-input h-128")
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


                    lt_output = gr.File(
                        interactive=False,
                        show_label=True,
                        label="Output file",
                        visible=True,
                        elem_classes="file-output",
                    )

                with gr.Column(scale=4):
                    with gr.Row():
                        with gr.Column(min_width=128):
                            input_vol = gr.Plot(
                                show_label=True, label="Original volume", visible=True
                            )

                        with gr.Column(min_width=128):
                            binary_vol = gr.Plot(
                                show_label=True, label="Binary volume", visible=True
                            )

                        with gr.Column(min_width=128):
                            histogram = gr.Plot(
                                show_label=True,
                                label="Thickness histogram",
                                visible=True,
                            )

                    output_vol = gr.Plot(
                        show_label=True, label="Local thickness volume", visible=True
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
                fn=pipeline.process_input, inputs=session, outputs=session, queue=False)
            # fmt: on

        return gradio_interface


class Session:
    def __init__(self):
        self.interface = None
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

class Pipeline:
    def run_localthickness(vol, scale, threshold):
        vol_binary = vol > (threshold * np.max(vol))
        thickness = lt.local_thickness(
            vol_binary,
            scale,
        )

        return thickness, vol_binary

    def process_input(self, session):

        session.vol = DataLoader.load(session.data.name)

        if session.dark_objects:
            session.vol = np.invert(session.vol)

        if session.flip_z:
            session.vol = np.flip(session.vol, axis=0)
            
        return session


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
    vol_input = tifffile.imread(gradio_file.name, dtype=np.uint8)



    # Get volume shape
    original_Z, original_Y, original_X = np.shape(vol_input)
    max_size = np.max([original_Z, original_Y, original_X])

    # Resize for display
    vol_input_display = ndimage.zoom(
        input=vol_input, zoom=(display_size_input / max_size), prefilter=False, order=0
    )

    display_size_z, display_size_y, display_size_x = np.shape(vol_input_display)

    # Create 3D grid
    Z, Y, X = np.mgrid[0:display_size_z, 0:display_size_y, 0:display_size_x]

    # Generate input visualization
    data = go.Volume(
        z=Z.flatten(),
        y=Y.flatten(),
        x=X.flatten(),
        value=vol_input_display.flatten(),
        opacity=0.3,
        isomin=0.05 * np.max(vol_input_display),
        isomax=1.0 * np.max(vol_input_display),
        cmin=np.min(vol_input_display),
        cmax=np.max(vol_input_display),
        opacityscale="uniform",
        surface_count=surface_count_input,
        caps=dict(x_show=show_caps, y_show=show_caps, z_show=show_caps),
        showscale=False,
        reversescale=reversescale,
    )

    fig_input = go.Figure(data)
    fig_input.update_layout(scene_aspectmode="data", height=256)

    # Compute local thickness
    vol_output, vol_binary = run_localthickness(
        vol_input, scale=lt_scale, threshold=threshold
    )

    # Display binary volume
    vol_binary_display = ndimage.zoom(
        input=vol_binary * 255,
        zoom=(display_size_input / max_size),
        prefilter=False,
        order=0,
    )

    # Generate binary visualization
    data = go.Volume(
        z=Z.flatten(),
        y=Y.flatten(),
        x=X.flatten(),
        value=vol_binary_display.flatten(),
        opacity=0.4,
        isomin=0.99 * np.max(vol_binary_display),
        isomax=1.00 * np.max(vol_binary_display),
        cmin=np.min(vol_binary_display),
        cmax=np.max(vol_binary_display),
        opacityscale="max",
        surface_count=2,
        caps=dict(x_show=show_caps, y_show=show_caps, z_show=show_caps),
        showscale=False,
        colorscale="Greys",
    )

    # Display Locak thickness
    vol_output_display = ndimage.zoom(
        input=vol_output,
        zoom=(display_size_output / max_size),
        prefilter=False,
        order=0,
    )

    display_size_z, display_size_y, display_size_x = np.shape(vol_output_display)

    # Create 3D grid
    Z, Y, X = np.mgrid[0:display_size_z, 0:display_size_y, 0:display_size_x]

    fig_binary = go.Figure(data)
    fig_binary.update_layout(scene_aspectmode="data", height=256)

    # Make data histogram
    fig_hist = px.histogram(
        vol_output.flatten()[vol_output.flatten() > 0],
        nbins=nbins,
        histnorm="percent",
        labels="Thickness",
    )

    fig_hist.update_layout(
        showlegend=False,
        height=256,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=300, r=300, t=300, b=300),
    )

    # Generate output visualization
    data = go.Volume(
        z=Z.flatten(),
        y=Y.flatten(),
        x=X.flatten(),
        value=vol_output_display.flatten(),
        opacity=0.1,
        isomin=0.01 * np.max(vol_output_display),
        isomax=1.0 * np.max(vol_output_display),
        cmin=np.min(vol_output_display),
        cmax=np.max(vol_output_display),
        opacityscale="uniform",
        surface_count=surface_count_output,
        caps=dict(x_show=show_caps, y_show=show_caps, z_show=show_caps),
        colorbar=dict(thickness=8, outlinecolor="#fff", len=0.5, orientation="h"),
        reversescale=reversescale,
    )

    fig_output = go.Figure(data)
    fig_output.update_layout(scene_aspectmode="data", height=512)

    # Adjust 3D plots
    for fig in [fig_input, fig_binary, fig_output]:
        fig.update_layout(
            scene_xaxis_showticklabels=show_ticks,
            scene_yaxis_showticklabels=show_ticks,
            scene_zaxis_showticklabels=show_ticks,
            scene_xaxis_visible=show_axis,
            scene_yaxis_visible=show_axis,
            scene_zaxis_visible=show_axis,
            scene_aspectmode="data",
            hovermode=False,
            scene_camera_eye=dict(x=1.5, y=-1.5, z=1.2),
        )

    # Save output image in a temp space
    tifffile.imwrite("localthickness.tif", vol_output)

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
