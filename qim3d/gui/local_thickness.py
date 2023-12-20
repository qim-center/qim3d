import gradio as gr
import numpy as np
import os
from qim3d.utils import internal_tools
from qim3d.io import DataLoader
from qim3d.io.logger import log
import tifffile
import plotly.express as px
from scipy import ndimage
import outputformat as ouf
import plotly.graph_objects as go
import localthickness as lt
import matplotlib

# matplotlib.use("Agg")
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
        examples_dir = ["..", "img_examples"]
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
        session.nbins = args[4]
        session.zpos = args[5]
        session.cmap_originals = args[6]
        session.cmap_lt = args[7]

        return session

    def update_session_zpos(self, session, zpos):
        session.zpos = zpos
        return session

    def launch(self, img=None):
        # Show header
        if self.show_header:
            internal_tools.gradio_header(self.title, self.port)

        # Create gradio interfaces

        self.interface = self.create_interface(img=img)

        # Set gradio verbose level
        if self.verbose:
            quiet = False
        else:
            quiet = True

        self.interface.launch(
            quiet=quiet,
            height=self.height,
            width=self.width,
        )

        return

    def get_result(self):
        # Get the temporary files from gradio
        temp_sets = self.interface.temp_file_sets
        for temp_set in temp_sets:
            if "localthickness" in str(temp_set):
                # Get the lsit of the temporary files
                temp_path_list = list(temp_set)

        # Files are not in creation order,
        # so we need to get find the latest
        creation_time_list = []
        for path in temp_path_list:
            creation_time_list.append(os.path.getctime(path))

        # Get index for the latest file
        file_idx = np.argmax(creation_time_list)

        # Load the temporary file
        vol_lt = DataLoader().load(temp_path_list[file_idx])

        return vol_lt

    def create_interface(self, img=None):
        with gr.Blocks(css=self.css_path) as gradio_interface:
            gr.Markdown(
                "# 3D Local thickness \n Interface for _Fast local thickness in 3D and 2D_ (https://github.com/vedranaa/local-thickness)"
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    if img is not None:
                        data = gr.State(value=img)
                    else:
                        with gr.Tab("Input"):
                            data = gr.File(
                                show_label=False,
                                elem_classes="file-input h-128",
                                value=img,
                            )
                        with gr.Tab("Examples"):
                            gr.Examples(examples=self.img_examples, inputs=data)

                    with gr.Row():
                        zpos = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.01,
                            label="Z position",
                            info="Local thickness is calculated in 3D, this slider controls the visualization only.",
                        )

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

                        dark_objects = gr.Checkbox(
                            value=False,
                            label="Dark objects",
                            info="Inverts the image before trhesholding. Use in case your foreground is darker than the background.",
                        )

                    with gr.Tab("Display options"):
                        cmap_original = gr.Dropdown(
                            value="viridis",
                            choices=plt.colormaps(),
                            label="Colormap - input",
                            interactive=True,
                        )
                        cmap_lt = gr.Dropdown(
                            value="magma",
                            choices=plt.colormaps(),
                            label="Colormap - local thickness",
                            interactive=True,
                        )

                        nbins = gr.Slider(
                            5, 50, value=25, step=1, label="Histogram bins"
                        )

                    # Run button
                    with gr.Row():
                        with gr.Column(scale=3, min_width=64):
                            btn = gr.Button(
                                "Run local thickness", elem_classes="btn btn-run"
                            )
                        with gr.Column(scale=1, min_width=64):
                            btn_clear = gr.Button("Clear", elem_classes="btn btn-clear")

                    inputs = [
                        data,
                        lt_scale,
                        threshold,
                        dark_objects,
                        nbins,
                        zpos,
                        cmap_original,
                        cmap_lt,
                    ]

                with gr.Column(scale=4):
                    with gr.Row():
                        input_vol = gr.Plot(
                            show_label=True,
                            label="Original",
                            visible=True,
                            elem_classes="plot",
                        )

                        binary_vol = gr.Plot(
                            show_label=True,
                            label="Binary",
                            visible=True,
                            elem_classes="plot",
                        )

                        output_vol = gr.Plot(
                            show_label=True,
                            label="Local thickness",
                            visible=True,
                            elem_classes="plot",
                        )
                    with gr.Row():
                        histogram = gr.Plot(
                            show_label=True,
                            label="Thickness histogram",
                            visible=True,
                        )
                    with gr.Row():
                        lt_output = gr.File(
                            interactive=False,
                            show_label=True,
                            label="Output file",
                            visible=False,
                            elem_classes="",
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
                fn=pipeline.input_viz, inputs=session, outputs=input_vol).success(
                fn=pipeline.make_binary, inputs=session, outputs=session).success(
                fn=pipeline.binary_viz, inputs=session, outputs=binary_vol).success(
                fn=pipeline.compute_localthickness, inputs=session, outputs=session).success(
                fn=pipeline.output_viz, inputs=session, outputs=output_vol).success(
                fn=pipeline.thickness_histogram, inputs=session, outputs=histogram).success(
                fn=pipeline.save_lt, inputs=session, outputs=lt_output).success(
                fn=pipeline.remove_unused_file).success(
                fn=self.make_visible, inputs=None, outputs=lt_output)


            zpos.change(
                fn=self.update_session_zpos, inputs=[session, zpos], outputs=session, show_progress=False).success(
                fn=pipeline.input_viz, inputs=session, outputs=input_vol, show_progress=False).success(
                fn=pipeline.binary_viz, inputs=session, outputs=binary_vol,show_progress=False).success(
                fn=pipeline.output_viz, inputs=session, outputs=output_vol,show_progress=False)
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
        self.lt_scale = None
        self.threshold = 0.5
        self.dark_objects = False
        self.flip_z = True
        self.nbins = 25
        self.reversescale = False

        # From pipeline
        self.vol = None
        self.vol_binary = None
        self.vol_thickness = None
        self.zpos = 0
        self.vmin = None
        self.vmax = None
        self.vmin_lt = None
        self.vmax_lt = None


class Pipeline:
    def __init__(self):
        self.figsize = 6

    def process_input(self, session):
        # Load volume
        try:
            session.vol = DataLoader().load(session.data.name)
        except:
            session.vol = session.data

        if session.dark_objects:
            session.vol = np.invert(session.vol)

        # Get min and max values for visualization
        session.vmin = np.min(session.vol)
        session.vmax = np.max(session.vol)

        return session

    def show_slice(self, vol, z_idx, vmin=None, vmax=None, cmap="viridis"):
        plt.close()
        fig, ax = plt.subplots(figsize=(self.figsize, self.figsize))

        ax.imshow(vol[z_idx], interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

        # Adjustments
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig

    def input_viz(self, session):
        # Generate input visualization
        z_idx = int(session.zpos * (session.vol.shape[0] - 1))
        fig = self.show_slice(
            vol=session.vol,
            z_idx=z_idx,
            cmap=session.cmap_originals,
            vmin=session.vmin,
            vmax=session.vmax,
        )
        return fig

    def make_binary(self, session):
        # Make a binary volume
        # Nothing fancy, but we could add new features here
        session.vol_binary = session.vol > (session.threshold * np.max(session.vol))

        return session

    def binary_viz(self, session):
        # Generate input visualization
        z_idx = int(session.zpos * (session.vol_binary.shape[0] - 1))
        fig = self.show_slice(
            vol=session.vol_binary, z_idx=z_idx, cmap=session.cmap_originals
        )
        return fig

    def compute_localthickness(self, session):
        session.vol_thickness = lt.local_thickness(session.vol_binary, session.lt_scale)

        # Valus for visualization
        session.vmin_lt = np.min(session.vol_thickness)
        session.vmax_lt = np.max(session.vol_thickness)

        return session

    def output_viz(self, session):
        # Generate input visualization
        z_idx = int(session.zpos * (session.vol_thickness.shape[0] - 1))
        fig = self.show_slice(
            vol=session.vol_thickness,
            z_idx=z_idx,
            cmap=session.cmap_lt,
            vmin=session.vmin_lt,
            vmax=session.vmax_lt,
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
    
    def remove_unused_file(self):
        # Remove localthickness.tif file from working directory
        # as it otherwise is not deleted
        os.remove('localthickness.tif')

def run_interface(host = "0.0.0.0"):
    gradio_interface = Interface().create_interface()
    internal_tools.run_gradio_app(gradio_interface,host)

if __name__ == "__main__":
    # Creates interface
    run_interface()