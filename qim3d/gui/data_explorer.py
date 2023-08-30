import gradio as gr
import numpy as np
import os
from qim3d.utils import internal_tools
from qim3d.io import load
from qim3d.io.logger import log
import tifffile
import outputformat as ouf
import datetime
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Interface:
    def __init__(self):
        self.show_header = False
        self.verbose = False
        self.title = "Data Explorer"

        self.height = 1024
        self.width = 900

        # CSS path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.css_path = os.path.join(current_dir, "..", "css", "gradio.css")

    def clear(self):
        """Used to reset outputs with the clear button"""
        return None

    def create_interface(self):
        with gr.Blocks(css=self.css_path) as gradio_interface:
            gr.Markdown("# Data Explorer")

            with gr.Row():
                with gr.Column(scale=0.75):
                    data_path = gr.Textbox(
                        value="gbar/zhome/15/b/200707/img_examples/shell_225x128x128.tif",
                        max_lines=1,
                        label="Path to the 3D volume",
                    )
                with gr.Column(scale=0.25):
                    dataset_name = gr.Textbox(
                        label="Dataset name (in case of H5 files, for example)"
                    )

            with gr.Row(elem_classes="w-256"):
                cmap = gr.Dropdown(
                    value="viridis",
                    choices=plt.colormaps(),
                    label="Colormap",
                    interactive=True,
                )
            with gr.Row(elem_classes="w-128"):
                btn_run = gr.Button(value="Load & Run", elem_classes="btn btn-run")
            # Outputs
            with gr.Row():
                gr.Markdown("## Data overview")

            with gr.Row():
                data_summary = gr.Text(
                    label=None, show_label=False, elem_classes="monospace-box"
                )
            with gr.Row():
                with gr.Column():
                    zslice_plot = gr.Plot(label="Z slice", elem_classes="rounded")
                    zpos = gr.Slider(
                        minimum=0, maximum=1, value=0.5, step=0.01, label="Z position"
                    )
                with gr.Column():
                    yslice_plot = gr.Plot(label="Y slice", elem_classes="rounded")

                    ypos = gr.Slider(
                        minimum=0, maximum=1, value=0.5, step=0.01, label="Y position"
                    )

                with gr.Column():
                    xslice_plot = gr.Plot(label="X slice", elem_classes="rounded")

                    xpos = gr.Slider(
                        minimum=0, maximum=1, value=0.5, step=0.01, label="X position"
                    )
            with gr.Row(elem_classes="h-32"):
                gr.Markdown()

            with gr.Row(elem_classes="h-480"):
                max_projection_plot = gr.Plot(
                    label="Z max projection", elem_classes="rounded"
                )
                min_projection_plot = gr.Plot(
                    label="Z min projection", elem_classes="rounded"
                )

                hist_plot = gr.Plot(label="Volume intensity histogram")

            pipeline = Pipeline()
            pipeline.verbose = self.verbose
            session = gr.State([])

            ### Gradio objects lists

            # Inputs
            inputs = [zpos, ypos, xpos, cmap, dataset_name]
            # Outputs
            outputs = [
                data_summary,
                zslice_plot,
                yslice_plot,
                xslice_plot,
                max_projection_plot,
                min_projection_plot,
            ]

            projection_outputs = [session, max_projection_plot, min_projection_plot]

            ### Listeners
            # Clear button
            # for gr_obj in outputs:
            #     btn_clear.click(fn=self.clear, inputs=[], outputs=gr_obj)

            # Run button
            # fmt: off
            btn_run.click(
                fn=self.start_session, inputs=inputs, outputs=session).success(
                fn=pipeline.process_input, inputs=[session, data_path], outputs=session).success(
                fn=pipeline.show_summary_str, inputs=session, outputs=data_summary).success(
                fn=pipeline.create_zslice_fig, inputs=session, outputs=zslice_plot).success(
                fn=pipeline.create_yslice_fig, inputs=session, outputs=yslice_plot).success(
                fn=pipeline.create_xslice_fig, inputs=session, outputs=xslice_plot).success(
                fn=pipeline.create_projections_figs, inputs=session, outputs=projection_outputs).success(
                fn=pipeline.show_summary_str, inputs=session, outputs=data_summary).success(
                fn=pipeline.plot_vol_histogram, inputs=session, outputs=hist_plot)
                

            zpos.release(
                fn=self.update_zpos, inputs=[session, zpos], outputs=[session, zslice_plot]).success(
                fn=pipeline.create_zslice_fig, inputs=session, outputs=zslice_plot,show_progress=False)
            ypos.release(
                fn=self.update_ypos, inputs=[session, ypos], outputs=[session, yslice_plot]).success(
                fn=pipeline.create_yslice_fig, inputs=session, outputs=yslice_plot,show_progress=False)
            
            xpos.release(
                fn=self.update_xpos, inputs=[session, xpos], outputs=[session, xslice_plot]).success(
                fn=pipeline.create_xslice_fig, inputs=session, outputs=xslice_plot,show_progress=False)

            # fmt: on

        return gradio_interface

    def start_session(self, *args):
        # Starts a new session dictionary
        session = Session()
        session.interface = "gradio"
        session.zpos = args[0]
        session.ypos = args[1]
        session.xpos = args[2]
        session.cmap = args[3]
        session.dataset_name = args[4]

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


class Session:
    def __init__(self):
        self.interface = None
        self.data_path = None
        self.vol = None
        self.zpos = 0.5
        self.ypos = 0.5
        self.xpos = 0.5
        self.cmap = "viridis"
        self.dataset_name = None
        self.error_message = None

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

    def get_data_info(self):
        # Open file
        try:
            vol = load(
                self.data_path, virtual_stack=True, dataset_name=self.dataset_name
            )
        except Exception as error_message:
            self.error_message = error_message
            return

        first_slice = vol[0]

        # Get info
        self.zsize = len(vol)
        self.ysize, self.xsize = first_slice.shape
        self.data_type = str(first_slice.dtype)
        self.last_modified = datetime.datetime.fromtimestamp(
            os.path.getmtime(self.data_path)
        ).strftime("%Y-%m-%d %H:%M")
        self.file_size = os.path.getsize(self.data_path)


    def create_summary_dict(self):
        # Create dictionary
        if self.error_message:
            self.summary_dict = {"error_mesage": self.error_message}

        else:
            self.summary_dict = {
                "Last modified": self.last_modified,
                "File size": internal_tools.sizeof(self.file_size),
                "Z-size": str(self.zsize),
                "Y-size": str(self.ysize),
                "X-size": str(self.xsize),
                "Data type": self.data_type,
                "Min value": self.min_value,
                "Mean value": self.mean_intensity,
                "Max value": self.max_value,
            }

    def summary_str(self):
        if "error_mesage" in self.summary_dict:
            error_box = ouf.boxtitle("ERROR", return_str=True)
            return f"{error_box}\n{self.summary_dict['error_mesage']}"
        else:
            display_dict = {k: v for k, v in self.summary_dict.items() if v is not None}
            return ouf.showdict(display_dict, return_str=True, title="Data summary")

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

    def process_input(self, *args):
        session = args[0]
        session.data_path = args[1]

        # Get info from Tiff file
        session.get_data_info()
        session.create_summary_dict()

        # Memory map data as a virtual stack

        try:
            session.vol = load(
                session.data_path, virtual_stack=True, dataset_name=session.dataset_name
            )
        except:
            return session

        if self.verbose:
            log.info(ouf.br(3, return_str=True) + session.summary_str())

        return session

    def show_summary_str(self, session):
        session.create_summary_dict()
        return session.summary_str()

    def create_zslice_fig(self, session):
        slice_fig = self.create_slice_fig("z", session)

        return slice_fig

    def create_yslice_fig(self, session):
        slice_fig = self.create_slice_fig("y", session)

        return slice_fig

    def create_xslice_fig(self, session):
        slice_fig = self.create_slice_fig("x", session)

        return slice_fig

    def create_slice_fig(self, axis, session):
        plt.close()
        vol = session.vol
        plt.set_cmap(session.cmap)

        zslice = session.zslice_from_zpos()
        yslice = session.yslice_from_ypos()
        xslice = session.xslice_from_xpos()

        # Check if we something to use as vmin and vmax
        if session.min_percentile and session.max_percentile:
            vmin = session.min_percentile
            vmax = session.max_percentile
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

    def create_projections_figs(self, session):
        vol = session.vol

        # Run projections
        max_projection, min_projection = self.get_projections(vol, session)

        # Generate figures
        max_projection_fig = self.create_img_fig(
            max_projection,
            vmin=session.min_percentile,
            vmax=session.max_percentile,
        )
        min_projection_fig = self.create_img_fig(
            min_projection,
            vmin=session.min_percentile,
            vmax=session.max_percentile,
        )
        return session, max_projection_fig, min_projection_fig

    def get_projections(self, vol, session):
        # Create arrays for iteration
        max_projection = np.zeros(np.shape(vol[0]))
        min_projection = np.ones(np.shape(vol[0])) * float("inf")
        intensity_sum = 0

        # Iterate over slices
        for zslice in vol:
            max_projection = np.maximum(max_projection, zslice)
            min_projection = np.minimum(min_projection, zslice)
            intensity_sum += np.sum(zslice)

        session.min_value = np.min(min_projection)
        session.min_percentile = np.percentile(
            min_projection, 100 - self.display_saturation_percentile
        )
        session.max_value = np.max(max_projection)
        session.max_percentile = np.percentile(
            max_projection, self.display_saturation_percentile
        )

        session.intensity_sum = intensity_sum

        nvoxels = session.zsize * session.ysize * session.xsize
        session.mean_intensity = intensity_sum / nvoxels
        return max_projection, min_projection

    def plot_vol_histogram(self, session):
        vol_hist, bin_edges = self.vol_histogram(
            session.vol, session.nbins, session.min_value, session.max_value
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


if __name__ == "__main__":
    app = Interface()
    app.show_header = True
    app.launch(server_name="0.0.0.0", show_error=True)
