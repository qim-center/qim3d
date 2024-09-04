import os
import gradio as gr
from qim3d.utils import internal_tools
from qim3d.io import DataLoader
from qim3d.io.logger import log
from qim3d.process import layers2d as l2d
from qim3d.io import load
import qim3d.viz

# matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Session:
    def __init__(self):
        self.data = None
        self.virtual_stack = True
        self.dataset_name = ""
        self.base_path = None
        self.explorer = None
        self.file_path = None

        # Layers2D objects
        self.l2d_obj_x = None
        self.l2d_obj_y = None
        self.l2d_obj_z = None
        
        # Layers2D parameters
        self.delta = 0
        self.min_margin = 0
        self.n_layers = 0
        self.is_inverted = False
        
        # Axis positions
        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0
        
        # Slices
        self.x_slice = None
        self.y_slice = None
        self.z_slice = None

class Interface:
    def __init__(self):
        self.title = "Layered surfaces 2D"

        # Data examples
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # examples_dir = ["..", "img_examples"]
        # examples = [
        #     "blobs_256x256x256.tif",
        #     "cement_128x128x128.tif",
        #     "bone_128x128x128.tif",
        #     "slice_218x193.png",
        # ]
        # self.img_examples = []
        # for example in examples:
        #     self.img_examples.append(
        #         [os.path.join(current_dir, *examples_dir, example)]
        #     )

        # CSS path
        self.css_path = os.path.join(current_dir, "..", "css", "gradio.css")

    def start_session(self, *args):
        session = Session()
        session.base_path = args[0]
        session.explorer = args[1]
        session.delta = args[2]
        session.min_margin = args[3]
        session.n_layers = args[4]
        session.is_inverted = args[5]

        session.x_pos = args[6]
        session.y_pos = args[7]
        session.z_pos = args[8]
        
        # Get the file path from the explorer or base path
        if session.base_path and os.path.isfile(session.base_path):
            session.file_path = session.base_path
        elif session.explorer and os.path.isfile(session.explorer):
            session.file_path = session.explorer
        else:
            raise ValueError("Invalid file path")

        return session

    def update_session_x_pos(self, session, x_pos):
        session.x_pos = x_pos
        return session
    
    def update_session_y_pos(self, session, y_pos):
        session.y_pos = y_pos
        return session
    
    def update_session_z_pos(self, session, z_pos):
        session.z_pos = z_pos
        return session
    
    def update_session_delta(self, session, delta):
        session.delta = delta
        return session
    
    def update_session_min_margin(self, session, min_margin):
        session.min_margin = min_margin
        return session
    
    def update_session_n_layers(self, session, n_layers):
        session.n_layers = n_layers
        return session
    
    def update_session_is_inverted(self, session, is_inverted):
        session.is_inverted = is_inverted
        return session
    
    def update_explorer(self, new_path):
        # Refresh the file explorer object
        new_path = os.path.expanduser(new_path)

        # In case we have a directory
        if os.path.isdir(new_path):
            return gr.update(root=new_path, label=new_path)

        elif os.path.isfile(new_path):
            parent_dir = os.path.dirname(new_path)
            file_name = str(os.path.basename(new_path))
            return gr.update(root=parent_dir, label=parent_dir, value=file_name)

        else:
            raise ValueError("Invalid path")

    def set_relaunch_button(self):
        # Sets the button to relaunch
        return gr.update(
            elem_classes="btn btn-run",
            value=f"Relaunch",
            interactive=True,
        )

    def set_spinner(self, message):
        # spinner icon/shows the user something is happeing
        return gr.update(
            elem_classes="btn btn-spinner",
            value=f"{message}",
            interactive=False,
        )

    def create_interface(self):
        with gr.Blocks(css=self.css_path) as gradio_interface:
            gr.Markdown(f"# {self.title}")

            with gr.Row():
                # Control panel: Input data and parameters
                with gr.Column(scale=1, min_width=320):
                    with gr.Row():
                        with gr.Column(scale=99, min_width=128):
                            base_path = gr.Textbox(
                                max_lines=1,
                                container=False,
                                label="Base path",
                                elem_classes="h-36",
                                value=os.getcwd(),
                            )
                        with gr.Column(scale=1, min_width=36):
                            reload_base_path = gr.Button(
                                value="⟳", elem_classes="btn-html h-36"
                            )
                    explorer = gr.FileExplorer(
                        glob="{*/,}{*.*}",
                        root=os.getcwd(),
                        label=os.getcwd(),
                        render=True,
                        file_count="single",
                        interactive=True,
                        elem_classes="h-256 hide-overflow",
                    )
                    
                    # Parameters sliders and checkboxes
                    with gr.Row():
                        delta = gr.Slider(
                            minimum=0.5,
                            maximum=1.0,
                            value=0.75,
                            step=0.01,
                            label="Delta value",
                            info="Delta value for the gradient calculation. The lower the delta value is set, the more accurate the gradient calculation will be. However, the calculation takes longer to execute.", 
                        )
                        
                    with gr.Row():
                        min_margin = gr.Slider(
                            minimum=1, 
                            maximum=50, 
                            value=10, 
                            step=1, 
                            label="Min margin",
                            info="Minimum margin between layers to be detected in the image.",
                        )

                    with gr.Row():
                        n_layers = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=2,
                            step=1,
                            label="Number of layers",
                            info="Number of layers to be detected in the image",
                        )

                    with gr.Row():
                        is_inverted = gr.Checkbox(
                            label="Is inverted",
                            info="To invert the image before processing, click this box. By inverting the source image before processing, the algorithm effectively flips the gradient.",
                        )                    

                    # Relaunch button
                    with gr.Row():
                        btn_run = gr.Button(
                            "Run Layers2D", elem_classes="btn btn-html btn-run"
                        )

                # Output panel: Plots
                with gr.Column(scale=2):
                    with gr.Row(): # Source image outputs
                        input_plot_x = gr.Plot(
                            show_label=True,
                            label="Slice along X-axis",
                            visible=True,
                            elem_classes="rounded",
                        )
                        input_plot_y = gr.Plot(
                            show_label=True,
                            label="Slice along Y-axis",
                            visible=True,
                            elem_classes="rounded",
                        )
                        input_plot_z = gr.Plot(
                            show_label=True,
                            label="Slice along Z-axis",
                            visible=True,
                            elem_classes="rounded",
                        )
                    with gr.Row(): # Detected layers outputs
                        output_plot_x = gr.Plot(
                            show_label=True,
                            label="Detected layers X-axis",
                            visible=True,
                            elem_classes="rounded",
                        )
                        output_plot_y = gr.Plot(
                            show_label=True,
                            label="Detected layers Y-axis",
                            visible=True,
                            elem_classes="rounded",
                        )
                        output_plot_z = gr.Plot(
                            show_label=True,
                            label="Detected layers Z-axis",
                            visible=True,
                            elem_classes="rounded",
                        )
                    with gr.Row(): # Axis position sliders
                        x_pos = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.01,
                            label="X position",
                            info="The 3D image is sliced along the X-axis.",
                        )
                        y_pos = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.01,
                            label="Y position",
                            info="The 3D image is sliced along the Y-axis.",
                        )
                        z_pos = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.01,
                            label="Z position",
                            info="The 3D image is sliced along the Z-axis.",
                        )
            
            session = gr.State([])
            
            pipeline = Pipeline()
            
            inputs = [base_path, explorer, delta, min_margin, n_layers, is_inverted, x_pos, y_pos, z_pos]
            spinner_loading = gr.Text("Loading data...", visible=False)
            spinner_running = gr.Text("Running pipeline...", visible=False)
            spinner_updating = gr.Text("Updating layers...", visible=False)

            # fmt: off
            reload_base_path.click(
                fn=self.update_explorer,inputs=base_path, outputs=explorer)

            btn_run.click(
                fn=self.start_session, inputs=inputs, outputs=session).then(
                fn=self.set_spinner, inputs=spinner_loading, outputs=btn_run).then(
                fn=pipeline.load_data, inputs=session, outputs=session).then(
                fn=self.set_spinner, inputs=spinner_running, outputs=btn_run).then(
                fn=pipeline.update_slices, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_x, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_y, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_z, inputs=session, outputs=session).then(
                fn=pipeline.plot_input_img_x, inputs=session, outputs=input_plot_x).then(
                fn=pipeline.plot_input_img_y, inputs=session, outputs=input_plot_y).then(
                fn=pipeline.plot_input_img_z, inputs=session, outputs=input_plot_z).then(
                fn=pipeline.plot_l2d_output_x, inputs=session, outputs=output_plot_x).then(
                fn=pipeline.plot_l2d_output_y, inputs=session, outputs=output_plot_y).then(
                fn=pipeline.plot_l2d_output_z, inputs=session, outputs=output_plot_z).then(
                fn=self.set_relaunch_button, inputs=[], outputs=btn_run)
            
            delta.change(
                fn=self.update_session_delta, inputs=[session, delta], outputs=session, show_progress=False).success(
                fn=self.set_spinner, inputs=spinner_updating, outputs=btn_run).then(
                fn=pipeline.update_slices, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_x, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_y, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_z, inputs=session, outputs=session).then(
                fn=pipeline.plot_l2d_output_x, inputs=session, outputs=output_plot_x).then(
                fn=pipeline.plot_l2d_output_y, inputs=session, outputs=output_plot_y).then(
                fn=pipeline.plot_l2d_output_z, inputs=session, outputs=output_plot_z).then(
                fn=self.set_relaunch_button, inputs=[], outputs=btn_run)
            
            min_margin.change(
                fn=self.update_session_min_margin, inputs=[session, min_margin], outputs=session, show_progress=False).success(
                fn=self.set_spinner, inputs=spinner_updating, outputs=btn_run).then(
                fn=pipeline.update_slices, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_x, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_y, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_z, inputs=session, outputs=session).then(
                fn=pipeline.plot_l2d_output_x, inputs=session, outputs=output_plot_x).then(
                fn=pipeline.plot_l2d_output_y, inputs=session, outputs=output_plot_y).then(
                fn=pipeline.plot_l2d_output_z, inputs=session, outputs=output_plot_z).then(
                fn=self.set_relaunch_button, inputs=[], outputs=btn_run)
            
            n_layers.change(
                fn=self.update_session_n_layers, inputs=[session, n_layers], outputs=session, show_progress=False).success(
                fn=self.set_spinner, inputs=spinner_updating, outputs=btn_run).then(
                fn=pipeline.update_slices, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_x, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_y, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_z, inputs=session, outputs=session).then(
                fn=pipeline.plot_l2d_output_x, inputs=session, outputs=output_plot_x).then(
                fn=pipeline.plot_l2d_output_y, inputs=session, outputs=output_plot_y).then(
                fn=pipeline.plot_l2d_output_z, inputs=session, outputs=output_plot_z).then(
                fn=self.set_relaunch_button, inputs=[], outputs=btn_run)
            
            is_inverted.change(
                fn=self.update_session_is_inverted, inputs=[session, is_inverted], outputs=session, show_progress=False).success(
                fn=self.set_spinner, inputs=spinner_updating, outputs=btn_run).then(
                fn=pipeline.update_slices, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_x, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_y, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_z, inputs=session, outputs=session).then(
                fn=pipeline.plot_l2d_output_x, inputs=session, outputs=output_plot_x).then(
                fn=pipeline.plot_l2d_output_y, inputs=session, outputs=output_plot_y).then(
                fn=pipeline.plot_l2d_output_z, inputs=session, outputs=output_plot_z).then(
                fn=self.set_relaunch_button, inputs=[], outputs=btn_run)            
            
            x_pos.change(
                fn=self.update_session_x_pos, inputs=[session, x_pos], outputs=session, show_progress=False).success(
                fn=self.set_spinner, inputs=spinner_updating, outputs=btn_run).then(
                fn=pipeline.update_slices, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_x, inputs=session, outputs=session).then(
                fn=pipeline.plot_input_img_x, inputs=session, outputs=input_plot_x).then(
                fn=pipeline.plot_l2d_output_x, inputs=session, outputs=output_plot_x).then(
                fn=self.set_relaunch_button, inputs=[], outputs=btn_run)
            
            y_pos.change(
                fn=self.update_session_y_pos, inputs=[session, y_pos], outputs=session, show_progress=False).success(
                fn=self.set_spinner, inputs=spinner_updating, outputs=btn_run).then(
                fn=pipeline.update_slices, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_y, inputs=session, outputs=session).then(
                fn=pipeline.plot_input_img_y, inputs=session, outputs=input_plot_y).then(
                fn=pipeline.plot_l2d_output_y, inputs=session, outputs=output_plot_y).then(
                fn=self.set_relaunch_button, inputs=[], outputs=btn_run)
            
            z_pos.change(
                fn=self.update_session_z_pos, inputs=[session, z_pos], outputs=session, show_progress=False).success(
                fn=self.set_spinner, inputs=spinner_updating, outputs=btn_run).then(
                fn=pipeline.update_slices, inputs=session, outputs=session).then(
                fn=pipeline.process_l2d_z, inputs=session, outputs=session).then(
                fn=pipeline.plot_input_img_z, inputs=session, outputs=input_plot_z).then(
                fn=pipeline.plot_l2d_output_z, inputs=session, outputs=output_plot_z).then(
                fn=self.set_relaunch_button, inputs=[], outputs=btn_run)
            
            # fmt: on
            
        return gradio_interface


class Pipeline:
    def __init__(self):
        self.figsize = (8, 8)

    def load_data(self, session):
        try:
            session.data = load(
                session.file_path,
                virtual_stack=session.virtual_stack,
                dataset_name=session.dataset_name,
            )
        except Exception as error_message:
            raise ValueError(
                f"Failed to load the image: {error_message}"
            ) from error_message

        return session
    
    def update_slices(self, session):
        x_idx = int(session.x_pos * (session.data.shape[0] - 1))
        y_idx = int(session.y_pos * (session.data.shape[1] - 1))
        z_idx = int(session.z_pos * (session.data.shape[2] - 1))
        
        session.x_slice = session.data[x_idx, :, :]
        session.y_slice = session.data[:, y_idx, :]
        session.z_slice = session.data[:, :, z_idx]
        return session

    def process_l2d_x(self, session):
        if session.l2d_obj_x is None:
            l2d_obj = l2d.Layers2d()
        else:
            l2d_obj = session.l2d_obj_x
            
        l2d_obj.prepare_update(
            data=session.x_slice,
            is_inverted=session.is_inverted,
            delta=session.delta,
            min_margin=session.min_margin,
            n_layers=session.n_layers,
        )
        l2d_obj.update()

        session.l2d_obj_x = l2d_obj

        return session
    
    def process_l2d_y(self, session):
        if session.l2d_obj_y is None:
            l2d_obj = l2d.Layers2d()
        else:
            l2d_obj = session.l2d_obj_y
            
        l2d_obj.prepare_update(
            data=session.y_slice,
            is_inverted=session.is_inverted,
            delta=session.delta,
            min_margin=session.min_margin,
            n_layers=session.n_layers,
        )
        l2d_obj.update()

        session.l2d_obj_y = l2d_obj

        return session

    def process_l2d_z(self, session):
        if session.l2d_obj_z is None:
            l2d_obj = l2d.Layers2d()
        else:
            l2d_obj = session.l2d_obj_z
            
        l2d_obj.prepare_update(
            data=session.z_slice,
            is_inverted=session.is_inverted,
            delta=session.delta,
            min_margin=session.min_margin,
            n_layers=session.n_layers,
        )
        l2d_obj.update()

        session.l2d_obj_z = l2d_obj

        return session
    
    # Plotting functions
    def plot_input_img_x(self, session, cmap="Greys_r"):
        plt.close()
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(session.x_slice, interpolation="nearest", cmap=cmap)

        # Adjustments
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig
    
    def plot_input_img_y(self, session, cmap="Greys_r"):
        plt.close()
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(session.y_slice, interpolation="nearest", cmap=cmap)

        # Adjustments
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig

    def plot_input_img_z(self, session, cmap="Greys_r"):
        plt.close()
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(session.z_slice, interpolation="nearest", cmap=cmap)

        # Adjustments
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig

    def plot_l2d_output_x(self, session):
        l2d_obj = session.l2d_obj_x
        fig, ax = qim3d.viz.layers2d.create_plot_of_2d_array(l2d_obj.get_data_not_inverted())

        data_lines = []
        for i in range(len(l2d_obj.get_segmentation_lines())):
            data_lines.append(l2d_obj.get_segmentation_lines()[i])

        # Show how add_line_to_plot works:
        for line in data_lines:
            qim3d.viz.layers2d.add_line_to_plot(ax, line)

        # Adjustments
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig
    
    def plot_l2d_output_y(self, session):
        l2d_obj = session.l2d_obj_y
        fig, ax = qim3d.viz.layers2d.create_plot_of_2d_array(l2d_obj.get_data_not_inverted())

        data_lines = []
        for i in range(len(l2d_obj.get_segmentation_lines())):
            data_lines.append(l2d_obj.get_segmentation_lines()[i])

        # Show how add_line_to_plot works:
        for line in data_lines:
            qim3d.viz.layers2d.add_line_to_plot(ax, line)

        # Adjustments
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig
    
    def plot_l2d_output_z(self, session):
        l2d_obj = session.l2d_obj_z
        fig, ax = qim3d.viz.layers2d.create_plot_of_2d_array(l2d_obj.get_data_not_inverted())

        data_lines = []
        for i in range(len(l2d_obj.get_segmentation_lines())):
            data_lines.append(l2d_obj.get_segmentation_lines()[i])

        # Show how add_line_to_plot works:
        for line in data_lines:
            qim3d.viz.layers2d.add_line_to_plot(ax, line)

        # Adjustments
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig

def run_interface(host = "0.0.0.0"):
    gradio_interface = Interface().create_interface()
    internal_tools.run_gradio_app(gradio_interface,host)

if __name__ == "__main__":
    # Creates interface
    run_interface()
    