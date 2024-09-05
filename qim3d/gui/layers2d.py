import os

import gradio as gr
import matplotlib.pyplot as plt

from .interface import BaseInterface

from qim3d.processing import layers2d as l2d
from qim3d.io import load
import qim3d.viz

#TODO figure out how not update anything and go through processing when there are no data loaded
# So user could play with the widgets but it doesnt throw error
# Right now its only bypassed with several if statements
# I opened an issue here https://github.com/gradio-app/gradio/issues/9273

class Interface(BaseInterface):
    def __init__(self):
        super().__init__("Layered surfaces 2D", 1080)

        self.l2d_obj_x = l2d.Layers2d()
        self.l2d_obj_y = l2d.Layers2d()
        self.l2d_obj_z = l2d.Layers2d()

        self.figsize = (8, 8)
        self.cmap = "Greys_r"

        self.data = None

        self.virtual_stack = True #TODO ask why
        self.dataset_name = '' #TODO check if necessary to even have

        self.error_state = False



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
                
                # Parameters sliders and checkboxes
                with gr.Row():
                    delta = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.75,
                        step=0.01,
                        label="Delta value",
                        info="The lower the delta is, the more accurate the gradient calculation will be. However, the calculation takes longer to execute.", 
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

                with gr.Row():
                    btn_run = gr.Button("Run Layers2D", variant = 'primary')

            # Output panel: Plots
            with gr.Column(scale=2):
                with gr.Row(): # Source image outputs
                    input_plot_x = gr.Plot(
                        show_label=True,
                        label="Slice along X-axis",
                        visible=True,
                    )
                    input_plot_y = gr.Plot(
                        show_label=True,
                        label="Slice along Y-axis",
                        visible=True,
                    )
                    input_plot_z = gr.Plot(
                        show_label=True,
                        label="Slice along Z-axis",
                        visible=True,
                    )
                with gr.Row(): # Detected layers outputs
                    output_plot_x = gr.Plot(
                        show_label=True,
                        label="Detected layers X-axis",
                        visible=True,

                    )
                    output_plot_y = gr.Plot(
                        show_label=True,
                        label="Detected layers Y-axis",
                        visible=True,

                    )
                    output_plot_z = gr.Plot(
                        show_label=True,
                        label="Detected layers Z-axis",
                        visible=True,
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
        
        positions = [x_pos, y_pos, z_pos]
        process_inputs = [is_inverted, delta, min_margin, n_layers]
        input_plots = [input_plot_x, input_plot_y, input_plot_z]
        output_plots = [output_plot_x, output_plot_y, output_plot_z]

        spinner_loading = gr.Text("Loading data...", visible=False)
        spinner_running = gr.Text("Running pipeline...", visible=False)
        spinner_updating = gr.Text("Updating layers...", visible=False)

        # fmt: off
        reload_base_path.click(
            fn=self.update_explorer,inputs=base_path, outputs=explorer)

        btn_run.click(
            fn=self.set_spinner, inputs=spinner_loading, outputs=btn_run).then(
            fn=self.load_data, inputs = [base_path, explorer]).then(
            fn=self.set_spinner, inputs=spinner_running, outputs=btn_run).then(
            fn=self.process_all, inputs = [*positions, *process_inputs]).then(
            fn=self.plot_input_img_all, inputs = positions, outputs = input_plots, show_progress='hidden').then(
            fn=self.plot_output_all, outputs = output_plots, show_progress='hidden').then(
            fn=self.set_relaunch_button, inputs=[], outputs=btn_run)
        
        gr.on(
            triggers=[delta.change, min_margin.change, n_layers.change, is_inverted.change],
            fn=self.set_spinner, inputs=spinner_updating, outputs=btn_run).then(
            fn=self.process_all, inputs = [*positions, *process_inputs]).then(
            fn=self.plot_output_all, outputs = output_plots, show_progress='hidden').then(
            fn=self.set_relaunch_button, inputs=[], outputs=btn_run)
                    
        slider_change_arguments = (
            (x_pos, self.process_x, self.plot_input_img_x, input_plot_x, self.l2d_obj_x, output_plot_x),
            (y_pos, self.process_y, self.plot_input_img_y, input_plot_y, self.l2d_obj_y, output_plot_y),
            (z_pos, self.process_z, self.plot_input_img_z, input_plot_z, self.l2d_obj_z, output_plot_z))
        
        for slider, process_func, plot_input_func, input_plot, l2d_obj, output_plot in slider_change_arguments:
            slider.change(
                fn=self.set_spinner, inputs=spinner_updating, outputs=btn_run).then(
                fn=process_func, inputs=[slider, *process_inputs]).then(
                fn=plot_input_func, inputs=slider, outputs=input_plot, show_progress='hidden').then(
                fn=self.plot_output_wrapper(l2d_obj), outputs=output_plot, show_progress='hidden').then(
                fn=self.set_relaunch_button, inputs=[], outputs=btn_run)

        
    def update_explorer(self, new_path):
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
        # Sets the button to relaunch
        return gr.update(
            value=f"Relaunch",
            interactive=True,
        )

    def set_spinner(self, message):
        if not self.error_state:
            return gr.Button()
        # spinner icon/shows the user something is happeing
        return gr.update(
            value=f"{message}",
            interactive=False,
        )
    
    def load_data(self, base_path, explorer):
        if base_path and os.path.isfile(base_path):
            file_path = base_path
        elif explorer and os.path.isfile(explorer):
            file_path = explorer
        else:
            raise gr.Error("Invalid file path")

        try:
            self.data = load(
                file_path,
                virtual_stack=self.virtual_stack,
                dataset_name=self.dataset_name,
            )
        except Exception as error_message:
            raise gr.Error(
                f"Failed to load the image: {error_message}"
            ) from error_message
        
    def idx(self, pos, axis):
        return int(pos * (self.data.shape[axis] - 1))
    

    # PROCESSING FUNCTIONS

    def process(self, l2d_obj:l2d.Layers2d, slice, is_inverted, delta, min_margin, n_layers):
        l2d_obj.prepare_update(
            data = slice,
            is_inverted = is_inverted,
            delta = delta,
            min_margin = min_margin,
            n_layers = n_layers,
        )
        l2d_obj.update()

    def process_x(self, x_pos, is_inverted, delta, min_margin, n_layers):
        if self.data is not None:
            slice = self.data[self.idx(x_pos, 0), :, :]
            self.process(self.l2d_obj_x, slice, is_inverted, delta, min_margin, n_layers)
    
    def process_y(self, y_pos, is_inverted, delta, min_margin, n_layers):
        if self.data is not None:
            slice = self.data[:, self.idx(y_pos, 1), :]
            self.process(self.l2d_obj_y, slice, is_inverted, delta, min_margin, n_layers)

    def process_z(self, z_pos, is_inverted, delta, min_margin, n_layers):
        if self.data is not None:
            slice = self.data[:, :, self.idx(z_pos, 2)]
            self.process(self.l2d_obj_z, slice, is_inverted, delta, min_margin, n_layers)

    def process_all(self, x_pos, y_pos, z_pos, is_inverted, delta, min_margin, n_layers):
        self.process_x(x_pos, is_inverted, delta, min_margin, n_layers)
        self.process_y(y_pos, is_inverted, delta, min_margin, n_layers)
        self.process_z(z_pos, is_inverted, delta, min_margin, n_layers)
        

    # PLOTTING FUNCTIONS

    def plot_input_img(self, slice):
        plt.close()
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(slice, interpolation="nearest", cmap = self.cmap)

        # Adjustments
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig
    
    def plot_input_img_x(self, x_pos):
        if self.data is None:
            return gr.Plot()
        slice = self.data[self.idx(x_pos, 0), :, :]
        return self.plot_input_img(slice)
    
    def plot_input_img_y(self, y_pos):
        if self.data is None:
            return gr.Plot()
        slice = self.data[:, self.idx(y_pos, 1), :]
        return self.plot_input_img(slice)
    
    def plot_input_img_z(self, z_pos):
        if self.data is None:
            return gr.Plot()
        slice = self.data[:, :, self.idx(z_pos, 2)]
        return self.plot_input_img(slice)
    
    def plot_input_img_all(self, x_pos, y_pos, z_pos):
        x_plot = self.plot_input_img_x(x_pos)
        y_plot = self.plot_input_img_y(y_pos)
        z_plot = self.plot_input_img_z(z_pos)
        return x_plot, y_plot, z_plot

    def plot_output_wrapper(self, l2d_obj:l2d.Layers2d):
        def plot_l2d_output():
            if self.data is None:
                return gr.Plot()
            fig, ax = qim3d.viz.layers2d.create_plot_of_2d_array(l2d_obj.get_data_not_inverted())

            for line in l2d_obj.segmentation_lines:
                qim3d.viz.layers2d.add_line_to_plot(ax, line)

            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            return fig
        return plot_l2d_output

    def plot_output_all(self):
        x_output = self.plot_output_wrapper(self.l2d_obj_x)()
        y_output = self.plot_output_wrapper(self.l2d_obj_y)()
        z_output = self.plot_output_wrapper(self.l2d_obj_z)()
        return x_output, y_output, z_output

if __name__ == "__main__":
    Interface().run_interface()
    