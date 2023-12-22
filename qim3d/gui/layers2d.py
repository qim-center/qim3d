import os
import gradio as gr
from qim3d.utils import internal_tools
from qim3d.io import DataLoader
from qim3d.io.logger import log
from qim3d.process import layers2d as l2d
from qim3d.io import load

# matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Session:
    def __init__(self):
        self.data = None
        self.virtual_stack = False
        self.dataset_name = "exchange/data"
        self.base_path = None
        self.explorer = None
        self.file_path = None


class Interface:
    def __init__(self):
        self.title = "Layered surfaces 2D"

        # Data examples
        current_dir = os.path.dirname(os.path.abspath(__file__))
        examples_dir = ["..", "img_examples"]
        examples = [
            "blobs_256x256x256.tif",
            "cement_128x128x128.tif",
            "bone_128x128x128.tif",
            "slice_218x193.png",
        ]
        self.img_examples = []
        for example in examples:
            self.img_examples.append(
                [os.path.join(current_dir, *examples_dir, example)]
            )

        # CSS path
        self.css_path = os.path.join(current_dir, "..", "css", "gradio.css")

    def start_session(self, *args):
        session = Session()
        session.base_path = args[0]
        session.explorer = args[1]

        # Get the file path from the explorer or base path
        if session.base_path and os.path.isfile(session.base_path):
            session.file_path = session.base_path
        elif session.explorer and os.path.isfile(session.explorer):
            session.file_path = session.explorer
        else:
            raise ValueError("Invalid file path")

        return session

    def update_explorer(self, new_path):
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

    def create_interface(self):
        with gr.Blocks(css=self.css_path) as gradio_interface:
            gr.Markdown(f"# {self.title}")

            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("### File selection")
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

                    # Run button
                    with gr.Row():
                        btn_run = gr.Button("Run Layers2D", elem_classes="btn btn-run")

                with gr.Column(scale=4):
                    with gr.Row():
                        input_plot = gr.Plot(
                            show_label=True,
                            label="Source image",
                            visible=True,
                            elem_classes="plot",
                        )
                        output_plot = gr.Plot(
                            show_label=True,
                            label="Detected layers",
                            visible=True,
                            elem_classes="plot",
                        )
            # Session
            session = gr.State([])
            pipeline = Pipeline()
            inputs = [base_path, explorer]

            # fmt: off
            reload_base_path.click(fn=self.update_explorer,inputs=base_path, outputs=explorer)

            btn_run.click(
                fn=self.start_session, inputs=inputs, outputs=session).then(
                fn=pipeline.load_data, inputs=session, outputs=session).then(
                fn=pipeline.plot_input_img, inputs=session, outputs=input_plot)

            # fmt: on
        return gradio_interface


class Pipeline:
    def __init__(self):
        self.figsize = (6, 6)

    def plot_input_img(self, session, cmap="viridis"):
        plt.close()
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.imshow(session.data, interpolation="nearest", cmap=cmap)

        # Adjustments
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig

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

        print(session.data)
        return session


if __name__ == "__main__":
    # Creates interface
    gradio_interface = Interface().create_interface()
    internal_tools.run_gradio_app(gradio_interface)
