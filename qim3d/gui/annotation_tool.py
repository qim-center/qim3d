"""
The GUI can be launched directly from the command line:

```bash
qim3d gui --annotation-tool
```

Or launched from a python script

```python
import qim3d

app = qim3d.gui.annotation_tool.Interface()
app.launch()
```
"""

import getpass
import os
import tempfile
import time

import gradio as gr
import numpy as np
import tifffile
from PIL import Image

import qim3d.utils
from qim3d.io import load, save
from qim3d.io.logger import log


class Session:
    def __init__(self):
        self.n_masks = 3
        self.img_editor = None
        self.masks_rgb = None
        self.mask_names = {0: "red", 1: "green", 2: "blue"}
        self.temp_files = []
        self.temp_dir = None


class Interface:
    def __init__(self):
        self.verbose = False
        self.title = "Annotation tool"
        self.height = 768
        self.interface = None
        self.username = getpass.getuser()
        self.temp_dir = os.path.join(tempfile.gettempdir(), f"qim-{self.username}")
        self.name_suffix = None

        # CSS path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.css_path = os.path.join(current_dir, "..", "css", "gradio.css")

    def launch(self, img=None, **kwargs):
        # Create gradio interfaces
        # img = "/tmp/qim-fima/2dimage.png"
        self.interface = self.create_interface(img)

        # Set gradio verbose level
        if self.verbose:
            quiet = False
        else:
            quiet = True

        self.interface.launch(
            quiet=quiet,
            height=self.height,
            # width=self.width,
            **kwargs,
        )

        return

    def get_result(self):
        # Get the temporary files from gradio
        temp_path_list = []
        for filename in os.listdir(self.temp_dir):
            if "mask" and self.name_suffix in str(filename):
                # Get the list of the temporary files
                temp_path_list.append(os.path.join(self.temp_dir, filename))

        # Make dictionary of maks
        masks = {}
        for temp_file in temp_path_list:
            mask_file = os.path.basename(temp_file)
            mask_name = os.path.splitext(mask_file)[0]
            masks[mask_name] = load(temp_file)

        return masks

    def set_visible(self):
        return gr.update(visible=True)

    def create_interface(self, img=None):

        if img is not None:
            custom_css = "annotation-tool"
        else:
            custom_css = "annotation-tool no-img"

        with gr.Blocks(css=self.css_path, title=self.title) as gradio_interface:

            brush = gr.Brush(
                colors=[
                    "rgb(255,50,100)",
                    "rgb(50,250,100)",
                    "rgb(50,100,255)",
                ],
                color_mode="fixed",
                default_size=10,
            )
            with gr.Row():

                with gr.Column(scale=6):
                    img_editor = gr.ImageEditor(
                        # ! Temporary fix for drawing at wrong location https://github.com/gradio-app/gradio/pull/7959
                        value=(
                            {
                                "background": img,
                                "layers": [Image.new("RGBA", img.shape, (0, 0, 0, 0))],
                                "composite": None,
                            }
                            if img is not None
                            else None
                        ),
                        type="numpy",
                        image_mode="RGB",
                        brush=brush,
                        sources="upload",
                        interactive=True,
                        show_download_button=True,
                        container=False,
                        transforms=[""],
                        elem_classes=custom_css,
                    )

                with gr.Column(scale=1, min_width=256):

                    with gr.Row():
                        btn_update = gr.Button(
                            value="Update", elem_classes="btn btn-html btn-run"
                        )

                    with gr.Row():
                        overlay_img = gr.Image(
                            show_download_button=False, show_label=False, visible=False
                        )
                    with gr.Row():
                        masks_download = gr.File(
                            label="Download masks",
                            visible=False,
                            elem_classes=custom_css,
                        )

            temp_dir = gr.Textbox(value=self.temp_dir, visible=False)
            name_suffix = gr.Textbox(value=self.name_suffix, visible=False)

            session = gr.State([])
            inputs = [img_editor]
            operations = Operations()
            # fmt: off
            btn_update.click(
                fn=operations.start_session, inputs=[img_editor,temp_dir, name_suffix] , outputs=session).then(
                fn=operations.preview, inputs=session, outputs=overlay_img).then(
                fn=self.set_visible, inputs=None, outputs=overlay_img).then(
                fn=operations.separate_masks, inputs=session, outputs=[session, masks_download]).then(
                fn=self.set_visible, inputs=None, outputs=masks_download)

            # fmt: on
        return gradio_interface


class Operations:

    def start_session(self, *args):
        session = Session()
        session.img_editor = args[0]
        session.temp_dir = args[1]
        session.mask_names = {
            0: f"red{args[2]}",
            1: f"green{args[2]}",
            2: f"blue{args[2]}",
        }

        # Clean up old files
        try:
            files = os.listdir(session.temp_dir)
            for filename in files:
                # Check if "mask" is in the filename
                if "mask" and args[2] in filename:
                    file_path = os.path.join(session.temp_dir, filename)
                    os.remove(file_path)

        except FileNotFoundError:
            files = None

        return session

    def overlay_images(self, background, masks, alpha=0.5):
        """Overlay multiple RGB masks onto an RGB background image using alpha blending.

        Args:
            background (numpy.ndarray): The background RGB image with shape (height, width, 3).
            masks (numpy.ndarray): The RGB mask images with shape (num_masks, height, width, 3).
            alpha (float, optional): The alpha value for blending. Defaults to 0.5.

        Returns:
            numpy.ndarray: The composite image with overlaid masks.

        Raises:
            ValueError: If input images have different shapes.

        Note:
            - The function performs alpha blending to overlay the masks onto the background.
            - It ensures that the background and masks have the same shape before blending.
            - It calculates the maximum projection of the masks and blends them onto the background.
            - Brightness outside the masks is adjusted to maintain consistency with the background.
        """

        # Igonore alpha in case its there
        background = background[..., :3]
        masks = masks[..., :3]

        # Ensure both images have the same shape
        if background.shape != masks.shape:
            raise ValueError("Input images must have the same shape")

        # Perform alpha blending
        masks_max_projection = np.amax(masks, axis=2)
        masks_max_projection = np.stack((masks_max_projection,) * 3, axis=-1)

        # Normalize if we have something
        if np.max(masks_max_projection) > 0:
            masks_max_projection = masks_max_projection / np.max(masks_max_projection)

        composite = background * (1 - alpha) + masks * alpha
        composite = np.clip(composite, 0, 255).astype("uint8")

        # Adjust brightness outside masks
        composite = composite + (background * (1 - alpha)) * (1 - masks_max_projection)

        return composite.astype("uint8")

    def preview(self, session):
        background = session.img_editor["background"]
        masks = session.img_editor["layers"][0]
        overlay_image = qim3d.utils.img.overlay_rgb_images(background, masks)

        return overlay_image

    def separate_masks(self, session):

        masks_rgb = session.img_editor["layers"][0]
        mask_threshold = 200  # This value is based

        mask_list = []
        files_list = []

        # Go through each channel
        for idx in np.arange(session.n_masks):

            mask_grayscale = masks_rgb[:, :, idx]
            mask = mask_grayscale > mask_threshold

            # Save only if we have a mask
            if np.sum(mask) > 0:
                mask_list.append(mask)
                filename = f"mask_{session.mask_names[idx]}.tif"
                if not os.path.exists(session.temp_dir):
                    os.makedirs(session.temp_dir)
                filepath = os.path.join(session.temp_dir, filename)
                files_list.append(filepath)

                save(filepath, mask, replace=True)
                session.temp_files.append(filepath)

        return session, files_list


def run_interface(host="0.0.0.0"):
    gradio_interface = Interface().create_interface()
    qim3d.utils.internal_tools.run_gradio_app(gradio_interface, host)


if __name__ == "__main__":
    # Creates interface
    run_interface()
