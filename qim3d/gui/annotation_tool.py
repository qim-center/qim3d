import tifffile
import os
import time
import getpass
import numpy as np
import gradio as gr
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
        self.gradio_temp = None
        self.username = getpass.getuser()


class Interface:
    def __init__(self):
        self.verbose = False
        self.title = "Annotation tool"
        self.height = 768
        self.interface = None
        self.username = getpass.getuser()

        # CSS path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.css_path = os.path.join(current_dir, "..", "css", "gradio.css")

    def launch(self, img=None, **kwargs):
        # Create gradio interfaces

        self.interface = self.create_interface(img)
        self.gradio_temp = self.interface.GRADIO_CACHE

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
        base = os.path.join(self.gradio_temp, "qim3d", self.username)
        temp_path_list = []
        for filename in os.listdir(base):
            if "mask" in str(filename):
                # Get the list of the temporary files
                temp_path_list.append(os.path.join(base, filename))

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

        with gr.Blocks(css=self.css_path) as gradio_interface:

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
                        value=img,
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

            temp_path = gr.Textbox(value=gradio_interface.GRADIO_CACHE, visible=False)
            session = gr.State([])
            inputs = [img_editor]
            operations = Operations()
            # fmt: off
            btn_update.click(
                fn=operations.start_session, inputs=[img_editor,temp_path] , outputs=session).then(
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
        session.gradio_temp = args[1]

        # Clean temp files
        base = os.path.join(session.gradio_temp, "qim3d", session.username)

        try:
            files = os.listdir(base)
            for filename in files:
                # Check if "mask" is in the filename
                if "mask" in filename:
                    file_path = os.path.join(base, filename)
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
                base = os.path.join(session.gradio_temp, "qim3d", session.username)
                if not os.path.exists(base):
                    os.makedirs(base)
                filepath = os.path.join(base, filename)
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
