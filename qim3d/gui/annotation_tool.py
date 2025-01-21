"""
The GUI can be launched directly from the command line:

```bash
qim3d gui --annotation-tool
```

Or launched from a python script

```python
import qim3d

vol = qim3d.examples.NT_128x128x128
annotation_tool = qim3d.gui.annotation_tool.Interface()

# We can directly pass the image we loaded to the interface
app = annotation_tool.launch(vol[0])
```
![gui-annotation_tool](../../assets/screenshots/gui-annotation_tool.gif)

"""

import getpass
import os
import tempfile

import gradio as gr
import numpy as np
from PIL import Image
import qim3d
from qim3d.gui.interface import BaseInterface

# TODO: img in launch should be self.img


class Interface(BaseInterface):
    def __init__(self, name_suffix: str = "", verbose: bool = False, img: np.ndarray = None):
        super().__init__(
            title="Annotation Tool",
            height=768,
            width="100%",
            verbose=verbose,
            custom_css="annotation_tool.css",
        )

        self.username = getpass.getuser()
        self.temp_dir = os.path.join(tempfile.gettempdir(), f"qim-{self.username}")
        self.name_suffix = name_suffix
        self.img = img

        self.n_masks = 3
        self.img_editor = None
        self.masks_rgb = None
        self.temp_files = []

    def get_result(self) -> dict:
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
            masks[mask_name] = qim3d.io.load(temp_file)

        return masks

    def clear_files(self):
        """
        Should be moved up to __init__ function, but given how is this interface implemented in some files
        this is safer and backwards compatible (should be)
        """
        self.mask_names = [
            f"red{self.name_suffix}",
            f"green{self.name_suffix}",
            f"blue{self.name_suffix}",
        ]

        # Clean up old files
        try:
            files = os.listdir(self.temp_dir)
            for filename in files:
                # Check if "mask" is in the filename
                if ("mask" in filename) and (self.name_suffix in filename):
                    file_path = os.path.join(self.temp_dir, filename)
                    os.remove(file_path)

        except FileNotFoundError:
            files = None

    def create_preview(self, img_editor: gr.ImageEditor) -> np.ndarray:
        background = img_editor["background"]
        masks = img_editor["layers"][0]
        overlay_image = qim3d.operations.overlay_rgb_images(background, masks)
        return overlay_image

    def cerate_download_list(self, img_editor: gr.ImageEditor) -> list[str]:
        masks_rgb = img_editor["layers"][0]
        mask_threshold = 200  # This value is based

        mask_list = []
        files_list = []

        # Go through each channel
        for idx in range(self.n_masks):
            mask_grayscale = masks_rgb[:, :, idx]
            mask = mask_grayscale > mask_threshold

            # Save only if we have a mask
            if np.sum(mask) > 0:
                mask_list.append(mask)
                filename = f"mask_{self.mask_names[idx]}.tif"
                if not os.path.exists(self.temp_dir):
                    os.makedirs(self.temp_dir)
                filepath = os.path.join(self.temp_dir, filename)
                files_list.append(filepath)

                qim3d.io.save(filepath, mask, replace=True)
                self.temp_files.append(filepath)

        return files_list

    def define_interface(self, **kwargs):
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
            with gr.Column(
                scale=6,
            ):
                img_editor = gr.ImageEditor(
                    value=(
                        {
                            "background": self.img,
                            "layers": [Image.new("RGBA", self.img.shape, (0, 0, 0, 0))],
                            "composite": None,
                        }
                        if self.img is not None
                        else None
                    ),
                    type="numpy",
                    image_mode="RGB",
                    brush=brush,
                    sources="upload",
                    interactive=True,
                    show_download_button=True,
                    container=False,
                    transforms=["crop"],
                    layers=False,
                )

            with gr.Column(scale=1, min_width=256):

                with gr.Row():
                    overlay_img = gr.Image(
                        show_download_button=False,
                        show_label=False,
                        visible=False,
                    )
                with gr.Row():
                    masks_download = gr.File(label="Download masks", visible=False)

        # fmt: off
        img_editor.change(
            fn = self.clear_files, inputs = None , outputs = None).then(                        # Prepares for handling the new update
            fn = self.create_preview, inputs = img_editor, outputs = overlay_img).then(         # Create the preview in top right corner                                           
            fn = self.set_visible, inputs = None, outputs = overlay_img).then(                  # Makes the preview visible
            fn = self.cerate_download_list, inputs = img_editor, outputs = masks_download).then(# Separates the color mask and put them into file list
            fn = self.set_visible, inputs = None, outputs = masks_download)                     # Displays the download file list
