import tifffile
import os
import numpy as np
import gradio as gr
from qim3d.io import load  # load or DataLoader?
from qim3d.utils import internal_tools


class Interface:
    def __init__(self):
        self.verbose = False
        self.title = "Annotation tool"
        # self.plot_height = 768
        self.height = 1024
        # self.width = 960
        self.max_masks = 3
        self.mask_opacity = 0.5
        self.cmy_hex = ["#00ffff", "#ff00ff", "#ffff00"]  # Colors for max_masks>3?

        # CSS path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.css_path = os.path.join(current_dir, "..", "css", "gradio.css")

    def launch(self, img=None, **kwargs):
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
            # width=self.width,
            show_tips=False,
            **kwargs,
        )

        return

    def get_result(self):
        # Get the temporary files from gradio
        temp_sets = self.interface.temp_file_sets
        for temp_set in temp_sets:
            if "mask" in str(temp_set):
                # Get the list of the temporary files
                temp_path_list = list(temp_set)

        # Files are not in creation order,
        # so we need to get find the latest
        creation_time_list = []
        for path in temp_path_list:
            creation_time_list.append(os.path.getctime(path))

        # Get index for the latest file
        file_idx = np.argmax(creation_time_list)

        # Load the temporary file
        mask = load(temp_path_list[file_idx])

        return mask

    def create_interface(self, img=None):
        with gr.Blocks(css=self.css_path) as gradio_interface:
            masks_state = gr.State(value={})
            counts = gr.Number(value=1, visible=False)

            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    upload_img_btn = gr.UploadButton(
                        label="Upload image",
                        file_types=["image"],
                        interactive=True if img is None else False,
                    )
                    clear_img_btn = gr.Button(
                        value="Clear image", interactive=False if img is None else True
                    )

                    with gr.Row():
                        with gr.Column(scale=2, min_width=32):
                            selected_mask = gr.Radio(
                                choices=["Mask 1"],
                                value="Mask 1",
                                label="Choose which mask to draw",
                                scale=1,
                            )
                        with gr.Column(scale=1, min_width=64):
                            add_mask_btn = gr.Button(
                                value="Add mask",
                                scale=2,
                            )
                    with gr.Row():
                        prep_dl_btn = gr.Button(
                            value="Prepare mask for download",
                            visible=False if img is None else True,
                        )
                    with gr.Row():
                        save_output = gr.File(
                            show_label=True,
                            label="Output file",
                            visible=False,
                        )

                with gr.Column(scale=4):
                    with gr.Row():
                        input_img = gr.Image(
                            label="Input",
                            tool="sketch",
                            value=img,
                            height=600,
                            width=600,
                            brush_color="#00ffff",
                            mask_opacity=self.mask_opacity,
                            interactive=False if img is None else True,
                        )

                    output_masks = []
                    for mask_idx in range(self.max_masks):
                        with gr.Row():  # make a new row for every mask
                            output_mask = gr.Image(
                                label=f"Mask {mask_idx+1}",
                                visible=True if mask_idx == 0 else False,
                                image_mode="L",
                                height=600,
                                width=600,
                                interactive=False
                                if img is None
                                else True,  # If statement added bc of bug after Gradio 3.44.x
                                show_download_button=False,
                            )
                            output_masks.append(output_mask)

            # Operations
            operations = Operations(max_masks=self.max_masks, cmy_hex=self.cmy_hex)

            # Update component configuration when image is uploaded
            upload_img_btn.upload(
                fn=operations.upload_img_update,
                inputs=upload_img_btn,
                outputs=[input_img, clear_img_btn, upload_img_btn, prep_dl_btn]
                + output_masks,
            )

            # Add mask below when 'add mask' button is clicked
            add_mask_btn.click(
                fn=operations.increment_mask,
                inputs=counts,
                outputs=[counts, selected_mask] + output_masks,
            )

            # Draw mask when input image is edited
            input_img.edit(
                fn=operations.update_masks,
                inputs=[input_img, selected_mask, masks_state, upload_img_btn],
                outputs=output_masks,
            )

            # Update brush color according to radio setting
            selected_mask.change(
                fn=operations.update_brush_color,
                inputs=selected_mask,
                outputs=input_img,
            )

            # Make file download visible
            prep_dl_btn.click(
                fn=operations.save_mask,
                inputs=output_masks,
                outputs=[save_output, save_output],
            ).success(
                fn=lambda: os.remove('mask.tif')
                ) # Remove mask file from working directory immediately after sending it to /tmp/gradio

            # Update 'Add mask' button interactivit according to the current count
            counts.change(
                fn=operations.set_add_mask_btn_interactivity,
                inputs=counts,
                outputs=add_mask_btn,
            )

            # Reset component configuration when image is cleared
            clear_img_btn.click(
                fn=operations.clear_img_update,
                inputs=None,
                outputs=[
                    selected_mask,
                    prep_dl_btn,
                    save_output,
                    counts,
                    input_img,
                    upload_img_btn,
                    clear_img_btn,
                ]
                + output_masks,
            )

        return gradio_interface


class Operations:
    def __init__(self, max_masks, cmy_hex):
        self.max_masks = max_masks
        self.cmy_hex = cmy_hex

    def update_masks(self, input_img, selected_mask, masks_state, file):
        # Binarize mask (it is not per default due to anti-aliasing)
        input_mask = input_img["mask"]
        input_mask[input_mask > 0] = 255

        try:
            file_name = file.name
        except AttributeError:
            file_name = "nb_img"

        # Add new file to state dictionary when this function sees it first time
        if file_name not in masks_state.keys():
            masks_state[file_name] = [[] for _ in range(self.max_masks)]

        # Get index of currently selected and non-selected masks
        sel_mask_idx = int(selected_mask[-1]) - 1
        nonsel_mask_idxs = [
            mask_idx
            for mask_idx in list(range(self.max_masks))
            if mask_idx != sel_mask_idx
        ]

        # Add background to state first time function is invoked in current session
        if len(masks_state[file_name][0]) == 0:
            for i in range(len(masks_state[file_name])):
                masks_state[file_name][i].append(input_mask)

        # Check for discrepancy between what is drawn and what is shown as output masks
        masks_state_combined = 0
        for i in range(len(masks_state[file_name])):
            masks_state_combined += masks_state[file_name][i][-1]
        discrepancy = masks_state_combined != input_mask
        if np.any(discrepancy):  # Correct discrepancy in output masks
            for i in range(self.max_masks):
                masks_state[file_name][i][-1][discrepancy] = 0

        # Add most recent change in input to currently selected mask
        mask2append = input_mask
        for mask_idx in nonsel_mask_idxs:
            mask2append -= masks_state[file_name][mask_idx][-1]
        masks_state[file_name][sel_mask_idx].append(mask2append)

        return [masks_state[file_name][i][-1] for i in range(self.max_masks)]

    def save_mask(self, *masks):
        # Go from multi-channel to single-channel mask
        stacked_masks = np.stack(masks, axis=-1)
        final_mask = np.zeros_like(masks[0])
        final_mask[np.where(stacked_masks == 255)[:2]] = (
            np.where(stacked_masks == 255)[-1] + 1
        )

        # Save output image in a temp space (and to current directory which is a bug)
        filename = "mask.tif"
        tifffile.imwrite(filename, final_mask)

        save_output_update = gr.File(visible=True)

        return save_output_update, filename

    def increment_mask(self, counts):
        # increment count by 1
        counts += 1
        counts = int(counts)

        counts_update = gr.Number(value=counts)
        selected_mask_update = gr.Radio(
            value=f"Mask {counts}", choices=[f"Mask {i+1}" for i in range(counts)]
        )
        output_masks_update = [gr.Image(visible=True)] * counts + [
            gr.Image(visible=False)
        ] * (self.max_masks - counts)

        return [counts_update, selected_mask_update] + output_masks_update

    def update_brush_color(self, selected_mask):
        sel_mask_idx = int(selected_mask[-1]) - 1
        if sel_mask_idx < len(self.cmy_hex):
            input_img_update = gr.Image(brush_color=self.cmy_hex[sel_mask_idx])
        else:
            input_img_update = gr.Image(brush_color="#000000")  # Return black brush

        return input_img_update

    def set_add_mask_btn_interactivity(self, counts):
        add_mask_btn_update = (
            gr.Button(interactive=True)
            if counts < self.max_masks
            else gr.Button(interactive=False)
        )
        return add_mask_btn_update

    def clear_img_update(self):
        selected_mask_update = gr.Radio(
            choices=["Mask 1"], value="Mask 1"
        )  # Reset radio component to only show 'Mask 1'
        prep_dl_btn_update = gr.Button(
            visible=False
        )  # Make 'Prepare mask for download' button invisible
        save_output_update = gr.File(visible=False)  # Make File save box invisible
        counts_update = gr.Number(value=1)  # Reset invisible counter to 1
        input_img_update = gr.Image(
            value=None, interactive=False
        )  # Set input image component to non-interactive (so a new image cannot be uploaded directly in the component)
        upload_img_btn_update = gr.Button(
            interactive=True
        )  # Make 'Upload image' button interactive
        clear_img_btn_update = gr.Button(
            interactive=False
        )  # Make 'Clear image' button non-interactive
        output_masks_update = [
            gr.Image(value=None, visible=True if i == 0 else False, interactive=False)
            for i in range(self.max_masks)
        ]  # Remove drawn masks and set as invisible except mask 1. 'interactive=False' added bc of bug after Gradio 3.44.x

        return [
            selected_mask_update,
            prep_dl_btn_update,
            save_output_update,
            counts_update,
            input_img_update,
            upload_img_btn_update,
            clear_img_btn_update,
        ] + output_masks_update

    def upload_img_update(self, file):
        input_img_update = gr.Image(
            value=load(file.name), interactive=True
        )  # Upload image from button to Image components
        clear_img_btn_update = gr.Button(
            interactive=True
        )  # Make 'Clear image' button interactive
        upload_img_btn_update = gr.Button(
            interactive=False
        )  # Make 'Upload image' button non-interactive
        prep_dl_btn_update = gr.Button(
            visible=True
        )  # Make 'Prepare mask for download' button visible
        output_masks_update = [
            gr.Image(interactive=True)
        ] * self.max_masks  # This line is added bc of bug in Gradio after 3.44.x

        return [
            input_img_update,
            clear_img_btn_update,
            upload_img_btn_update,
            prep_dl_btn_update,
        ] + output_masks_update


def run_interface(host = "0.0.0.0"):
    gradio_interface = Interface().create_interface()
    internal_tools.run_gradio_app(gradio_interface,host)

if __name__ == "__main__":
    # Creates interface
    run_interface()