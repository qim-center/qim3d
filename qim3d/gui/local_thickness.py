"""
!!! quote "Reference"
    Dahl, V. A., & Dahl, A. B. (2023, June). Fast Local Thickness. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 
    <https://doi.org/10.1109/cvprw59228.2023.00456>

    ```bibtex
    @inproceedings{Dahl_2023, title={Fast Local Thickness}, 
    url={http://dx.doi.org/10.1109/CVPRW59228.2023.00456}, 
    DOI={10.1109/cvprw59228.2023.00456}, 
    booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
    publisher={IEEE}, 
    author={Dahl, Vedrana Andersen and Dahl, Anders Bjorholm}, 
    year={2023}, 
    month=jun }

    ```


The GUI can be launched directly from the command line:

```bash
qim3d gui --local-thickness
```

Or launched from a python script

```python
import qim3d

app = qim3d.gui.local_thickness.Interface()
app.launch()
```

"""
import os

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import tifffile
import localthickness as lt
import qim3d



class Interface(qim3d.gui.interface.InterfaceWithExamples):
    def __init__(self,
                 img: np.ndarray = None,
                 verbose:bool = False,
                 plot_height:int = 768,
                 figsize:int = 6): 
        
        super().__init__(title = "Local thickness",
                       height = 1024,
                       width = 960,
                       verbose = verbose)

        self.plot_height = plot_height
        self.figsize = figsize
        self.img = img

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
        vol_lt = qim3d.io.load(temp_path_list[file_idx])

        return vol_lt

    def define_interface(self):
        gr.Markdown(
        "Interface for _Fast local thickness in 3D_ (https://github.com/vedranaa/local-thickness)"
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                if self.img is not None:
                    data = gr.State(value=self.img)
                else:
                    with gr.Tab("Input"):
                        data = gr.File(
                            show_label=False,
                            value=self.img,
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
                        info="Inverts the image before thresholding. Use in case your foreground is darker than the background.",
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
                            "Run local thickness", variant = "primary"
                        )
                    with gr.Column(scale=1, min_width=64):
                        btn_clear = gr.Button("Clear", variant = "stop")

                
            with gr.Column(scale=4):
                def create_uniform_image(intensity=1):
                    """
                    Generates a blank image with a single color.
                    Gradio `gr.Plot` components will flicker if there is no default value.
                    bug fix on gradio version 4.44.0
                    """
                    pixels = np.zeros((100, 100, 3), dtype=np.uint8) + int(intensity * 255)
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(pixels, interpolation="nearest")

                    # Adjustments
                    ax.axis("off")
                    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    return fig
                
                with gr.Row():
                    input_vol = gr.Plot(
                        show_label=True,
                        label="Original",
                        visible=True,
                        value=create_uniform_image(),
                    )

                    binary_vol = gr.Plot(
                        show_label=True,
                        label="Binary",
                        visible=True,
                        value=create_uniform_image(),
                    )

                    output_vol = gr.Plot(
                        show_label=True,
                        label="Local thickness",
                        visible=True,
                        value=create_uniform_image(),
                    )
                with gr.Row():
                    histogram = gr.Plot(
                        show_label=True,
                        label="Thickness histogram",
                        visible=True,
                        value=create_uniform_image(),
                    )
                with gr.Row():
                    lt_output = gr.File(
                        interactive=False,
                        show_label=True,
                        label="Output file",
                        visible=False,
                    )


        # Run button
        # fmt: off
        viz_input = lambda zpos, cmap: self.show_slice(self.vol, zpos, self.vmin, self.vmax, cmap)
        viz_binary = lambda zpos, cmap: self.show_slice(self.vol_binary, zpos, None, None, cmap)
        viz_output = lambda zpos, cmap: self.show_slice(self.vol_thickness, zpos, self.vmin_lt, self.vmax_lt, cmap)

        btn.click(
            fn=self.process_input, inputs = [data, dark_objects], outputs = []).success(
            fn=viz_input, inputs = [zpos, cmap_original], outputs = input_vol).success(
            fn=self.make_binary, inputs = threshold, outputs = []).success(
            fn=viz_binary, inputs = [zpos, cmap_original], outputs = binary_vol).success(
            fn=self.compute_localthickness, inputs = lt_scale, outputs = []).success(
            fn=viz_output, inputs = [zpos, cmap_lt], outputs = output_vol).success(
            fn=self.thickness_histogram, inputs = nbins, outputs = histogram).success(
            fn=self.save_lt, inputs = [], outputs = lt_output).success(
            fn=self.remove_unused_file).success(
            fn=self.set_visible, inputs= [], outputs=lt_output)

        # Clear button
        outputs = [input_vol, output_vol, binary_vol, histogram, lt_output]
        for gr_obj in outputs:
            btn_clear.click(fn=self.clear, inputs=None, outputs=gr_obj)

        btn_clear.click(fn = self.set_invisible, inputs = [], outputs = lt_output)


        # Event listeners
        zpos.change(
            fn=viz_input, inputs = [zpos, cmap_original], outputs=input_vol, show_progress=False).success(
            fn=viz_binary, inputs = [zpos, cmap_original], outputs=binary_vol, show_progress=False).success(
            fn=viz_output, inputs = [zpos, cmap_lt], outputs=output_vol, show_progress=False)
        
        cmap_original.change(
            fn=viz_input, inputs = [zpos, cmap_original],outputs=input_vol, show_progress=False).success(
            fn=viz_binary, inputs = [zpos, cmap_original], outputs=binary_vol, show_progress=False)
        
        cmap_lt.change(
            fn=viz_output, inputs = [zpos, cmap_lt], outputs=output_vol, show_progress=False
        )

        nbins.change(
            fn = self.thickness_histogram, inputs = nbins, outputs = histogram
        )
            # fmt: on

    #######################################################
    #
    #       PIPELINE
    #
    #######################################################

    def process_input(self, data: np.ndarray, dark_objects: bool):
        # Load volume
        try:
            self.vol = qim3d.io.load(data.name)
            assert self.vol.ndim == 3
        except AttributeError:
            self.vol = data
        except AssertionError:
            raise gr.Error(F"File has to be 3D structure. Your structure has {self.vol.ndim} dimension{'' if self.vol.ndim == 1 else 's'}")

        if dark_objects:
            self.vol = np.invert(self.vol)

        # Get min and max values for visualization
        self.vmin = np.min(self.vol)
        self.vmax = np.max(self.vol)

    def show_slice(self, vol: np.ndarray, zpos: int, vmin: float = None, vmax: float = None, cmap: str = "viridis"):
        plt.close()
        z_idx = int(zpos * (vol.shape[0] - 1))
        fig, ax = plt.subplots(figsize=(self.figsize, self.figsize))

        ax.imshow(vol[z_idx], interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

        # Adjustments
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        return fig

    def make_binary(self, threshold: float):
        # Make a binary volume
        # Nothing fancy, but we could add new features here
        self.vol_binary = self.vol > (threshold * np.max(self.vol))
    
    def compute_localthickness(self, lt_scale: float):
        self.vol_thickness = lt.local_thickness(self.vol_binary, lt_scale)

        # Valus for visualization
        self.vmin_lt = np.min(self.vol_thickness)
        self.vmax_lt = np.max(self.vol_thickness)

    def thickness_histogram(self, nbins: int):
        # Ignore zero thickness
        non_zero_values = self.vol_thickness[self.vol_thickness > 0]

        # Calculate histogram
        vol_hist, bin_edges = np.histogram(non_zero_values, nbins)

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

    def save_lt(self):
        filename = "localthickness.tif"
        # Save output image in a temp space
        tifffile.imwrite(filename, self.vol_thickness)

        return filename
    
    def remove_unused_file(self):
        # Remove localthickness.tif file from working directory
        # as it otherwise is not deleted
        os.remove('localthickness.tif')
    
if __name__ == "__main__":
    Interface().run_interface()