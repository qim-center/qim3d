import base64

import gradio as gr
from matplotlib.pyplot import colormaps

from qim3d.gui.interface import BaseInterface
from qim3d.io import  save
from qim3d.generate import volume
from qim3d.viz import volumetric

class Interface(BaseInterface):
    def __init__(
        self,
        verbose: bool = False
        ):
        """
        Parameters
        ----------
        verbose (bool, optional): If true, prints info during session into terminal. Defualt is False.
        """
        super().__init__(title='Volume generator', height=1024, width=900, verbose=verbose)
        self.error_message = None
        self.fig = None
        self.og_vol = None
        self.resized_vol = None

    def save_volume(self, extension:str):
        filename = f'generated_volume{extension}'
        save(filename, self.og_vol, replace = True)
        return gr.update(value = filename, visible = True)
    
    def save_plot(self):
        snapshot = self.fig.get_snapshot()
        with open('generated_volume.html', 'w') as f:
            f.write(snapshot)

        return gr.update(value = 'generated_volume.html', visible = True)

    def generate_volume(self,
            noise_type,
            noise,
            gamma,
            decay,
            threshold,
            shape,
            axis,
            tube_hole_ratio,
            hollow,
            seed):
        
        shape = None if shape == 'None' else shape.lower()
        self.og_vol = volume(
                    noise_type = noise_type.lower(),
                    noise_scale = noise,
                    gamma = gamma,
                    decay_rate = decay,
                    threshold = threshold,
                    shape = shape,
                    axis = axis,
                    tube_hole_ratio = tube_hole_ratio,
                    dtype = 'float32',
                    hollow=hollow,
                    seed = seed
                    )

    def plot_volume(self, colormap:str):
        self.fig = volumetric(self.og_vol, show = False, color_map=colormap)
        self.fig.snapshot_type = "inline"
        snapshot  = self.fig.get_snapshot()
        snapshot = base64.b64encode(snapshot.encode("utf-8")).decode("utf-8")
        html = f'<iframe src="data:text/html;base64,{snapshot}"style="width:100%;height:600px;border:0"></iframe>'

        return html, gr.update(visible = False), gr.update(visible = False)

    def toggle_axis(self, shape):
        if shape == 'None':
            return gr.update(visible=False)
        else:
            return gr.update(visible=True)

    def define_interface(self, gradio_interface, *args, **kwargs):
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    noise_type = gr.Dropdown(['Perlin', 'Simplex', 'PNoise', 'P', 'SNoise', 'S'], label = 'Noise type')
                    noise = gr.Slider(0, 0.1, 0.02, label = 'Noise')
                    decay = gr.Slider(0.1, 20, 10, label = 'Decay')
                    gamma = gr.Slider(0.1, 2, 1, label = 'Gamma')
                    threshold = gr.Slider(0, 1, 0.5, label = 'Threshold')
                    shape = gr.Dropdown(['None', 'Tube', 'Cylinder'], label = 'Shape')
                    axis = gr.Slider(0, 2, 0,step = 1, visible=False, label = 'Axis of shape')
                    tube_hole_ration = gr.Slider(0, 1, 0.5, label = 'Tube-hole ratio')
                    hollow = gr.Slider(0, 20, 0, step = 1, label = 'Thickness of hollowing')
                    seed = gr.Slider(0, 1000, 0, step = 1, label = 'Seed')

                    colormap = gr.Dropdown(
                            choices=colormaps,
                            value='magma',
                            label='Colormap',
                        )
                # with gr.Row():
                    # TODO: When they implement this https://github.com/gradio-app/gradio/issues/9230
                    # it would be nice to use it instead of first generate and then download

                with gr.Group():
                    generate_volume = gr.Button('Save volume', variant = 'primary')
                    file_extensions = gr.Dropdown(
                        choices = [
                            '.tiff', 
                            '.nii.gz', 
                            '.h5', 
                            '.vol', 
                            '.dcm', 
                            '.zarr'
                            ], 
                            value = '.tiff', 
                            label = 'File format', 
                            interactive=True)
                    volume_file = gr.File(visible = False)
                # with gr.Row():
                with gr.Group():
                    generate_html = gr.Button('Save interactive plot', variant = 'primary')
                    html_file = gr.File(visible = False)
                        

            with gr.Column(scale= 3):

                viz = gr.HTML()



        volume_inputs = [
            noise_type,
            noise,
            gamma,
            decay,
            threshold,
            shape,
            axis,
            tube_hole_ration,
            hollow,
            seed,
        ]

        display_inputs = [
            colormap
        ]

        viz_outputs = [viz, volume_file, html_file]

        # Change triggers generating new volume and updating layout
        gr.on(triggers = [ input.change for input in volume_inputs],
            fn = self.generate_volume,
            inputs = volume_inputs,
        ).success(fn = self.plot_volume, inputs = display_inputs, outputs = viz_outputs)

        # Changes the display settings
        gr.on(triggers=[ input.change for input in display_inputs],
              fn = self.plot_volume,
              inputs = display_inputs,
              outputs = viz_outputs)
        
        # Axis of shape if only available if shape is not None
        shape.change(self.toggle_axis, inputs=shape, outputs=axis)

        generate_volume.click(fn = self.save_volume, inputs = file_extensions, outputs = volume_file)
        generate_html.click(fn = self.save_plot, outputs = html_file)

        gradio_interface.load(fn = lambda: 420, inputs = None, outputs = seed)