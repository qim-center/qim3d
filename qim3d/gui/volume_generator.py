import datetime
import os
import re
from typing import Any, Callable, Dict
import tempfile

import gradio as gr
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import outputformat as ouf
import plotly.graph_objects as go

from qim3d.gui.interface import BaseInterface, COLORMAPS
from qim3d.io import load, save
from qim3d.utils import _misc
from qim3d.utils._logger import log
from qim3d.generate import volume
from scipy import ndimage

class Interface(BaseInterface):
    def __init__(
        self,
        verbose: bool = False,
        figsize: int = 8,
        display_saturation_percentile: int = 99,
    ):
        """
        Parameters
        ----------
        verbose (bool, optional): If true, prints info during session into terminal. Defualt is False.
        figsize (int, optional): Sets the size of plots displaying the slices. Default is 8.
        display_saturation_percentile (int, optional): Sets the display saturation percentile. Defaults to 99.

        """
        super().__init__(title='Volume generator', height=1024, width=900, verbose=verbose)
        self.error_message = None
        self.fig = None
        self.og_vol = None
        self.resized_vol = None

    def save_volume(self, extension:str):
        if self.og_vol is None:
            gr.Warning('There is no volume to download.')
            return None

        filename = f'generated_volume{extension}'
        save(filename, self.og_vol, replace = True)
        return gr.update(value = filename, visible = True)
    
    def save_plot(self):
        if self.fig is None:
            gr.Warning('There is no plot to dowlnoad.')
            return None
        
        self.fig.write_html('generated_volume.html')
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
                    seed = seed
                    )

    def resize_vol(self, display_size: int):
        """Resizes the loaded volume to the display size"""

        # Get original size
        vol = self.og_vol
        original_Z, original_Y, original_X = np.shape(vol)
        max_size = np.max([original_Z, original_Y, original_X])
        if self.verbose:
            log.info(f'\nOriginal volume: {original_Z, original_Y, original_X}')

        # Resize for display
        self.resized_vol = ndimage.zoom(
            input=vol,
            zoom=display_size / max_size,
            order=0,
            prefilter=False,
        )

        if self.verbose:
            log.info(
                f'Resized volume: {vol.shape}'
            )

    def plot_volume(self, colormap:str, visible_axes:bool, opacity:float):
        z, y, x = np.indices(self.resized_vol.shape)
    
        self.fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=self.resized_vol.flatten(),
            isomin=self.resized_vol.max()*0.1,  # lower threshold
            isomax=self.resized_vol.max(),  # upper threshold
            opacity=opacity,       # transparency
            surface_count=5,  # number of contour surfaces
            colorscale=colormap,
        ))    
        self.fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),  # remove layout padding
            scene=dict(
                xaxis=dict(visible=visible_axes),
                yaxis=dict(visible=visible_axes),
                zaxis=dict(visible=visible_axes),
                aspectmode='data'  # ensures axes scale matches data
            ),
            height = self.height,
            hovermode = False
        ) 
        return self.fig, gr.update(visible = False), gr.update(visible = False)

    def toggle_axis(self, shape):
        if shape == 'None':
            return gr.update(visible=False)
        else:
            return gr.update(visible=True)

    def define_interface(self, **kwargs):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Tab('Noise settings'):
                        noise_type = gr.Dropdown(['Perlin', 'Simplex', 'PNoise', 'P', 'SNoise', 'S'], label = 'Noise type')
                        noise = gr.Slider(0, 0.1, 0.02, label = 'Noise')
                        decay = gr.Slider(0.1, 20, 10, label = 'Decay')
                        gamma = gr.Slider(0.1, 2, 1, label = 'Gamma')
                        threshold = gr.Slider(0, 1, 0.5, label = 'Threshold')
                        shape = gr.Dropdown(['None', 'Tube', 'Cylinder'], label = 'Shape')
                        axis = gr.Slider(0, 2, 0,step = 1, visible=False, label = 'Axis of shape')
                        tube_hole_ration = gr.Slider(0, 1, 0.5, label = 'Tube-hole ratio')
                        hollow = gr.Slider(0, 20, 0, step = 1, label = 'Thickness of hollowing')
                        seed = gr.Slider(0, 1000, 420, step = 1, label = 'Seed')
                    with gr.Tab('Display settings'):
                        display_resolution = gr.Slider(32, 128, 64, step = 4, label = 'Display resolution')
                        visible_axes = gr.Checkbox(True, label = 'Visible axes')
                        opacity = gr.Slider(0, 1, 0.6, step = 0.05, label = 'Opacity')
                        colormap = gr.Dropdown(
                                choices=COLORMAPS,
                                value='Viridis',
                                label='Colormap',
                            )
                with gr.Row():
                    # with gr.Gallery():
                    # TODO: When they implement this https://github.com/gradio-app/gradio/issues/9230
                    # it would be nice to use it instead of first generate and then download

                    with gr.Group():
                        generate_volume = gr.Button('Generate volume', variant = 'primary')
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
                with gr.Row():
                    with gr.Group():
                        generate_html = gr.Button('Generate .html', variant = 'primary')
                        html_file = gr.File(visible = False)
                        

            with gr.Column(scale= 3):
                viz = gr.Plot()



        volume_inputs = [
            noise_type,
            noise,
            gamma,
            decay,
            threshold,
            shape,
            axis,
            tube_hole_ration,
            seed
        ]

        display_inputs = [
            colormap,
            visible_axes,
            opacity,
        ]

        viz_outputs = [viz, volume_file, html_file]
        # CHange triggers generating new volume and updating layout
        gr.on(triggers = [ input.change for input in volume_inputs],
            fn = self.generate_volume,
            inputs = volume_inputs,
        ).success(fn = self.resize_vol, inputs = display_resolution
        ).success(fn = self.plot_volume, inputs = display_inputs, outputs = viz_outputs)

        # Changes the display resolution and updates the layout
        display_resolution.change(fn = self.resize_vol, inputs = display_resolution
        ).success(fn = self.plot_volume, inputs = display_inputs, outputs = viz_outputs)

        # Change triggers updating layout
        gr.on(triggers = [input.change for input in display_inputs],
              fn = self.plot_volume, 
              inputs = display_inputs,
              outputs = viz_outputs)
        
        # Axis of shape if only available if shape is not None
        shape.change(self.toggle_axis, inputs=shape, outputs=axis)

        generate_volume.click(fn = self.save_volume, inputs = file_extensions, outputs = volume_file)
        generate_html.click(fn = self.save_plot, outputs = html_file)
