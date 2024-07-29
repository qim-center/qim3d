from pathlib import Path
from abc import abstractmethod, ABC
from os import path

import gradio as gr

from .qim_theme import QimTheme
import qim3d.gui


# TODO: when offline it throws an error in cli
class BaseInterface(ABC):
    """
    Annotation tool and Data explorer as those don't need any examples.
    """

    def __init__(
        self,
        title: str,
        height: int,
        width: int = "100%",
        verbose: bool = False,
        custom_css: str = None,
    ):
        """
        title: Is displayed in tab
        height, width: If inline in launch method is True, sets the paramters of the widget. Inline defaults to True in py notebooks, otherwise is False
        verbose: If True, updates are printed into terminal
        custom_css: Only the name of the file in the css folder.
        """

        self.title = title
        self.height = height
        self.width = width
        self.verbose = bool(verbose)
        self.interface = None

        self.qim_dir = Path(qim3d.__file__).parents[0]
        self.custom_css = (
            path.join(self.qim_dir, "css", custom_css)
            if custom_css is not None
            else None
        )

    def set_visible(self):
        return gr.update(visible=True)

    def set_invisible(self):
        return gr.update(visible=False)

    def launch(self, img=None, force_light_mode: bool = True, **kwargs):
        """
        img: If None, user can upload image after the interface is launched.
            If defined, the interface will be launched with the image already there
            This argument is used especially in jupyter notebooks, where you can launch
            interface in loop with different picture every step
        force_light_mode: The qim platform doesn't have night mode. The qim_theme thus
            has option to display only light mode so it corresponds with the website. Preferably
            will be removed as we add night mode to the website.
        """

        # Create gradio interface
        if img is not None:
            self.img = img
        self.interface = self.create_interface(force_light_mode=force_light_mode)

        self.interface.launch(
            quiet=not self.verbose,
            height=self.height,
            width=self.width,
            favicon_path=Path(qim3d.__file__).parents[0]
            / "../docs/assets/qim3d-icon.svg",
            **kwargs,
        )

    def clear(self):
        """Used to reset outputs with the clear button"""
        return None

    def create_interface(self, force_light_mode: bool = True, **kwargs):
        # kwargs["img"] = self.img
        with gr.Blocks(
            theme=qim3d.gui.QimTheme(force_light_mode=force_light_mode),
            title=self.title,
            css=self.custom_css,
        ) as gradio_interface:
            gr.Markdown(f"# {self.title}")
            self.define_interface(**kwargs)
        return gradio_interface

    @abstractmethod
    def define_interface(self, **kwargs):
        pass

    def run_interface(self, host: str = "0.0.0.0"):
        qim3d.gui.run_gradio_app(self.create_interface(), host)


class InterfaceWithExamples(BaseInterface):
    """
    For Iso3D and Local Thickness
    """

    def __init__(
        self,
        title: str,
        height: int,
        width: int,
        verbose: bool = False,
        custom_css: str = None,
    ):
        super().__init__(title, height, width, verbose, custom_css)
        self._set_examples_list()

    def _set_examples_list(self):
        examples = [
            "blobs_256x256x256.tif",
            "fly_150x256x256.tif",
            "cement_128x128x128.tif",
            "NT_10x200x100.tif",
            "NT_128x128x128.tif",
            "shell_225x128x128.tif",
            "bone_128x128x128.tif",
        ]
        self.img_examples = []
        for example in examples:
            self.img_examples.append([path.join(self.qim_dir, "img_examples", example)])
