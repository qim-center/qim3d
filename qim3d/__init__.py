import logging

logging.basicConfig(level=logging.ERROR)

from qim3d import io
from qim3d import gui
from qim3d import viz
from qim3d import utils
from qim3d import models
from qim3d import processing

__version__ = "0.3.2"
examples = io.ImgExamples()
io.logger.set_level_info()
