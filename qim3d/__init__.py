import logging

logging.basicConfig(level=logging.ERROR)

from . import io
from . import gui
from . import viz

from . import utils
from . import models
from . import processing


__version__ = "0.3.2"
examples = io.ImgExamples()
io.logger.set_level_info()
