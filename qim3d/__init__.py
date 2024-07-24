"""qim3d: A Python package for 3D image processing and visualization.

The qim3d library is designed to make it easier to work with 3D imaging data in Python. 
It offers a range of features, including data loading and manipulation,
 image processing and filtering, visualization of 3D data, and analysis of imaging results.

Documentation available at https://platform.qim.dk/qim3d/

"""

__version__ = "0.3.9"

from . import io
from . import gui
from . import viz
from . import utils
from . import processing
from . import generate

# commented out to avoid torch import
# from . import models

examples = io.ImgExamples()
io.logger.set_level_info()
