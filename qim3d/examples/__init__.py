""" Example images for testing and demonstration purposes. """

from pathlib import Path as _Path
from qim3d.utils._logger import log as _log
from qim3d.io import load as _load

# Save the original log level and set to ERROR
# to suppress the log messages during loading
_original_log_level = _log.level
_log.setLevel("ERROR")

# Load image examples
for _file_path in _Path(__file__).resolve().parent.glob("*.tif"):
    globals().update({_file_path.stem: _load(_file_path, progress_bar=False)})

# Restore the original log level
_log.setLevel(_original_log_level)
