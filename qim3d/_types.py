import os
from pathlib import Path

import matplotlib

PathLike = os.PathLike  # Aliased for discoverability; I often forget about os.PathLike
ColormapLike = str | matplotlib.colors.Colormap
