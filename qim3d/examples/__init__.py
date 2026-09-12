"""Example images for testing and demonstration purposes."""

from __future__ import annotations

from pathlib import Path as _Path
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

bone_128x128x128: npt.NDArray[np.uint8]
cement_128x128x128: npt.NDArray[np.uint8]
fibers_150x150x150: npt.NDArray[np.uint8]
fly_150x256x256: npt.NDArray[np.uint8]
NT_128x128x128: npt.NDArray[np.uint8]
shell_225x128x128: npt.NDArray[np.uint8]

# All annotations above this line define the public example volumes.
_volume_names = tuple(__annotations__)

_examples_dir = _Path(__file__).resolve().parent
_volume_paths = {name: _examples_dir / f"{name}.tif" for name in _volume_names}


def __dir__() -> list[str]:
    """Return the names of the public example volumes."""
    return list(_volume_names)


def __getattr__(name: str):
    """Dynamically load an example volume into memory and cache it."""
    if name not in _volume_paths:
        raise AttributeError(
            f"No example volume named {name!r} in {__name__}; "
            f"available volumes: {', '.join(list(_volume_names))}"
        )

    from qim3d.io import load

    volume = load(_volume_paths[name])
    # __getattr__ is called only when the attribute is missing from the module.
    globals()[name] = volume
    return volume
