"""Example images for testing and demonstration purposes."""

from __future__ import annotations

from pathlib import Path as _Path
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from numpy import ndarray as _ndarray

bone_128x128x128: _ndarray
cement_128x128x128: _ndarray
fibers_150x150x150: _ndarray
fly_150x256x256: _ndarray
NT_128x128x128: _ndarray
shell_225x128x128: _ndarray

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
    globals()[name] = volume
    return volume
