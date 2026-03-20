import os
import platform
from pathlib import Path
from typing import Callable

import qim3d


class NotInstalledError(Exception):
    pass


SOURCE_FNM = 'fnm env --use-on-cd | Out-String | Invoke-Expression;'

LINUX = 'Linux'
WINDOWS = 'Windows'
MAC = 'Darwin'


def get_volume_explorer_dir() -> Path:
    """Return the path to the bundled Volume Explorer assets inside qim3d."""
    qim_dir = Path(qim3d.__file__).parents[0]
    return qim_dir.joinpath('viz/volume_explorer')


def get_nvm_dir(dir: Path | None = None) -> Path:
    base_dir = dir or get_volume_explorer_dir()
    following_folder = '.nvm' if platform.system() in [LINUX, MAC] else ''
    return base_dir.joinpath(following_folder)


def get_node_binaries_dir(nvm_dir: Path | None = None) -> Path:
    """
    Versions could change in time. This makes sure we use the newest one.

    For Windows we have to pass the argument nvm_dir and it is the volume-explorer dir
    """
    if platform.system() in [LINUX, MAC]:
        following_folder = 'versions/node'
        binaries_folder = 'bin'
    elif platform.system() == WINDOWS:
        following_folder = 'node-versions'
        binaries_folder = 'installation'

    node_folder = (nvm_dir or get_nvm_dir()).joinpath(following_folder)

    # We don't want to throw an error
    # Instead we return None and check the returned value in run.py
    if not os.path.isdir(node_folder):
        return None

    for name in sorted(os.listdir(node_folder))[::-1]:
        path = node_folder.joinpath(name)
        if os.path.isdir(path):
            return path.joinpath(binaries_folder)


def get_viewer_dir(dir: Path | None = None) -> Path:
    base_dir = dir or get_volume_explorer_dir()
    return base_dir.joinpath('viewer_app')


def get_viewer_binaries(viewer_dir: Path | None = None) -> Path:
    following_folder1 = 'node_modules'
    following_folder2 = '.bin'
    viewer_dir = viewer_dir or get_viewer_dir()
    return viewer_dir.joinpath(following_folder1).joinpath(following_folder2)


def run_for_platform(
    linux_func: Callable, windows_func: Callable, macos_func: Callable
):
    this_platform = platform.system()
    if this_platform == LINUX:
        return linux_func()
    elif this_platform == WINDOWS:
        return windows_func()
    elif this_platform == MAC:
        return macos_func()


def lambda_raise(err):
    raise err
