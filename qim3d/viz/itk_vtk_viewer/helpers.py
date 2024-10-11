from pathlib import Path
import os
import platform
from typing import Callable

import qim3d

class NotInstalledError(Exception): pass

SOURCE_FNM = "fnm env --use-on-cd | Out-String | Invoke-Expression;"

LINUX = 'Linux'
WINDOWS = 'Windows'
MAC = 'Darwin'

def get_itk_dir() -> Path:
    qim_dir = Path(qim3d.__file__).parents[0] #points to .../qim3d/qim3d/
    dir = qim_dir.joinpath("viz/itk_vtk_viewer")
    return dir

def get_nvm_dir(dir:Path = None) -> Path:
    if platform.system() in [LINUX, MAC]:
        following_folder = ".nvm"
    elif platform.system() == WINDOWS:
        following_folder = ''
    return dir.joinpath(following_folder) if dir is not None else get_qim_dir().joinpath(following_folder)

def get_node_binaries_dir(nvm_dir:Path = None) -> Path:
    """
    Versions could change in time. This makes sure we use the newest one.

    For windows we have to pass the argument nvm_dir and it is the itk-vtk_dir
    """
    if platform.system() in [LINUX, MAC]:
        following_folder = "versions/node"
        binaries_folder = 'bin'
    elif platform.system() == WINDOWS:
        following_folder = 'node-versions'
        binaries_folder = 'installation'

    node_folder =  nvm_dir.joinpath(following_folder) if nvm_dir is not None else get_nvm_dir().joinpath(following_folder)
    
    # We don't wanna throw an error
    # Instead we return None and check the returned value in run.py
    if not os.path.isdir(node_folder):
        return None
    l = sorted(os.listdir(node_folder))

    for name in l[::-1]:
        path = node_folder.joinpath(name)
        if os.path.isdir(path):
            return path.joinpath(binaries_folder)
    
def get_viewer_dir(dir:Path = None) -> Path:
    following_folder = "viewer_app"
    return dir.joinpath(following_folder) if dir is not None else get_qim_dir().joinpath(following_folder)

def get_viewer_binaries(viewer_dir:Path = None) -> Path:
    following_folder1 = 'node_modules'
    following_folder2 = '.bin'
    if viewer_dir is None:
        viewer_dir = get_viewer_dir()
    return viewer_dir.joinpath(following_folder1).joinpath(following_folder2)

def run_for_platform(linux_func:Callable, windows_func:Callable, macos_func:Callable):
    this_platform = platform.system()
    if this_platform == LINUX:
        return linux_func()
    elif this_platform == WINDOWS:
        return windows_func()
    elif this_platform == MAC:
        return macos_func()
    
def lambda_raise(err):
    raise err
