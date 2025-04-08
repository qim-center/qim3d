"""Helper functions for testing"""

import os
import inspect
import importlib
import shutil
import socket
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mktestdocs import grab_code_blocks

from qim3d.utils._logger import log
from qim3d.io import save


def mock_plot():
    """
    Creates a mock plot of a sine wave.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.

    Example:
        Creates a mock plot of a sine wave and displays the plot using `plt.show()`.

        >>> fig = mock_plot()
        >>> plt.show()

    """

    matplotlib.use('Agg')

    fig = plt.figure(figsize=(5, 4))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    values = np.arange(0, 2 * np.pi, 0.01)
    axes.plot(values, np.sin(values))

    return fig


def mock_write_file(path, content='File created by qim3d'):
    """
    Creates a file at the specified path and writes a predefined text into it.

    Args:
        path (str): The path to the file to be created.

    Example:
        >>> mock_write_file("example.txt")

    """
    _file = open(path, 'w', encoding='utf-8')
    _file.write(content)
    _file.close()


def is_server_running(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False


def temp_data(folder, remove=False, n=3, img_shape=(32, 32, 32)):
    """
    Creates a temporary folder to test deep learning tools.

    Creates two folders, 'train' and 'test', who each also have two subfolders 'images' and 'labels'.
    n random images are then added to all four subfolders.
    If the 'remove' variable is True, the folders and their content are removed.

    Args:
        folder (str): The path where the folders should be placed.
        remove (bool, optional): If True, all folders are removed from their location.
        n (int, optional): Number of random images and labels in the temporary dataset.
        img_shape (tuple, options): Tuple with the height and width of the images and labels.

    Example:
        >>> tempdata('temporary_folder', n = 10, img_shape = (16, 16, 16))

    """
    folder_trte = ['train', 'test']
    sub_folders = ['images', 'labels']

    # Creating train/test folder
    path_train = Path(folder) / folder_trte[0]
    path_test = Path(folder) / folder_trte[1]

    # Creating folders for images and labels
    path_train_im = path_train / sub_folders[0]
    path_train_lab = path_train / sub_folders[1]
    path_test_im = path_test / sub_folders[0]
    path_test_lab = path_test / sub_folders[1]

    # Random image
    img = np.random.randint(2, size=img_shape, dtype=np.uint8)

    if not os.path.exists(path_train):
        os.makedirs(path_train_im)
        os.makedirs(path_test_im)
        os.makedirs(path_train_lab)
        os.makedirs(path_test_lab)
        for i in range(n):

            save(os.path.join(path_train_im, f'img_train{i}.nii.gz'), img, compression = True, replace = True)
            save(os.path.join(path_train_lab, f'img_train{i}.nii.gz'), img, compression = True, replace = True)
            save(os.path.join(path_test_im, f'img_train{i}.nii.gz'), img, compression = True, replace = True)
            save(os.path.join(path_test_lab, f'img_train{i}.nii.gz'), img, compression = True, replace = True)

    if remove:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                log.warning('Failed to delete %s. Reason: %s' % (file_path, e))

        os.rmdir(folder)

def get_all_functions_by_module():
    """ 
    Creates and returns a dictionary of functions from the qim3d modules.
    """

    # List of qim3d modules
    # TODO: Get this list automatically from the qim3d package information
    modules = [
        'io', 'generate', 'viz', 'features', 'filters', 'detection', 
        'segmentation', 'operations', 'processing', 'mesh', 'ml',
    ]

    # Dictionary to store functions from each module
    functions_by_module = {}

    for module_name in modules:

        # Dynamically import the module
        module = importlib.import_module(f'qim3d.{module_name}')
        
        # Retrieve all functions listed in the __all__ variable
        functions = [
            getattr(module, name)
            for name in getattr(module, '__all__', [])
            if callable(getattr(module, name))
        ]

        # Only keep functions (not classes) in the list
        functions = [func for func in functions if inspect.isfunction(func)]
        
        # Store the functions in the dictionary
        functions_by_module[module_name] = functions

    return functions_by_module

def exec_python(source):
    """
    Execute a Python code block.
    """
    try:
        exec(source, {"__name__": "__main__"})
    except Exception:
        print(source)
        raise

def merge_code_blocks(code_blocks):
    """
    Merge code blocks such that every time there is an 'import qim3d',
    a new code block starts, and all preceding code blocks are merged into the previous one.
    """
    merged_blocks = []
    current_block = ""

    for block in code_blocks:
        if "import qim3d" in block:

            # If there's an existing block, add it to the merged list
            if current_block.strip():
                merged_blocks.append(current_block.strip())
                
            # Start a new block
            current_block = block
        else:
            # Append to the current block
            current_block += "\n" + block

    # Add the last block if it exists
    if current_block.strip():
        merged_blocks.append(current_block.strip())

    return merged_blocks

def filter_code_blocks(code_blocks):
    """Filter out code blocks that contain bibtex references."""

    filtered_blocks = []

    for block in code_blocks:
        if not any(line.startswith("@") for line in block.splitlines()):
            filtered_blocks.append(block)

    return filtered_blocks

def check_docstring(obj):
    """
    Given a function, test the contents of the docstring.
    Custom function inspired by the mktestdocs.check_docstring function.
    """

    # Get all code blocks from the docstring
    code_blocks = grab_code_blocks(obj.__doc__, lang="")

    # Filter out code blocks that contain bibtex references
    code_blocks = filter_code_blocks(code_blocks)

    # Merge code blocks based on 'import qim3d'
    merged_code_blocks = merge_code_blocks(code_blocks)

    # Execute each merged code block
    for b in merged_code_blocks:
        exec_python(b)