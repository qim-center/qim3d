"Helper functions for testing"

import os
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from PIL import Image
import socket
import numpy as np
from qim3d.utils.logger import log


def mock_plot():
    """Creates a mock plot of a sine wave.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.

    Example:
        Creates a mock plot of a sine wave and displays the plot using `plt.show()`.

        >>> fig = mock_plot()
        >>> plt.show()
    """

    matplotlib.use("Agg")

    fig = plt.figure(figsize=(5, 4))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    values = np.arange(0, 2 * np.pi, 0.01)
    axes.plot(values, np.sin(values))

    return fig


def mock_write_file(path, content="File created by qim3d"):
    """
    Creates a file at the specified path and writes a predefined text into it.

    Args:
        path (str): The path to the file to be created.

    Example:
        >>> mock_write_file("example.txt")
    """
    _file = open(path, "w", encoding="utf-8")
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


def temp_data(folder, remove=False, n=3, img_shape=(32, 32)):
    """Creates a temporary folder to test deep learning tools.

    Creates two folders, 'train' and 'test', who each also have two subfolders 'images' and 'labels'.
    n random images are then added to all four subfolders.
    If the 'remove' variable is True, the folders and their content are removed.

    Args:
        folder (str): The path where the folders should be placed.
        remove (bool, optional): If True, all folders are removed from their location.
        n (int, optional): Number of random images and labels in the temporary dataset.
        img_shape (tuple, options): Tuple with the height and width of the images and labels.

    Example:
        >>> tempdata('temporary_folder',n = 10, img_shape = (16,16))
    """
    folder_trte = ["train", "test"]
    sub_folders = ["images", "labels"]

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
    img = Image.fromarray(img)

    if not os.path.exists(path_train):
        os.makedirs(path_train_im)
        os.makedirs(path_test_im)
        os.makedirs(path_train_lab)
        os.makedirs(path_test_lab)
        for i in range(n):
            img.save(path_train_im / f"img_train{i}.png")
            img.save(path_train_lab / f"img_train{i}.png")
            img.save(path_test_im / f"img_test{i}.png")
            img.save(path_test_lab / f"img_test{i}.png")

    if remove:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                log.warning("Failed to delete %s. Reason: %s" % (file_path, e))

        os.rmdir(folder)
