""" Provides a collection of internal utility functions."""

import socket
import hashlib
import outputformat as ouf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import shutil
import requests
import getpass
from PIL import Image
from pathlib import Path
from qim3d.io.logger import log
from fastapi import FastAPI
import gradio as gr
from uvicorn import run

def mock_plot():
    """Creates a mock plot of a sine wave.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.

    Example:
        Creates a mock plot of a sine wave and displays the plot using `plt.show()`.

        >>> fig = mock_plot()
        >>> plt.show()
    """

    # TODO: Check if using Agg backend conflicts with other pipelines

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


def get_local_ip():
    """Retrieves the local IP address of the current machine.

    The function uses a socket to determine the local IP address.
    Then, it tries to connect to the IP address "192.255.255.255"

    If the connection attempt is successful, the local IP address is retrieved
    and returned. However, if an exception occurs, which could happen if the
    network is not available, the function falls back to returning
    the loopback address "127.0.0.1".

    Returns:
        str: The local IP address.

    Example usage:
        ip_address = get_local_ip()
    """

    _socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        _socket.connect(("192.255.255.255", 1))
        ip_address = _socket.getsockname()[0]
    except socket.error:
        ip_address = "127.0.0.1"
    finally:
        _socket.close()
    return ip_address


def port_from_str(s):
    """
    Generates a port number from a given string.

    The function uses the SHA-1 hash algorithm generate a hash value from the input string `s`.
    This hash value is then converted to an integer using base 16 (hexadecimal) representation.

    To obtain the port number, the integer hash value is modulo divided by 10,000 (`10**4`).
    This ensures that the generated port number
    falls within the range of valid port numbers (0-65535).

    Args:
        s (str): The input string to generate the port number from.

    Returns:
        int: The generated port number.

    Example usage:
        port = port_from_str("my_specific_app_name")
    """
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10**4)


def gradio_header(title, port):
    """Display the header for a Gradio server.

    Displays a formatted header containing the provided title,
    the port number being used, and the IP address where the server is running.

    Args:
        title (str): The title to be displayed.
        port (int): The port number used by the server.

    Returns:
        None


    Example:
        >>> gradio_header("My gradio app", 4242)
        ╭────────────────────────╮
        │ Starting gradio server │
        ├────────────────────────╯
        ├ My gradio app
        ├ Using port 4242
        ╰ Running at 10.52.0.211

    """

    ouf.br(2)
    details = [
        f'{ouf.c(title, color="rainbow", cmap="cool", bold=True, return_str=True)}',
        f"Using port {port}",
        f"Running at {get_local_ip()}",
    ]
    ouf.showlist(details, style="box", title="Starting gradio server")


def sizeof(num, suffix="B"):
    """Converts a number to a human-readable string representing its size.

    Converts the given number to a human-readable string representing its size in
    a more readable format, such as bytes (B), kilobytes (KB), megabytes (MB),
    gigabytes (GB), etc. The function iterates through a list of unit suffixes
    and divides the number by 1024 until the absolute value of the number is
    less than 1024. It then formats the number and appends the appropriate unit
    suffix.

    Args:
        num (float): The number to be converted.
        suffix (str, optional): The suffix to be appended to the converted size.
            Defaults to "B".

    Returns:
        str: The human-readable string representing the converted size.


    Example:
        >>> sizeof(1024)
        '1.0 KB'
        >>> sizeof(1234567890)
        '1.1 GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Y{suffix}"


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


def stringify_path(path):
    """Converts an os.PathLike object to a string"""
    if isinstance(path, os.PathLike):
        path = path.__fspath__()
    return path


def get_port_dict():
    # Gets user and port
    username = getpass.getuser()
    url = f"https://platform.qim.dk/qim-api/get-port/{username}"

    response = requests.get(url, timeout=10)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response into a Python dictionary
        port_dict = response.json()
    else:
        # Print an error message if the request was not successful
        raise (f"Error: {response.status_code}")

    return port_dict

def run_gradio_app(gradio_interface, host = "0.0.0.0"):

    # Get port using the QIM API
    port_dict = get_port_dict()

    if "gradio_port" in port_dict:
        port = port_dict["gradio_port"]
    elif "port" in port_dict:
        port = port_dict["port"]
    else:
        raise Exception("Port not specified from QIM API")
    
    print(port_dict)
    gradio_header(gradio_interface.title, port)

    # Create FastAPI with mounted gradio interface
    app = FastAPI()
    path = f"/gui/{port_dict['username']}/{port}/"
    app = gr.mount_gradio_app(app, gradio_interface, path=path)

    # Full path
    print(f"http://{host}:{port}{path}")

    # Run the FastAPI server usign uvicorn
    run(app, host=host, port=int(port))


def get_css():

    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    css_path = os.path.join(parent_directory,"css","gradio.css")
    
    with open(css_path,'r') as file:
        css_content = file.read()
    
    return css_content