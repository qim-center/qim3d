""" Provides a collection of internal utility functions."""

import getpass
import hashlib
import os
import socket
import numpy as np
import outputformat as ouf
import requests
from scipy.ndimage import zoom
import difflib
import qim3d


def get_local_ip() -> str:
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


def port_from_str(s: str) -> int:
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


def gradio_header(title: str, port: int) -> None:
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


def sizeof(num: float, suffix: str = "B") -> str:
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
        >>> import qim3d
        >>> qim3d.utils.sizeof(1024)
        '1.0 KB'
        >>> qim3d.utils.sizeof(1234567890)
        '1.1 GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Y{suffix}"


def find_similar_paths(path: str) -> list[str]:
    parent_dir = os.path.dirname(path) or "."
    parent_files = os.listdir(parent_dir) if os.path.isdir(parent_dir) else ""
    valid_paths = [os.path.join(parent_dir, file) for file in parent_files]
    similar_paths = difflib.get_close_matches(path, valid_paths)

    return similar_paths


def get_file_size(file_path: str) -> int:
    """
    Args:
    -----
        filename (str): Specifies full path to file

    Returns:
    ---------
        size (int): size of file in bytes
    """
    try:
        file_size = os.path.getsize(file_path)
    except FileNotFoundError:
        similar_paths = qim3d.utils._misc.find_similar_paths(file_path)

        if similar_paths:
            suggestion = similar_paths[0]  # Get the closest match
            message = f"Invalid path. Did you mean '{suggestion}'?"
            raise FileNotFoundError(repr(message))
        else:
            raise FileNotFoundError("Invalid path")
    
    return file_size


def stringify_path(path: os.PathLike) -> str:
    """Converts an os.PathLike object to a string"""
    if isinstance(path, os.PathLike):
        path = path.__fspath__()
    return path


def get_port_dict() -> dict:
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


def get_css() -> str:

    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    css_path = os.path.join(parent_directory, "css", "gradio.css")

    with open(css_path, "r") as file:
        css_content = file.read()

    return css_content


def downscale_img(img: np.ndarray, max_voxels: int = 512**3) -> np.ndarray:
    """Downscale image if total number of voxels exceeds 512³.

    Args:
        img (np.Array): Input image.
        max_voxels (int, optional): Max number of voxels. Defaults to 512³=134217728.

    Returns:
        np.Array: Downscaled image if total number of voxels exceeds 512³.
    """

    # Calculate total number of pixels in the image
    total_voxels = np.prod(img.shape)

    # If total pixels is less than or equal to 512³, return original image
    if total_voxels <= max_voxels:
        return img

    # Calculate zoom factor
    zoom_factor = (max_voxels / total_voxels) ** (1 / 3)

    # Downscale image
    return zoom(img, zoom_factor, order=0)


def scale_to_float16(arr: np.ndarray) -> np.ndarray:
    """
    Scale the input array to the float16 data type.

    Parameters:
    arr (np.ndarray): Input array to be scaled.

    Returns:
    np.ndarray: Scaled array with dtype=np.float16.

    This function scales the input array to the float16 data type, ensuring that the
    maximum value of the array does not exceed the maximum representable value
    for float16. If the maximum value of the input array exceeds the maximum
    representable value for float16, the array is scaled down proportionally
    to fit within the float16 range.
    """

    # Get the maximum value to comprare with the float16 maximum value
    arr_max = np.max(arr)
    float16_max = np.finfo(np.float16).max

    # If the maximum value of the array exceeds the float16 maximum value, scale the array
    if arr_max > float16_max:
        arr = (arr / arr_max) * float16_max

    # Convert the scaled array to float16 data type
    arr = arr.astype(np.float16)

    return arr
