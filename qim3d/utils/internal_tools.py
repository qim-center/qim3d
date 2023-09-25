""" Provides a collection of internal utility functions."""

import socket
import hashlib
import outputformat as ouf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import socket



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