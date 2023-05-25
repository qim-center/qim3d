import socket
import hashlib
import qim3d

def log_test():
    log = qim3d.io.logger.log
    log.debug('This is a debug level message')
    log.info('This is a info level message')
    log.warning('This is a warning level message')
    log.error('This is an error level message')
    log.critical('This is a critical level message')

def mock_plot():
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    xx = np.arange(0, 2 * np.pi, 0.01)
    ax.plot(xx, np.sin(xx))

    return fig


def mock_write_file(path):
    f = open(path, "w")
    f.write("File created by apptools")
    f.close()


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(("192.255.255.255", 1))
        IP = s.getsockname()[0]
    except:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


def port_from_str(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10**4)


def gradio_header(title, port):
    import outputformat as ouf

    ouf.br(2)
    details = [
        f'{ouf.c(title, color="rainbow", cmap="cool", bold=True, return_str=True)}',
        f"Using port {port}",
        f"Running at {get_local_ip()}",
    ]
    ouf.showlist(details, style="box", title="Starting gradio server")


def sizeof(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Y{suffix}"
