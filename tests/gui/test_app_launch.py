import multiprocessing
import socket
import time

import pytest

import qim3d


def is_server_running(ip, port):
    try:
        with socket.create_connection((ip, int(port)), timeout=1):
            return True
    except OSError:
        return False


def get_free_port(ip):
    with socket.socket() as s:
        # Binding to 0 makes the OS look for any free port
        s.bind((ip, 0))
        return s.getsockname()[1]


def wait_for_server(proc, ip, port, timeout=60, interval=0.2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_server_running(ip, port):
            return True
        if not proc.is_alive():
            return False
        time.sleep(interval)
    return False


def start_server(interface_cls, ip, port):
    app = interface_cls()
    app.launch(server_name=ip, server_port=port)


@pytest.mark.parametrize(
    "interface_cls",
    [
        qim3d.gui.annotation_tool.Interface,
        qim3d.gui.iso3d.Interface,
    ],
    ids=["annotation_tool", "iso3d"],
)
def test_app_launch(interface_cls):
    ip = "localhost"
    port = get_free_port(ip)

    proc = multiprocessing.Process(target=start_server, args=(interface_cls, ip, port))
    proc.start()
    try:
        server_running = wait_for_server(proc, ip, port)
    finally:
        # Stop the server and wait for it to release the port
        proc.terminate()
        proc.join()

    assert server_running is True
