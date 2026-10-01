"""Fixtures for the GUI tests"""

import socket

import pytest


@pytest.fixture
def is_server_running():
    """Returns a function that checks whether a server accepts connections on `ip:port`."""

    def _is_server_running(ip, port):
        try:
            with socket.create_connection((ip, int(port)), timeout=1):
                return True
        except OSError:
            return False

    return _is_server_running
