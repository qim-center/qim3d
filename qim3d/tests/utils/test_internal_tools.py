import qim3d
import os
import re


def test_mock_plot():
    fig = qim3d.utils.internal_tools.mock_plot()

    assert fig.get_figwidth() == 5.0


def test_mock_write_file():
    filename = "test.txt"
    content = "test file"
    qim3d.utils.internal_tools.mock_write_file(filename, content=content)

    # Check contents
    with open(filename, "r", encoding="utf-8") as f:
        file_content = f.read()

    # Remove temp file
    os.remove(filename)

    assert content == file_content


def test_get_local_ip():
    def validate_ip(ip_str):
        reg = r"^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$"
        if re.match(reg, ip_str):
            return True
        else:
            return False

    local_ip = qim3d.utils.internal_tools.get_local_ip()

    assert validate_ip(local_ip) == True
