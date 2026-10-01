import os
import re
from pathlib import Path

import qim3d


def test_get_local_ip():
    def validate_ip(ip_str):
        reg = r"^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$"
        if re.match(reg, ip_str):
            return True
        else:
            return False

    local_ip = qim3d.utils._misc.get_local_ip()

    assert validate_ip(local_ip) == True


def test_stringify_path1():
    """Test that the function converts os.PathLike objects to strings"""
    blobs_path = Path(qim3d.__file__).parents[0] / "img_examples" / "blobs_256x256.tif"

    assert str(blobs_path) == qim3d.utils._misc.stringify_path(blobs_path)


def test_stringify_path2():
    """Test that the function returns input unchanged if input is a string"""
    # Create test_path
    test_path = os.path.join("this", "path", "doesnt", "exist.tif")

    assert test_path == qim3d.utils._misc.stringify_path(test_path)
