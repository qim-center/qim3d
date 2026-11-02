"""
Smoke test for built distributions.

Run as a plain script against an installed wheel or sdist (see .github/workflows/publish.yml):

    uv run --isolated --no-project --with dist/*.whl tests/test_smoke.py
"""

from importlib.metadata import version

import numpy as np

import qim3d


def test_version():
    assert version("qim3d")


def test_bundled_example_volume():
    vol = qim3d.examples.bone_128x128x128
    assert vol.shape == (128, 128, 128)


def test_filter():
    vol = np.random.default_rng(0).random((16, 16, 16))
    filtered = qim3d.filters.gaussian(vol, sigma=1)
    assert filtered.shape == vol.shape


if __name__ == "__main__":
    test_version()
    test_bundled_example_volume()
    test_filter()
    print(f"Smoke test passed for qim3d {version('qim3d')} ({qim3d.__file__})")
