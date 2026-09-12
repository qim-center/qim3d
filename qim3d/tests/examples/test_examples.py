import importlib
from pathlib import Path

import pytest

import qim3d.examples as examples


def test_exposed_example_volumes_match_included_files():
    """Exposed example volume names match the included data files."""
    examples_folder = Path(examples.__file__).parent
    packaged_names = {path.stem for path in examples_folder.glob("*.tif")}

    assert set(examples._volume_names) == packaged_names


def test_unknown_example_volume_raises_attribute_error():
    """Accessing an unknown example volume raises AttributeError."""
    with pytest.raises(AttributeError):
        examples.unknown_volume


def test_example_volume_is_loaded_individually_and_cached(monkeypatch):
    """Accessing a volume loads it once into cache without loading other volumes."""
    examples_namespace = vars(examples)
    io = importlib.import_module("qim3d.io")

    volume_name = "bone_128x128x128"
    volume_path = examples._volume_paths[volume_name]

    # Make the test independent of volumes potentially loaded by other tests.
    for name in examples._volume_names:
        monkeypatch.delitem(examples_namespace, name, raising=False)

    loaded_volume = object()
    load_calls = []

    def fake_load(path):
        load_calls.append(path)
        return loaded_volume

    monkeypatch.setattr(io, "load", fake_load)

    first_result = getattr(examples, volume_name)

    loaded_names = {
        name for name in examples._volume_names if name in examples_namespace
    }

    second_result = getattr(examples, volume_name)

    assert first_result is loaded_volume
    assert second_result is loaded_volume
    assert loaded_names == {volume_name}
    assert load_calls == [volume_path]

    examples_namespace.pop(volume_name)
